"""Batched PyTorch implementation of the RefVil gradient flows.

The leading tensor dimension is a batch of statistically independent saved
configurations. It is the shared CPU/CUDA flow engine used by the active
online pipeline and the legacy numerical-parity tests.
"""

import math

import numpy as np
import torch


ENGINE_VERSION = "torch_flow_v1"


def _log_i0(x):
    return torch.log(torch.special.i0e(x)) + torch.abs(x)


def _i1_over_i0(x):
    return torch.special.i1e(x) / torch.special.i0e(x)


def _roll(value, shift, dim):
    return torch.roll(value, shifts=shift, dims=dim)


class TorchCPNFlowBatch:
    """Flow a batch of configurations using float64/complex128 tensors."""

    def __init__(self, z, a, s, params, kind, device):
        if kind not in ("covariant", "halfRefVil", "RefVil_fix_s"):
            raise ValueError(f"unsupported flow kind: {kind}")
        self.kind = kind
        self.device = torch.device(device)
        self.z = torch.as_tensor(z, dtype=torch.complex128, device=self.device).clone()
        self.a = torch.as_tensor(a, dtype=torch.float64, device=self.device).clone()
        self.s = (None if s is None else
                  torch.as_tensor(s, dtype=torch.int64, device=self.device).clone())
        if self.z.ndim != 4 or self.a.ndim != 4:
            raise ValueError("batched z and a must have shapes (batch,Lx,Ly,N/2)")
        self.batch, self.Lx, self.Ly, self.N = self.z.shape
        if self.a.shape != (self.batch, self.Lx, self.Ly, 2):
            raise ValueError("a shape is inconsistent with z")
        if kind == "RefVil_fix_s" and (self.s is None or self.s.shape != self.z.shape[:3]):
            raise ValueError("RefVil_fix_s requires one integer s field per configuration")
        self.V = self.Lx * self.Ly
        self.epsilon = float(params["flow_epsilon"])
        self.mass_a = float(params["mass_a"])
        self.mass_z = float(params["mass_z"])

        if kind == "covariant":
            beta, beta1, alpha, alpha1 = 1.0, 0.0, 0.0, 0.0
        else:
            beta = float(params["beta"])
            beta1 = float(params["beta1"])
            alpha = float(params.get("alpha", 0.0))
            alpha1 = float(params["alpha1"])
        norm = (beta + beta1) * 2.0 * self.N
        if abs(norm) < 1e-14:
            raise ValueError("flow normalization is zero")
        self.beta = beta / norm
        self.beta1 = beta1 / norm
        self.alpha = alpha / norm
        self.alpha1 = alpha1 / norm
        self._sync_U_from_a()

    def _sync_U_from_a(self):
        self.U = torch.exp(1j * self.a)

    def _normalize_z_inplace(self):
        self.z = self.z / torch.linalg.vector_norm(self.z, dim=-1, keepdim=True)

    @staticmethod
    def _project_tangent(z, field):
        overlap = torch.sum(torch.conj(z) * field, dim=-1)
        return field - torch.real(overlap)[..., None] * z

    def _spin_inner_x(self):
        return torch.sum(torch.conj(self.z) * _roll(self.z, -1, 1), dim=-1)

    def _spin_inner_y(self):
        return torch.sum(torch.conj(self.z) * _roll(self.z, -1, 2), dim=-1)

    def _plaquette(self):
        Ux, Uy = self.U[..., 0], self.U[..., 1]
        return Ux * _roll(Uy, -1, 1) * torch.conj(_roll(Ux, -1, 2) * Uy)

    def _plaquette_da(self):
        if self.s is None:
            raise RuntimeError("integer plaquette field is unavailable")
        return (self.a[..., 0] + _roll(self.a, -1, 1)[..., 1]
                - _roll(self.a, -1, 2)[..., 0] - self.a[..., 1]
                + (2.0 * math.pi) * self.s.to(dtype=self.a.dtype))

    def _improved_action(self, inner_x, inner_y, mod):
        if mod == 0:
            return self.N * self.beta1 * (
                torch.abs(inner_x) ** 2 + torch.abs(inner_y) ** 2 - 2.0)
        if mod == 1:
            ref_arg = torch.as_tensor(2.0 * self.N * self.beta1,
                                      dtype=torch.float64, device=self.device)
            ref = _log_i0(ref_arg)
            sign = float(np.sign(self.beta1))
            return sign * (_log_i0(ref_arg * torch.abs(inner_x)) - ref
                           + _log_i0(ref_arg * torch.abs(inner_y)) - ref)
        raise ValueError(f"unsupported mod={mod}; expected 0 or 1")

    def action(self, mod=0):
        inner_x = torch.sum(torch.conj(_roll(self.z, -1, 1)) * self.z, dim=-1)
        inner_y = torch.sum(torch.conj(_roll(self.z, -1, 2)) * self.z, dim=-1)
        cp_term = torch.real(self.U[..., 0] * inner_x + self.U[..., 1] * inner_y)
        cp_term_1 = self._improved_action(inner_x, inner_y, mod)
        if self.kind == "RefVil_fix_s":
            plaq = self._plaquette_da()
            density = (-2.0 * self.N * self.beta * (cp_term - 2.0) - cp_term_1
                       + 0.5 * self.alpha * plaq ** 2
                       - self.alpha1 * (torch.cos(plaq) - 1.0))
        else:
            density = (-2.0 * self.N * self.beta * (cp_term - 2.0) - cp_term_1
                       - self.alpha1 * (torch.real(self._plaquette()) - 1.0))
        return torch.sum(density, dim=(1, 2))

    @staticmethod
    def _safe_phase(inner):
        magnitude = torch.abs(inner)
        return torch.where(magnitude > 0, inner / magnitude, torch.zeros_like(inner))

    def _z_force(self, mod=0):
        z, U = self.z, self.U
        inner_x, inner_y = self._spin_inner_x(), self._spin_inner_y()
        force = torch.zeros_like(z)
        improved = torch.zeros_like(z)
        if self.Lx > 1:
            force += _roll(U[..., 0], 1, 1)[..., None] * _roll(z, 1, 1)
            force += torch.conj(U[..., 0])[..., None] * _roll(z, -1, 1)
            if mod == 0:
                temp_x = inner_x * self.beta1
            elif mod == 1:
                arg = 2.0 * self.N * self.beta1 * torch.abs(inner_x)
                temp_x = _i1_over_i0(arg) * self._safe_phase(inner_x) * abs(self.beta1)
            else:
                raise ValueError(f"unsupported mod={mod}; expected 0 or 1")
            improved += _roll(temp_x, 1, 1)[..., None] * _roll(z, 1, 1)
            improved += torch.conj(temp_x)[..., None] * _roll(z, -1, 1)
        if self.Ly > 1:
            force += _roll(U[..., 1], 1, 2)[..., None] * _roll(z, 1, 2)
            force += torch.conj(U[..., 1])[..., None] * _roll(z, -1, 2)
            if mod == 0:
                temp_y = inner_y * self.beta1
            elif mod == 1:
                arg = 2.0 * self.N * self.beta1 * torch.abs(inner_y)
                temp_y = _i1_over_i0(arg) * self._safe_phase(inner_y) * abs(self.beta1)
            else:
                raise ValueError(f"unsupported mod={mod}; expected 0 or 1")
            improved += _roll(temp_y, 1, 2)[..., None] * _roll(z, 1, 2)
            improved += torch.conj(temp_y)[..., None] * _roll(z, -1, 2)
        return 2.0 * self.N * (self.beta * force + improved)

    def _half_a_force(self):
        Ux, Uy = self.U[..., 0], self.U[..., 1]
        force = torch.zeros_like(self.U)
        force[..., 0] = 2.0 * self.N * self.beta * self._spin_inner_x()
        force[..., 1] = 2.0 * self.N * self.beta * self._spin_inner_y()
        if self.alpha1 != 0.0:
            if self.Ly > 1:
                staple_1 = torch.conj(_roll(Uy, -1, 1)) * _roll(Ux, -1, 2) * Uy
                staple_2 = (_roll(Uy, (-1, 1), (1, 2)) * _roll(Ux, 1, 2)
                            * torch.conj(_roll(Uy, 1, 2)))
                force[..., 0] += self.alpha1 * (staple_1 + staple_2)
            if self.Lx > 1:
                staple_1 = torch.conj(_roll(Ux, -1, 2)) * _roll(Uy, -1, 1) * Ux
                staple_2 = (_roll(Ux, (1, -1), (1, 2)) * _roll(Uy, 1, 1)
                            * torch.conj(_roll(Ux, 1, 1)))
                force[..., 1] += self.alpha1 * (staple_1 + staple_2)
        return torch.imag(torch.conj(self.U) * force)

    def _fixed_s_a_force(self):
        Ux, Uy = self.U[..., 0], self.U[..., 1]
        force = torch.zeros_like(self.a)
        force[..., 0] += 2.0 * self.N * self.beta * torch.imag(
            torch.conj(Ux) * self._spin_inner_x())
        force[..., 1] += 2.0 * self.N * self.beta * torch.imag(
            torch.conj(Uy) * self._spin_inner_y())
        f = self._plaquette_da()
        sinf = torch.sin(f)
        force[..., 0] -= (self.alpha * (f - _roll(f, 1, 2))
                          + self.alpha1 * (sinf - _roll(sinf, 1, 2)))
        force[..., 1] -= (self.alpha * (_roll(f, 1, 1) - f)
                          + self.alpha1 * (_roll(sinf, 1, 1) - sinf))
        return force

    def _apply_periodicity_wrap(self):
        wrapped = torch.remainder(self.a + math.pi, 2.0 * math.pi) - math.pi
        delta = torch.round((self.a - wrapped) / (2.0 * math.pi)).to(torch.int64)
        self.a = wrapped
        self.s += (delta[..., 0] + _roll(delta, -1, 1)[..., 1]
                   - _roll(delta, -1, 2)[..., 0] - delta[..., 1])

    def flow_step(self, mod=0):
        force_a = self._fixed_s_a_force() if self.kind == "RefVil_fix_s" else self._half_a_force()
        force_z = self._project_tangent(self.z, self._z_force(mod=mod))
        self.a = self.a + self.epsilon * force_a / self.mass_a
        if self.kind != "RefVil_fix_s":
            self.a = torch.remainder(self.a + math.pi, 2.0 * math.pi) - math.pi
        self._sync_U_from_a()
        norm = torch.linalg.vector_norm(force_z, dim=-1)
        safe_norm = torch.where(norm < 1e-10, torch.ones_like(norm), norm)
        angle = self.epsilon * safe_norm / self.mass_z
        self.z = (torch.cos(angle)[..., None] * self.z
                  + torch.sin(angle)[..., None] * force_z / safe_norm[..., None])
        self._normalize_z_inplace()
        if self.kind == "RefVil_fix_s":
            self._apply_periodicity_wrap()
            self._sync_U_from_a()

    def PP_corr_k(self):
        projector = self.z[..., :, None] * torch.conj(self.z[..., None, :])
        eye = torch.eye(self.N, dtype=torch.complex128, device=self.device)
        centered = projector - eye[None, None, None, :, :] / self.N
        transformed = torch.fft.fftn(centered, dim=(1, 2))
        return torch.sum(torch.abs(transformed) ** 2, dim=(3, 4)) / self.V

    def topo_charge_z(self):
        z = self.z

        def inner(left, right):
            return torch.sum(torch.conj(left) * right, dim=-1)

        z_x = _roll(z, -1, 1)
        z_y = _roll(z, -1, 2)
        z_xy = _roll(z_x, -1, 2)
        temp0 = inner(z, z_x) * inner(z_x, z_xy) * inner(z_xy, z)
        temp1 = inner(z, z_xy) * inner(z_xy, z_y) * inner(z_y, z)
        return torch.sum(torch.angle(temp0) + torch.angle(temp1), dim=(1, 2)) / (2.0 * math.pi)

    def topo_charge_u(self):
        return torch.sum(torch.angle(self._plaquette()), dim=(1, 2)) / (2.0 * math.pi)

    def measure(self, mod=0):
        modes = self.PP_corr_k()
        return torch.stack((self.action(mod=mod) / self.V, modes[:, 0, 0],
                            modes[:, 1, 0], modes[:, 0, 1], self.topo_charge_z(),
                            self.topo_charge_u()), dim=1)


def flow_batch(z, a, s, params, device):
    """Return NumPy observables for one CUDA-resident configuration batch."""
    if params["do_covariant"]:
        kind = "covariant"
    elif abs(params["alpha"]) < 1e-8:
        kind = "halfRefVil"
    else:
        kind = "RefVil_fix_s"
    flow = TorchCPNFlowBatch(z, a, s, params, kind, device)
    output_steps = np.asarray(params["output_steps"], dtype=int)
    values = torch.empty((len(z), len(output_steps), 6), dtype=torch.float64, device=device)
    values[:, 0] = flow.measure(mod=params["mod"])
    old_action = values[:, 0, 0] * flow.V
    violations = torch.zeros(len(z), dtype=torch.int64, device=device)
    max_increase = torch.zeros(len(z), dtype=torch.float64, device=device)
    previous = int(output_steps[0])
    with torch.no_grad():
        for out_index in range(1, len(output_steps)):
            target = int(output_steps[out_index])
            for _ in range(target - previous):
                flow.flow_step(mod=params["mod"])
            measured = flow.measure(mod=params["mod"])
            values[:, out_index] = measured
            new_action = measured[:, 0] * flow.V
            increase = new_action - old_action
            bad = increase > 1e-10 * torch.maximum(torch.ones_like(old_action), torch.abs(old_action))
            violations += bad.to(torch.int64)
            max_increase = torch.maximum(max_increase, torch.where(bad, increase, torch.zeros_like(increase)))
            old_action, previous = new_action, target
    return (values.cpu().numpy(), kind, int(violations.sum().cpu()),
            float(max_increase.max().cpu()))
