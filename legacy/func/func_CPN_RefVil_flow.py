import numpy as np
import scipy
from numpy.fft import fftn, ifftn

def _log_i0(x):
    """Stable log(I0(x)) for scalar or ndarray input."""
    x = np.asarray(x)
    return np.log(scipy.special.i0e(x)) + np.abs(x)


def _i1_over_i0(x):
    """Stable I1(x) / I0(x) for scalar or ndarray input."""
    x = np.asarray(x)
    return scipy.special.ive(1, x) / scipy.special.i0e(x)

def vdot_z(z1, z2):
    return np.einsum("ijk,ijk->ij", np.conj(z1), z2)

def vouter_z(z1, z2):
    return np.einsum("ijk,ijl->ijkl", z1, np.conj(z2))

def U1_clip(x):
    return np.mod(x+np.pi, 2*np.pi) - np.pi

class CPN_halfRefVil_flow:
    """
    CP^{N-1} + U(1) gradient flow.

    - z field: shape (Lx, Ly, N), complex, with per-site norm |z|^2 = 1.
    - U field: shape (Lx, Ly, 2), complex phases on links.
    - a field: real phases such that U = exp(1j * a).

    Action is half-Villainized improved form with beta1 * |z' z|^2 term. Action is normalized so that 2N(beta + beta1) = 1
    """

    def __init__(
        self,
        Lx,
        Ly,
        N,
        beta,
        beta1,
        alpha1,
        seed=None,
        epsilon=0.05,
        n_step=20,
        mass_a=1.0,
        mass_z=1.0,
    ):
        self.Lx = Lx
        self.Ly = Ly
        if N <= 1:
            raise Exception("N too small")
        self.N = N
        norm = (beta + beta1) * 2 * N
        self.beta = beta / norm
        self.beta1 = beta1 / norm
        self.alpha1 = alpha1 / norm
        self.V = Lx * Ly

        self.epsilon = float(epsilon)
        self.n_step = int(n_step)
        self.mass_a = float(mass_a)
        self.mass_z = float(mass_z)

        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.n_step <= 0:
            raise ValueError("n_step must be positive")
        if self.mass_a <= 0 or self.mass_z <= 0:
            raise ValueError("mass_a and mass_z must be positive")

        if seed is not None:
            np.random.seed(seed)

        self.z = self._random_unit_vectors((Lx, Ly, N))
        self.a = 2 * np.pi * np.random.rand(Lx, Ly, 2)
        self._wrap_phase_inplace()
        self._sync_U_from_a()

    @property
    def accept_rate(self):
        if self.attempted == 0:
            return 0.0
        return self.accepted / self.attempted

    def _random_unit_vectors(self, shape):
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
        v = real + 1j * imag
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / norm

    def _wrap_phase_inplace(self):
        self.a = (self.a + np.pi) % (2 * np.pi) - np.pi

    def _sync_U_from_a(self):
        self.U = np.exp(1j * self.a)

    def _normalize_z_inplace(self):
        norm = np.linalg.norm(self.z, axis=-1, keepdims=True)
        self.z /= norm

    def _project_tangent(self, z, field):
        """
        Project complex field onto tangent space of |z|^2=1 manifold.
        Constraint is Re(<z, field>) = 0 at each site.
        """
        overlap = np.einsum("ijk,ijk->ij", np.conj(z), field)
        return field - np.real(overlap)[:, :, None] * z

    def _spin_inner_x(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=0))

    def _spin_inner_y(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=1))

    def _plaquette(self):
        Ux = self.U[:, :, 0]
        Uy = self.U[:, :, 1]
        return Ux * np.roll(Uy, -1, axis=0) * np.conj(np.roll(Ux, -1, axis=1) * Uy)

    def action_density(self, mod=0):
        z = self.z
        U = self.U
        V = self.V

        cp_term = np.real(U[:, :, 0] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z))
        cp_term += np.real(U[:, :, 1] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z))
        inner_x = np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z)
        inner_y = np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z)
        if mod == 0:
            cp_term_1 = self.N * self.beta1 * (
                np.abs(inner_x) ** 2 + np.abs(inner_y) ** 2 - 2
            )
        elif mod == 1:
            ref = _log_i0(2 * self.N * self.beta1)
            cp_term_1 = np.sign(self.beta1) * (
                _log_i0(2 * self.N * self.beta1 * np.abs(inner_x)) - ref
                + _log_i0(2 * self.N * self.beta1 * np.abs(inner_y)) - ref
            )
        else:
            raise ValueError(f"unsupported mod={mod}; expected 0 or 1")
        plaq_term = np.real(self._plaquette())

        return -2.0 * self.N * self.beta * (cp_term - 2) - cp_term_1 - self.alpha1 * (plaq_term - 1)
    
    def action_density_plaq(self, mod=0):
        z = self.z
        U = self.U
        V = self.V

        cp_term_x = np.real(U[:, :, 0] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z))
        cp_term_y = np.real(U[:, :, 1] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z))
        inner_x = np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z)
        inner_y = np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z)
        if mod == 0:
            cp_term_1_x = self.N * self.beta1 * (np.abs(inner_x) ** 2 - 1)
            cp_term_1_y = self.N * self.beta1 * (np.abs(inner_y) ** 2 - 1)
        elif mod == 1:
            ref = _log_i0(2 * self.N * self.beta1)
            cp_term_1_x = np.sign(self.beta1) * (
                _log_i0(2 * self.N * self.beta1 * np.abs(inner_x)) - ref
            )
            cp_term_1_y = np.sign(self.beta1) * (
                _log_i0(2 * self.N * self.beta1 * np.abs(inner_y)) - ref
            )
        else:
            raise ValueError(f"unsupported mod={mod}; expected 0 or 1")
        plaq_term = np.real(self._plaquette())

        S0 = -self.N * self.beta * (cp_term_x + np.roll(cp_term_x, -1, axis=1) + cp_term_y + np.roll(cp_term_y, -1, axis=0) - 4)
        S1 = -(cp_term_1_x + np.roll(cp_term_1_x, -1, axis=1) + cp_term_1_y + np.roll(cp_term_1_y, -1, axis=0)) / 2.0

        return S0 + S1 - self.alpha1 * (plaq_term - 1)

    def action(self, mod=0):
        return np.sum(self.action_density(mod=mod))

    def _z_force(self, mod=0):
        """
        Returns force F_z = -dS/d(conj(z)) * 2.
        """
        z = self.z
        U = self.U

        F = np.zeros_like(z)
        F1 = np.zeros_like(z)
        inner_x = self._spin_inner_x()
        inner_y = self._spin_inner_y()
        if self.Lx > 1:
            F += np.roll(U[:, :, 0], 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0)
            F += np.conj(U[:, :, 0])[:, :, None] * np.roll(z, -1, axis=0)
            if mod == 0:
                F1 += np.roll(inner_x, 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0) * self.beta1
                F1 += np.conj(inner_x)[:, :, None] * np.roll(z, -1, axis=0) * self.beta1
            elif mod == 1:
                abs_x = np.abs(inner_x)
                ratio_x = _i1_over_i0(2 * self.N * self.beta1 * abs_x)
                phase_x = np.divide(inner_x, abs_x, out=np.zeros_like(inner_x), where=abs_x > 0)
                temp_x = ratio_x * phase_x
                F1 += np.roll(temp_x, 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0) * np.abs(self.beta1)
                F1 += np.conj(temp_x)[:, :, None] * np.roll(z, -1, axis=0) * np.abs(self.beta1)
            else:
                raise ValueError(f"unsupported mod={mod}; expected 0 or 1")
        if self.Ly > 1:
            F += np.roll(U[:, :, 1], 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1)
            F += np.conj(U[:, :, 1])[:, :, None] * np.roll(z, -1, axis=1)
            if mod == 0:
                F1 += np.roll(inner_y, 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1) * self.beta1
                F1 += np.conj(inner_y)[:, :, None] * np.roll(z, -1, axis=1) * self.beta1
            elif mod == 1:
                abs_y = np.abs(inner_y)
                ratio_y = _i1_over_i0(2 * self.N * self.beta1 * abs_y)
                phase_y = np.divide(inner_y, abs_y, out=np.zeros_like(inner_y), where=abs_y > 0)
                temp_y = ratio_y * phase_y
                F1 += np.roll(temp_y, 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1) * np.abs(self.beta1)
                F1 += np.conj(temp_y)[:, :, None] * np.roll(z, -1, axis=1) * np.abs(self.beta1)
            else:
                raise ValueError(f"unsupported mod={mod}; expected 0 or 1")

        return self.N * self.beta * F * 2.0 + self.N * F1 * 2.0

    def _link_force_complex(self):
        """
        Returns complex F for each link such that local action is
        S_link = -Re(conj(U) * F).
        """
        z = self.z
        Ux = self.U[:, :, 0]
        Uy = self.U[:, :, 1]

        F = np.zeros((self.Lx, self.Ly, 2), dtype=complex)

        F[:, :, 0] += 2.0 * self.N * self.beta * self._spin_inner_x()
        F[:, :, 1] += 2.0 * self.N * self.beta * self._spin_inner_y()

        if self.alpha1 != 0.0:
            if self.Ly > 1:
                staple_x_1 = np.conj(np.roll(Uy, -1, axis=0)) * np.roll(Ux, -1, axis=1) * Uy
                staple_x_2 = np.roll(Uy, (-1, 1), axis=(0, 1)) * np.roll(Ux, 1, axis=1) * np.conj(np.roll(Uy, 1, axis=1))
                F[:, :, 0] += self.alpha1 * (staple_x_1 + staple_x_2)
            if self.Lx > 1:
                staple_y_1 = np.conj(np.roll(Ux, -1, axis=1)) * np.roll(Uy, -1, axis=0) * Ux
                staple_y_2 = np.roll(Ux, (1, -1), axis=(0, 1)) * np.roll(Uy, 1, axis=0) * np.conj(np.roll(Ux, 1, axis=0))
                F[:, :, 1] += self.alpha1 * (staple_y_1 + staple_y_2)

        return F

    def _a_force(self):
        """
        Returns force for momentum equation da/dt = p_a/mass_a,
        dp_a/dt = force_a = -dS/da.
        """
        F_complex = self._link_force_complex()
        return np.imag(np.conj(self.U) * F_complex)

    def flow_step(self, epsilon=None, n_step=None, mod=0):
        '''
        Perform one flow step with step size epsilon and n_step steps.
        '''
        eps = self.epsilon if epsilon is None else float(epsilon)
        nsteps = self.n_step if n_step is None else int(n_step)
        if eps <= 0:
            raise ValueError("epsilon must be positive")
        if nsteps <= 0:
            raise ValueError("n_step must be positive")
        # action_old = self.action()

        for step in range(nsteps):
            force_a = self._a_force()
            force_z = self._project_tangent(self.z, self._z_force(mod=mod))
            # flow a
            self.a += eps * force_a / self.mass_a
            self._wrap_phase_inplace()
            self._sync_U_from_a()
            # flow z
            force_z_norm = np.linalg.norm(force_z, axis=-1)
            # check zero norm
            zero_mask = force_z_norm < 1e-10
            force_z_norm[zero_mask] = 1.0
            cos_force_z = np.cos(eps * force_z_norm / self.mass_z)
            sin_force_z = np.sin(eps * force_z_norm / self.mass_z)
            force_z_unit = force_z / force_z_norm[:, :, None]
            self.z = cos_force_z[:,:,None] * self.z + sin_force_z[:,:,None] * force_z_unit
            self._normalize_z_inplace()

        # action_new = self.action()
        # return action_new - action_old
    
    # ----------------------------------- Observables -----------------------------------

    def PP_corr_k(self):
        """
        Return fourier transformation of connected PP correlation.
        """
        N = self.N
        P = np.einsum("ijk,ijl->ijkl", self.z, np.conj(self.z))
        Px = P - 1 / N * np.eye(N)[None, None, :, :]
        Pk = fftn(Px, axes=(0, 1))
        S = np.sum(np.abs(Pk) ** 2, axis=(2, 3)) / self.V
        return S

    def topo_charge_z(self):
        """
        Skyrmion number of the system, using z field only.
        """
        z = self.z

        temp0 = (
            vdot_z(z, np.roll(z, -1, 0))
            * vdot_z(np.roll(z, -1, 0), np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), z)
        )
        temp1 = (
            vdot_z(z, np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), np.roll(z, -1, 1))
            * vdot_z(np.roll(z, -1, 1), z)
        )
        Q_z = np.angle(temp0) + np.angle(temp1)

        return np.sum(Q_z) / (2 * np.pi)

class CPN_RefVil_flow_fix_s:
    """
    CP^{N-1} + U(1) gradient flow.

    - z field: shape (Lx, Ly, N), complex, with per-site norm |z|^2 = 1.
    - U field: shape (Lx, Ly, 2), complex phases on links.
    - a field: real phases such that U = exp(1j * a).
    - s field: shape (Lx, Ly), integer on plaquettes, fixed during flow.

    Action is improved form with beta1 * |z' z|^2 term or log I0(beta1 * |z' z|) term,
    determined by mod = 0, 1.

    Action is Villainized, with alpha / 2 * (da + 2 pi s)^2 and -alpha1 * cos(da) term. Action is normalized so that beta + beta1 = 1
    """

    def __init__(
        self,
        N,
        Lx,
        Ly,
        beta,
        beta1,
        alpha,
        alpha1,
        epsilon=0.05,
        n_step=20,
        mass_a=1.0,
        mass_z=1.0,
    ):
        self.Lx = Lx
        self.Ly = Ly
        if N <= 1:
            raise Exception("N too small")
        self.N = N
        norm = (beta + beta1) * 2 * N
        self.beta = beta / norm
        self.beta1 = beta1 / norm
        self.alpha = alpha / norm
        self.alpha1 = alpha1 / norm
        self.V = Lx * Ly

        self.epsilon = float(epsilon)
        self.n_step = int(n_step)
        self.mass_a = float(mass_a)
        self.mass_z = float(mass_z)

        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.n_step <= 0:
            raise ValueError("n_step must be positive")
        if self.mass_a <= 0 or self.mass_z <= 0:
            raise ValueError("mass_a and mass_z must be positive")

        self.z = np.concatenate([np.ones((Lx, Ly, 1), dtype=complex), np.zeros((Lx, Ly, N-1), dtype=complex)], axis=-1)
        self.a = 2 * np.pi * np.zeros((Lx, Ly, 2))
        self.s = np.zeros((Lx, Ly), dtype=int)
        self._sync_U_from_a()

    def _apply_periodicity_wrap(self):
        """Wrap a into [-pi, pi] and shift the integer field s on every plaquette
        so that the physical plaquette variable da + 2*pi*s is unchanged.

        PBC version: one plaquette per site, neighbours wrap periodically.
        """
        delta_a = np.rint((self.a - U1_clip(self.a)) / (2 * np.pi)).astype(int)
        self.a = U1_clip(self.a)
        self.s += (delta_a[:, :, 0]
                   + np.roll(delta_a, -1, axis=0)[:, :, 1]
                   - np.roll(delta_a, -1, axis=1)[:, :, 0]
                   - delta_a[:, :, 1])

    def _sync_U_from_a(self):
        self.U = np.exp(1j * self.a)

    def _normalize_z_inplace(self):
        norm = np.linalg.norm(self.z, axis=-1, keepdims=True)
        self.z /= norm

    def _project_tangent(self, z, field):
        """
        Project complex field onto tangent space of |z|^2=1 manifold.
        Constraint is Re(<z, field>) = 0 at each site.
        """
        overlap = np.einsum("ijk,ijk->ij", np.conj(z), field)
        return field - np.real(overlap)[:, :, None] * z

    def _spin_inner_x(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=0))

    def _spin_inner_y(self):
        return np.einsum("ijk,ijk->ij", np.conj(self.z), np.roll(self.z, -1, axis=1))

    def _plaquette_da(self):
        a = self.a
        f = a[:,:,0] + np.roll(a, -1, axis=0)[:,:,1] - np.roll(a, -1, axis=1)[:,:,0] - a[:,:,1] + 2 * np.pi * self.s
        return f

    def action_density(self, mod=0):
        z = self.z
        U = self.U

        cp_term = np.real(U[:, :, 0] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z))
        cp_term += np.real(U[:, :, 1] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z))
        if mod == 0:
            cp_term_1 = np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z)) ** 2 - 1
            cp_term_1 += np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z)) ** 2 - 1
            cp_term_1 *= self.N * self.beta1
        elif mod == 1:
            ref = _log_i0(2 * self.N * self.beta1)
            cp_term_1 = np.sign(self.beta1) * (_log_i0(
                np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z))
                * 2 * self.N * self.beta1
            ) - ref)
            cp_term_1 += np.sign(self.beta1) * (_log_i0(
                np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z))
                * 2 * self.N * self.beta1
            ) - ref)
        plaq = self._plaquette_da()
        plaq_term = (plaq ** 2) / 2
        plaq_term_1 = np.cos(plaq)

        return -2.0 * self.N * self.beta * (cp_term - 2) - cp_term_1 + self.alpha * (plaq_term) - self.alpha1 * (plaq_term_1 - 1)
    
    def action_density_plaq(self, mod=0):
        z = self.z
        U = self.U

        cp_term_x = np.real(U[:, :, 0] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z))
        cp_term_y = np.real(U[:, :, 1] * np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z))
        if mod == 0:
            cp_term_1_x = np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z)) ** 2 - 1
            cp_term_1_y = np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z)) ** 2 - 1
            cp_term_1_x *= self.N * self.beta1
            cp_term_1_y *= self.N * self.beta1
        elif mod == 1:
            ref = _log_i0(2 * self.N * self.beta1)
            cp_term_1_x = np.sign(self.beta1) * (_log_i0(
                np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=0)), z))
                * 2 * self.N * self.beta1
            ) - ref)
            cp_term_1_y = np.sign(self.beta1) * (_log_i0(
                np.abs(np.einsum("ijk,ijk->ij", np.conj(np.roll(z, -1, axis=1)), z))
                * 2 * self.N * self.beta1
            ) - ref)
        plaq = self._plaquette_da()
        plaq_term = (plaq ** 2) / 2
        plaq_term_1 = np.cos(plaq)

        S0 = -self.N * self.beta * (cp_term_x + np.roll(cp_term_x, -1, axis=1) + cp_term_y + np.roll(cp_term_y, -1, axis=0) - 4)
        S1 = -(cp_term_1_x + np.roll(cp_term_1_x, -1, axis=1) + cp_term_1_y + np.roll(cp_term_1_y, -1, axis=0)) / 2.0

        return S0 + S1 + self.alpha * (plaq_term) - self.alpha1 * (plaq_term_1 - 1)

    def action(self, mod=0):
        return np.sum(self.action_density(mod=mod))

    def _z_force(self, mod=0):
        """
        Returns force F_z = -dS/d(conj(z)) * 2.
        """
        z = self.z
        U = self.U
        N = self.N
        beta = self.beta
        beta1 = self.beta1

        F = np.zeros_like(z)
        F1 = np.zeros_like(z)
        inner_x = self._spin_inner_x()
        inner_y = self._spin_inner_y()
        if self.Lx > 1:
            F += np.roll(U[:, :, 0], 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0)
            F += np.conj(U[:, :, 0])[:, :, None] * np.roll(z, -1, axis=0)
            if mod == 0:
                F1 += np.roll(inner_x, 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0) * beta1
                F1 += np.conj(inner_x)[:, :, None] * np.roll(z, -1, axis=0) * beta1
            elif mod == 1:
                inner_x_abs = np.abs(inner_x)
                ratio = _i1_over_i0(inner_x_abs * 2 * N * beta1)
                phase = np.divide(inner_x, inner_x_abs, out=np.zeros_like(inner_x), where=inner_x_abs > 0)
                temp = ratio * phase
                F1 += np.roll(temp, 1, axis=0)[:, :, None] * np.roll(z, 1, axis=0) * np.abs(beta1)
                F1 += np.conj(temp)[:, :, None] * np.roll(z, -1, axis=0) * np.abs(beta1)

        if self.Ly > 1:
            F += np.roll(U[:, :, 1], 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1)
            F += np.conj(U[:, :, 1])[:, :, None] * np.roll(z, -1, axis=1)
            if mod == 0:
                F1 += np.roll(inner_y, 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1) * beta1
                F1 += np.conj(inner_y)[:, :, None] * np.roll(z, -1, axis=1) * beta1
            elif mod == 1:
                inner_y_abs = np.abs(inner_y)
                ratio = _i1_over_i0(inner_y_abs * 2 * N * beta1)
                phase = np.divide(inner_y, inner_y_abs, out=np.zeros_like(inner_y), where=inner_y_abs > 0)
                temp = ratio * phase
                F1 += np.roll(temp, 1, axis=1)[:, :, None] * np.roll(z, 1, axis=1) * np.abs(beta1)
                F1 += np.conj(temp)[:, :, None] * np.roll(z, -1, axis=1) * np.abs(beta1)

        return N * beta * F * 2.0 + N * F1 * 2.0
    
    def _a_force(self):
        """
        Returns force for momentum equation da/dt = p_a/mass_a,
        dp_a/dt = force_a = -dS/da.
        """
        a = self.a
        s = self.s
        alpha = self.alpha
        alpha1 = self.alpha1
        Ux = self.U[:, :, 0]
        Uy = self.U[:, :, 1]

        F = np.zeros((self.Lx, self.Ly, 2), dtype=complex)

        F[:, :, 0] += 2.0 * self.N * self.beta * np.imag(np.conj(Ux) * self._spin_inner_x())
        F[:, :, 1] += 2.0 * self.N * self.beta * np.imag(np.conj(Uy) * self._spin_inner_y())

        # ---------- Villain plaquette contribution ----------
        f = (a[:, :, 0]
            + np.roll(a, -1, axis=0)[:, :, 1]
            - np.roll(a, -1, axis=1)[:, :, 0]
            - a[:, :, 1]
            + 2 * np.pi * s)
        sinf = np.sin(f)

        # derivatives
        F[:, :, 0] -= (alpha * (f - np.roll(f, 1, axis=1)) + alpha1 * (sinf - np.roll(sinf, 1, axis=1)))
        F[:, :, 1] -= (alpha * (np.roll(f, 1, axis=0) - f) + alpha1 * (np.roll(sinf, 1, axis=0) - sinf))

        return F.real

    def flow_step(self, epsilon=None, n_step=None, mod=0):
        '''
        Perform one flow step with step size epsilon and n_step steps.
        '''
        eps = self.epsilon if epsilon is None else float(epsilon)
        nsteps = self.n_step if n_step is None else int(n_step)
        if eps <= 0:
            raise ValueError("epsilon must be positive")
        if nsteps <= 0:
            raise ValueError("n_step must be positive")

        for step in range(nsteps):
            force_a = self._a_force()
            force_z = self._project_tangent(self.z, self._z_force(mod=mod))
            # flow a
            self.a += eps * force_a / self.mass_a
            self._sync_U_from_a()
            # flow z
            force_z_norm = np.linalg.norm(force_z, axis=-1)
            # check zero norm
            zero_mask = force_z_norm < 1e-10
            force_z_norm[zero_mask] = 1.0
            cos_force_z = np.cos(eps * force_z_norm / self.mass_z)
            sin_force_z = np.sin(eps * force_z_norm / self.mass_z)
            force_z_unit = force_z / force_z_norm[:, :, None]
            self.z = cos_force_z[:,:,None] * self.z + sin_force_z[:,:,None] * force_z_unit
            self._normalize_z_inplace()

        self._apply_periodicity_wrap()
    
    # ----------------------------------- Observables -----------------------------------

    def PP_corr_k(self):
        """
        Return fourier transformation of connected PP correlation.
        """
        N = self.N
        P = np.einsum("ijk,ijl->ijkl", self.z, np.conj(self.z))
        Px = P - 1 / N * np.eye(N)[None, None, :, :]
        Pk = fftn(Px, axes=(0, 1))
        S = np.sum(np.abs(Pk) ** 2, axis=(2, 3)) / self.V
        return S
    
    def topo_charge_z(self):
        """
        Skyrmion number of the system, using z field only.
        """
        z = self.z

        temp0 = (
            vdot_z(z, np.roll(z, -1, 0))
            * vdot_z(np.roll(z, -1, 0), np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), z)
        )
        temp1 = (
            vdot_z(z, np.roll(z, -1, (0, 1)))
            * vdot_z(np.roll(z, -1, (0, 1)), np.roll(z, -1, 1))
            * vdot_z(np.roll(z, -1, 1), z)
        )
        Q_z = np.angle(temp0) + np.angle(temp1)

        return np.sum(Q_z) / (2 * np.pi)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L = 5
    N = 2
    beta = 1.0
    beta1 = 0
    alpha = 0.5
    alpha1 = 0.5

    flow = CPN_RefVil_flow_fix_s(N, L, L, beta, beta1, alpha, alpha1, epsilon=0.05, n_step=1)

    def _random_unit_vectors(shape):
        real = np.random.randn(*shape)
        imag = np.random.randn(*shape)
        v = real + 1j * imag
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / norm
    
    flow.a = 2 * np.pi * np.random.rand(L, L, 2)
    flow.z = _random_unit_vectors((L, L, N))
    flow.s[0,0] = 1

    mod = 0
    action = []
    topo = []
    for _ in range(100):
        flow.flow_step(epsilon=0.02, n_step=1, mod=mod)
        action.append(flow.action(mod=mod))
        topo.append(flow.topo_charge())

    topo = np.array(topo)

    plt.plot(action)
    plt.show()

    plt.plot(topo[:, 0], label="Q_U")
    plt.plot(topo[:, 1], label="Q_z")
    plt.plot(topo[:, 2], label="sum(s)")
    plt.legend()
    plt.show()
