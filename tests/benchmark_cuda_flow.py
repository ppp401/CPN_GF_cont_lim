"""Synthetic CPU/CUDA flow benchmark. Edit parameters below and run from repo root."""

import time

import numpy as np
import torch

from legacy.func.func_CPN_RefVil_flow import CPN_RefVil_flow_fix_s
from cpn_gf.flow_engine import TorchCPNFlowBatch


L = 100
N = 2
BATCH_SIZE = 16
FLOW_STEPS = 100
MOD = 1
EPSILON = 0.01
BETA, BETA1, ALPHA, ALPHA1 = 1.15, -0.15, 0.35, 0.22


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required")
    rng = np.random.default_rng(12345)
    z = rng.normal(size=(BATCH_SIZE, L, L, N)) + 1j * rng.normal(
        size=(BATCH_SIZE, L, L, N))
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    a = rng.uniform(-np.pi, np.pi, size=(BATCH_SIZE, L, L, 2))
    s = rng.integers(-2, 3, size=(BATCH_SIZE, L, L), dtype=np.int64)
    params = {"beta": BETA, "beta1": BETA1, "alpha": ALPHA, "alpha1": ALPHA1,
              "flow_epsilon": EPSILON, "mass_a": 1.0, "mass_z": 1.0}

    cpu_start = time.perf_counter()
    cpu_measurements = []
    for index in range(BATCH_SIZE):
        flow = CPN_RefVil_flow_fix_s(
            N, L, L, BETA, BETA1, ALPHA, ALPHA1, epsilon=EPSILON, n_step=1)
        flow.z, flow.a, flow.s = z[index].copy(), a[index].copy(), s[index].copy()
        flow._sync_U_from_a()
        for _ in range(FLOW_STEPS):
            flow.flow_step(mod=MOD)
        modes = flow.PP_corr_k()
        cpu_measurements.append((flow.action(mod=MOD) / flow.V, modes[0, 0],
                                 modes[1, 0], modes[0, 1], flow.topo_charge_z(),
                                 np.sum(np.angle(np.exp(1j * flow._plaquette_da())))
                                 / (2.0 * np.pi)))
    cpu_seconds = time.perf_counter() - cpu_start

    warmup = TorchCPNFlowBatch(z[:1], a[:1], s[:1], params, "RefVil_fix_s", "cuda:0")
    warmup.flow_step(mod=MOD)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_start = time.perf_counter()
    gpu_flow = TorchCPNFlowBatch(z, a, s, params, "RefVil_fix_s", "cuda:0")
    for _ in range(FLOW_STEPS):
        gpu_flow.flow_step(mod=MOD)
    gpu_values = gpu_flow.measure(mod=MOD).cpu().numpy()
    torch.cuda.synchronize()
    gpu_seconds = time.perf_counter() - gpu_start
    difference = np.max(np.abs(gpu_values - np.asarray(cpu_measurements)))
    peak_mib = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"L={L}, N={N}, batch={BATCH_SIZE}, steps={FLOW_STEPS}, dtype=float64/complex128")
    print(f"CPU: {cpu_seconds:.3f} s")
    print(f"CUDA end-to-end: {gpu_seconds:.3f} s")
    print(f"end-to-end speedup: {cpu_seconds / gpu_seconds:.2f}x")
    print(f"peak allocated: {peak_mib:.1f} MiB")
    print(f"max observable difference: {difference:.3e}")


if __name__ == "__main__":
    main()
