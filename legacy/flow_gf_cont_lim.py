"""Flow saved MC configurations with resumable per-chunk measurements."""

import glob
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import RLock, current_process

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func.func_CPN_RefVil_flow import CPN_RefVil_flow_fix_s, CPN_halfRefVil_flow
from func.obs import corr_len_stats, to_jsonable


# ============================== User parameters ==============================
N = 2
beta, beta1, alpha, alpha1 = 1.0, 0.0, 0.0, 0.0
# beta, beta1, alpha, alpha1 = 1.242, -0.974, 0.0, 0.147
norm = beta + beta1
beta /= norm
beta1 /= norm
alpha /= norm
alpha1 /= norm
mod = 1

mul_list = None
do_covariant = False

flow_t_ratio_list = [a for a in np.arange(0.01, 0.26, 0.04)]
flow_epsilon = 0.01
flow_mass_a = 1.0
flow_mass_z = 1.0
n_workers = 10
tau_window_c = 5.0

compute_backend = "torch_cuda"       # "numpy" or "torch_cuda"
torch_device = "cuda:0"
torch_dtype = "float64"         # CUDA v1 intentionally supports double precision only
gpu_batch_size = 100
gpu_allow_oom_backoff = True

SCHEMA_VERSION = 2


def mod_for(beta1_val):
    return 0 if abs(beta1_val) < 1e-8 else int(mod)


def parameter_folder(root):
    return os.path.join(root, f"N{N}_mod{mod_for(beta1)}",
                        f"beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}")


def flow_result_folder(root):
    return os.path.join(parameter_folder(root), "covariant" if do_covariant else "normal")


def _jsonable_canonical(obj):
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=2)
    os.replace(tmp, path)


def _diff(stored, current, prefix=""):
    out = []
    for key in sorted(set(stored) | set(current)):
        name = f"{prefix}.{key}" if prefix else key
        if key not in stored:
            out.append(f"{name}: stored=<missing>, current={current[key]!r}")
        elif key not in current:
            out.append(f"{name}: stored={stored[key]!r}, current=<missing>")
        elif isinstance(stored[key], dict) and isinstance(current[key], dict):
            out.extend(_diff(stored[key], current[key], name))
        elif stored[key] != current[key]:
            out.append(f"{name}: stored={stored[key]!r}, current={current[key]!r}")
    return out


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_manifests():
    paths = glob.glob(os.path.join(parameter_folder("gf_data"), "mul*", "manifest.json"))
    wanted = None if mul_list is None else [float(x) for x in mul_list]
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("status") != "ok":
            print(f"[skip] unsuccessful manifest: {path}")
            continue
        mul = float(data["parameters"]["mul"])
        if wanted is None or any(np.isclose(mul, x, atol=1e-9, rtol=0) for x in wanted):
            out.append((mul, path, data))
    return sorted(out, key=lambda x: x[0])


def _xi_from_modes(S00, S10, S01, L):
    m00, m10, m01 = np.mean(S00, axis=0), np.mean(S10, axis=0), np.mean(S01, axis=0)
    rx, ry = m00 / m10 - 1.0, m00 / m01 - 1.0
    denom = 2.0 * np.sin(np.pi / L)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (np.where(rx > 0, np.sqrt(rx) / denom, np.nan) +
                      np.where(ry > 0, np.sqrt(ry) / denom, np.nan))


def _xi_from_mode_chains(mode_chains, L, exclude=None):
    selected = [c for i, c in enumerate(mode_chains) if i != exclude]
    if not selected:
        return float("nan")
    return float(_xi_from_modes(np.concatenate([c[0] for c in selected]),
                                np.concatenate([c[1] for c in selected]),
                                np.concatenate([c[2] for c in selected]), L))


def _simulation_identity(manifest):
    """Return the new signature or a stable identity for a legacy inventory."""
    if manifest.get("config_signature"):
        return str(manifest["config_signature"])
    identity = {
        "parameters": manifest.get("parameters", {}),
        "run_config": manifest.get("run_config", {}),
        "L": manifest.get("L"),
        "measurements_path": manifest.get("measurements_path"),
        "config_chunks": manifest.get("config_chunks"),
        "chains": manifest.get("chains"),
    }
    digest = hashlib.sha256(_jsonable_canonical(identity).encode()).hexdigest()
    return f"legacy:{digest}"


def _load_inputs(manifest_path, manifest):
    schema = manifest.get("schema_version")
    if schema not in (None, SCHEMA_VERSION):
        raise RuntimeError(f"unsupported simulation manifest schema {schema!r}: {manifest_path}")
    base = os.path.dirname(manifest_path)
    if "measurements_path" in manifest:
        path = os.path.join(base, manifest["measurements_path"])
        with np.load(path, allow_pickle=False) as data:
            if schema == SCHEMA_VERSION and (
                    "config_signature" not in data.files or
                    str(data["config_signature"]) != manifest["config_signature"]):
                raise ValueError(f"simulation measurement parameters do not match manifest: {path}")
            required = ("chain_ids", "S00", "S10", "S01")
            missing = [name for name in required if name not in data.files]
            if missing:
                raise RuntimeError(f"simulation measurement archive {path} lacks {missing}")
            chain_ids = np.asarray(data["chain_ids"], dtype=int)
            mode_chains = [(data["S00"][i].copy(), data["S10"][i].copy(), data["S01"][i].copy())
                           for i in range(len(chain_ids))]
        chunks = []
        for index, chunk in enumerate(manifest.get("config_chunks", [])):
            item = dict(chunk)
            item["path"] = os.path.join(base, chunk["path"])
            item["index"] = index
            chunks.append(item)
        if not chunks:
            raise RuntimeError(f"simulation manifest contains no configuration chunks: {manifest_path}")
        return mode_chains, chain_ids, chunks

    # Compatibility with older one-file-per-chain/per-chain-chunk manifests.
    mode_chains, chain_ids, chunks = [], [], []
    for chain in sorted(manifest.get("chains", []), key=lambda item: item["rid"]):
        modes_name = chain.get("modes_path", chain.get("path"))
        if modes_name is None:
            raise KeyError(f"legacy chain {chain['rid']} has no modes_path or path")
        with np.load(os.path.join(base, modes_name), allow_pickle=False) as data:
            mode_chains.append((data["S00"].copy(), data["S10"].copy(), data["S01"].copy()))
        chain_ids.append(int(chain["rid"]))
        config_names = chain.get("config_paths")
        if config_names is None:
            config_names = [chain.get("path")]
        for name in config_names:
            if name is None:
                raise KeyError(f"legacy chain {chain['rid']} has no configuration path")
            chunks.append({"path": os.path.join(base, name), "chain_id": int(chain["rid"]),
                           "n_configurations": None, "index": len(chunks)})
    if not mode_chains or not chunks:
        raise RuntimeError(f"legacy simulation manifest has no usable chain data: {manifest_path}")
    return mode_chains, np.asarray(chain_ids, dtype=int), chunks


def _validate_simulation_parameters(manifest_path, manifest):
    expected = {"N": N, "mod": mod_for(beta1), "beta": beta, "beta1": beta1,
                "alpha": alpha, "alpha1": alpha1}
    stored = {key: manifest.get("parameters", {}).get(key) for key in expected}
    differences = _diff(stored, expected, "parameters")
    if differences:
        raise ValueError(f"simulation parameters do not match flow code for {manifest_path}:\n  "
                         + "\n  ".join(differences))


def _maybe_skip_legacy_result(out_json, manifest, nominal_rho):
    """Return True for a complete, validated pre-schema flow result."""
    if not os.path.exists(out_json):
        return False
    with open(out_json, encoding="utf-8") as fh:
        old = json.load(fh)
    if old.get("schema_version") is not None:
        return False
    if old.get("status") != "ok":
        raise RuntimeError(f"legacy partial/error flow result cannot be resumed: {out_json}")
    p = manifest["parameters"]
    expected_parameters = {key: p[key] for key in (
        "N", "mul", "mod", "beta", "beta1", "alpha", "alpha1",
        "beta_scaled", "beta1_scaled", "alpha_scaled", "alpha1_scaled")}
    stored_parameters = {key: old.get("parameters", {}).get(key) for key in expected_parameters}
    differences = _diff(stored_parameters, expected_parameters, "parameters")
    coupling_norm = float(p["beta_scaled"]) + float(p["beta1_scaled"])
    expected_flow = {
        "type": "covariant" if do_covariant else "normal",
        "flow_t_ratio_list": nominal_rho.tolist(),
        "flow_epsilon": float(flow_epsilon),
        "mass_a": float(flow_mass_a),
        "mass_z": float(flow_mass_z),
        "normalization": coupling_norm,
    }
    stored_flow = old.get("flow_config", {})
    differences += _diff(
        {key: stored_flow.get(key) for key in expected_flow}, expected_flow, "flow_config")
    expected_couplings = ({"beta": 1.0, "beta1": 0.0, "alpha1": 0.0}
                          if do_covariant else
                          {"beta": float(p["beta_scaled"]) / coupling_norm,
                           "beta1": float(p["beta1_scaled"]) / coupling_norm,
                           "alpha": float(p["alpha_scaled"]) / coupling_norm,
                           "alpha1": float(p["alpha1_scaled"]) / coupling_norm})
    stored_couplings = stored_flow.get("flow_couplings", {})
    differences += _diff(
        {key: stored_couplings.get(key) for key in expected_couplings},
        expected_couplings, "flow_config.flow_couplings")
    if old.get("L") != manifest.get("L"):
        differences.append(f"L: stored={old.get('L')!r}, current={manifest.get('L')!r}")
    if differences:
        raise ValueError(f"parameters do not match existing legacy flow result {out_json}:\n  "
                         + "\n  ".join(differences))
    data_name = old.get("flow_data_path")
    if not data_name:
        raise RuntimeError(f"legacy flow result lacks flow_data_path: {out_json}")
    data_path = os.path.join(os.path.dirname(out_json), data_name)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"legacy combined flow archive is missing: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        required = ("rho", "target_times", "E_action", "E_action_error",
                    "tE_action", "tE_action_error")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise RuntimeError(f"legacy combined flow archive {data_path} lacks {missing}")
        if not np.array_equal(np.asarray(data["rho"], dtype=float), nominal_rho):
            raise ValueError(f"flow_data.rho does not match existing legacy result: {data_path}")
    print(f"[skip] validated complete legacy result: {out_json}")
    return True


def _make_flow(params):
    if params["do_covariant"]:
        return CPN_halfRefVil_flow(params["L"], params["L"], params["N"], 1.0, 0.0, 0.0,
                                  epsilon=params["flow_epsilon"], n_step=1,
                                  mass_a=params["mass_a"], mass_z=params["mass_z"]), "covariant"
    if abs(params["alpha"]) < 1e-8:
        return CPN_halfRefVil_flow(params["L"], params["L"], params["N"], params["beta"],
                                  params["beta1"], params["alpha1"], epsilon=params["flow_epsilon"],
                                  n_step=1, mass_a=params["mass_a"], mass_z=params["mass_z"]), "halfRefVil"
    return CPN_RefVil_flow_fix_s(params["N"], params["L"], params["L"], params["beta"],
                                params["beta1"], params["alpha"], params["alpha1"],
                                epsilon=params["flow_epsilon"], n_step=1,
                                mass_a=params["mass_a"], mass_z=params["mass_z"]), "RefVil_fix_s"


def _cuda_execution_info():
    if compute_backend not in ("numpy", "torch_cuda"):
        raise ValueError("compute_backend must be 'numpy' or 'torch_cuda'")
    if compute_backend == "numpy":
        return {"backend": "numpy"}
    if torch_dtype != "float64":
        raise ValueError("the CUDA flow backend currently requires torch_dtype='float64'")
    if int(gpu_batch_size) <= 0:
        raise ValueError("gpu_batch_size must be positive")
    try:
        import torch
        from cpn_gf.flow_engine import ENGINE_VERSION
    except ImportError as exc:
        raise RuntimeError("compute_backend='torch_cuda' requires a CUDA-enabled PyTorch install") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "compute_backend='torch_cuda' was selected, but torch.cuda.is_available() is False")
    try:
        device = torch.device(torch_device)
        index = torch.cuda.current_device() if device.index is None else device.index
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"cannot initialize CUDA device {torch_device!r}") from exc
    return {"backend": "torch_cuda", "engine": ENGINE_VERSION, "dtype": torch_dtype,
            "device": str(device), "gpu_name": name, "compute_capability": capability,
            "torch_version": str(torch.__version__), "torch_cuda_version": str(torch.version.cuda),
            "requested_batch_size": int(gpu_batch_size),
            "oom_backoff": bool(gpu_allow_oom_backoff)}


def _init_tqdm_worker(lock):
    tqdm.set_lock(lock)


def _worker_progress_slot(worker_count):
    identity = current_process()._identity
    return 0 if not identity else (identity[-1] - 1) % worker_count


def _measure(flow, mod_value):
    modes = flow.PP_corr_k()
    return (float(flow.action(mod=mod_value)) / flow.V, float(modes[0, 0]),
            float(modes[1, 0]), float(modes[0, 1]), float(flow.topo_charge_z()))


def flow_chunk(params):
    output_steps = np.asarray(params["output_steps"], dtype=int)
    with np.load(params["config_path"], allow_pickle=False) as data:
        if params["require_config_signature"] and (
                "config_signature" not in data.files or
                str(data["config_signature"]) != params["simulation_signature"]):
            raise ValueError(f"configuration parameters do not match manifest: {params['config_path']}")
        z_configs, a_configs = data["z"], data["a"]
        s_configs = data["s"] if "s" in data.files else None
        if "chain_id" in data.files:
            chain_id = int(data["chain_id"])
        elif "rid" in data.files:
            chain_id = int(data["rid"])
        else:
            chain_id = int(params["chain_id"])
        n_cfg, n_out = len(z_configs), len(output_steps)
        values = np.empty((n_cfg, n_out, 5), dtype=float)
        violations, max_increase, flow_kind = 0, 0.0, None
        slot = _worker_progress_slot(params["worker_count"])
        with tqdm(total=n_cfg, desc=f"flow mul={params['mul']:.3f}, worker {slot + 1}",
                  position=slot + 1, leave=False, dynamic_ncols=True) as progress:
            for cfg_index in range(n_cfg):
                flow, flow_kind = _make_flow(params)
                flow.z = np.array(z_configs[cfg_index], copy=True)
                flow.a = np.array(a_configs[cfg_index], copy=True)
                if s_configs is not None and hasattr(flow, "s"):
                    flow.s = np.array(s_configs[cfg_index], copy=True)
                flow._sync_U_from_a()
                values[cfg_index, 0] = _measure(flow, params["mod"])
                old_action = values[cfg_index, 0, 0] * flow.V
                previous = int(output_steps[0])
                for out_index in range(1, n_out):
                    target = int(output_steps[out_index])
                    for _ in range(target - previous):
                        flow.flow_step(epsilon=params["flow_epsilon"], n_step=1, mod=params["mod"])
                    values[cfg_index, out_index] = _measure(flow, params["mod"])
                    new_action = values[cfg_index, out_index, 0] * flow.V
                    increase = new_action - old_action
                    if increase > 1e-10 * max(1.0, abs(old_action)):
                        violations += 1
                        max_increase = max(max_increase, increase)
                    old_action, previous = new_action, target
                progress.update()
    if n_cfg == 0:
        raise RuntimeError(f"configuration chunk is empty: {params['config_path']}")
    tmp = params["batch_path"] + ".tmp.npz"
    np.savez_compressed(tmp, metadata=np.asarray(params["batch_metadata"]),
                        E_action=values[:, :, 0], S00=values[:, :, 1], S10=values[:, :, 2],
                        S01=values[:, :, 3], Q_z=values[:, :, 4], chain_id=np.asarray(chain_id),
                        n_configurations=np.asarray(n_cfg), flow_kind=np.asarray(flow_kind),
                        action_violation_count=np.asarray(violations),
                        max_action_increase=np.asarray(max_increase))
    os.replace(tmp, params["batch_path"])
    return params["batch_path"]


def flow_chunk_torch(params):
    """Flow one input archive on a single GPU, in memory-bounded sub-batches."""
    import torch
    from cpn_gf.flow_engine import flow_batch

    output_steps = np.asarray(params["output_steps"], dtype=int)
    with np.load(params["config_path"], allow_pickle=False) as data:
        if params["require_config_signature"] and (
                "config_signature" not in data.files or
                str(data["config_signature"]) != params["simulation_signature"]):
            raise ValueError(f"configuration parameters do not match manifest: {params['config_path']}")
        z_configs = data["z"].copy()
        a_configs = data["a"].copy()
        s_configs = data["s"].copy() if "s" in data.files else None
        if "chain_id" in data.files:
            chain_id = int(data["chain_id"])
        elif "rid" in data.files:
            chain_id = int(data["rid"])
        else:
            chain_id = int(params["chain_id"])

    n_cfg, n_out = len(z_configs), len(output_steps)
    if n_cfg == 0:
        raise RuntimeError(f"configuration chunk is empty: {params['config_path']}")
    values = np.empty((n_cfg, n_out, 5), dtype=float)
    requested = min(int(params["gpu_batch_size"]), n_cfg)
    current_batch = requested
    start, violations, max_increase, used_sizes = 0, 0, 0.0, []
    flow_kind = None
    with tqdm(total=n_cfg, desc=f"CUDA flow mul={params['mul']:.3f}", leave=False,
              dynamic_ncols=True) as progress:
        while start < n_cfg:
            stop = min(n_cfg, start + current_batch)
            try:
                batch_s = None if s_configs is None else s_configs[start:stop]
                batch_values, batch_kind, batch_violations, batch_max = flow_batch(
                    z_configs[start:stop], a_configs[start:stop], batch_s, params,
                    params["torch_device"])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if not params["gpu_allow_oom_backoff"] or current_batch <= 1:
                    raise
                current_batch = max(1, current_batch // 2)
                print(f"[CUDA OOM] retrying with gpu_batch_size={current_batch}")
                continue
            # Active flow engine also returns Q_U as column 5; schema-2 legacy
            # output intentionally keeps its original five-column layout.
            values[start:stop] = batch_values[..., :5]
            flow_kind = batch_kind
            violations += batch_violations
            max_increase = max(max_increase, batch_max)
            used_sizes.append(stop - start)
            progress.update(stop - start)
            start = stop

    tmp = params["batch_path"] + ".tmp.npz"
    np.savez_compressed(
        tmp, metadata=np.asarray(params["batch_metadata"]),
        E_action=values[:, :, 0], S00=values[:, :, 1], S10=values[:, :, 2],
        S01=values[:, :, 3], Q_z=values[:, :, 4], chain_id=np.asarray(chain_id),
        n_configurations=np.asarray(n_cfg), flow_kind=np.asarray(flow_kind),
        action_violation_count=np.asarray(violations),
        max_action_increase=np.asarray(max_increase),
        gpu_batch_size_used=np.asarray(max(used_sizes)))
    os.replace(tmp, params["batch_path"])
    return params["batch_path"]


def _load_batch(path, expected_metadata):
    with np.load(path, allow_pickle=False) as data:
        stored = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
        differences = _diff(stored, expected_metadata, "batch")
        if differences:
            raise ValueError(f"parameters do not match existing flow batch {path}:\n  " + "\n  ".join(differences))
        required = ("E_action", "S00", "S10", "S01", "Q_z")
        if any(name not in data.files for name in required):
            raise RuntimeError(f"incomplete flow batch: {path}")
        arrays = {name: data[name].copy() for name in required}
        shape = arrays["E_action"].shape
        if any(x.shape != shape for x in arrays.values()) or shape != (expected_metadata["n_configurations"],
                                                                       len(expected_metadata["output_steps"])):
            raise RuntimeError(f"invalid observable shapes in flow batch: {path}")
        arrays.update(chain_id=int(data["chain_id"]), flow_kind=str(data["flow_kind"]),
                      action_violation_count=int(data["action_violation_count"]),
                      max_action_increase=float(data["max_action_increase"]),
                      gpu_batch_size_used=(int(data["gpu_batch_size_used"])
                                           if "gpu_batch_size_used" in data.files else None))
    return arrays


def _jackknife_error(replicas):
    replicas = np.asarray(replicas, dtype=float)
    n = replicas.shape[0]
    center = np.mean(replicas, axis=0)
    return np.sqrt((n - 1.0) / n * np.sum((replicas - center) ** 2, axis=0))


def _interp_rows(values, times, target):
    if target < times[0] or target > times[-1]:
        raise ValueError(f"target flow time {target} is outside [{times[0]}, {times[-1]}]")
    return np.asarray([np.interp(target, times, row) for row in values])


def _observables(raw, L):
    E = np.mean(raw["E_action"], axis=0)
    chi_m = np.mean(raw["S00"], axis=0)
    xi = _xi_from_modes(raw["S00"], raw["S10"], raw["S01"], L)
    q_mean = np.mean(raw["Q_z"], axis=0)
    chi_t = np.mean((raw["Q_z"] - q_mean) ** 2, axis=0) / float(L * L)
    return np.stack((E, chi_m, xi, chi_t), axis=-1)


def _select_chains(chains, omitted=None):
    keys = ("E_action", "S00", "S10", "S01", "Q_z")
    return {key: np.concatenate([chain[key] for i, chain in enumerate(chains) if i != omitted], axis=0)
            for key in keys}


def _fixed_observables(chains, times, target, L, omitted=None):
    keys = ("E_action", "S00", "S10", "S01", "Q_z")
    raw = {key: np.concatenate([_interp_rows(chain[key], times, target)
                                for i, chain in enumerate(chains) if i != omitted])[:, None]
           for key in keys}
    return _observables(raw, L)[0]


def run_one_mul(mul, manifest_path, manifest):
    out_root = flow_result_folder("gf_results")
    npz_dir = os.path.join(out_root, "npz")
    batch_dir = os.path.join(npz_dir, f"mul{mul:.3f}_batches")
    out_json = os.path.join(out_root, f"mul{mul:.3f}.json")
    _validate_simulation_parameters(manifest_path, manifest)
    nominal_rho = np.asarray(flow_t_ratio_list, dtype=float)
    if nominal_rho.ndim != 1 or nominal_rho.size == 0 or not np.all(np.isfinite(nominal_rho)) \
            or np.any(nominal_rho <= 0) or np.any(np.diff(nominal_rho) <= 0):
        raise ValueError("flow_t_ratio_list must contain finite, positive, strictly increasing values")
    if flow_epsilon <= 0:
        raise ValueError("flow_epsilon must be positive")
    execution_info = _cuda_execution_info()
    if _maybe_skip_legacy_result(out_json, manifest, nominal_rho):
        return
    os.makedirs(batch_dir, exist_ok=True)
    mode_chains, chain_ids, chunks = _load_inputs(manifest_path, manifest)
    if len(mode_chains) < 2:
        raise RuntimeError("at least two independent chains are required for jackknife")
    L = int(manifest["L"])
    simulation_identity = _simulation_identity(manifest)
    xi = _xi_from_mode_chains(mode_chains, L)
    xi_jack = np.asarray([_xi_from_mode_chains(mode_chains, L, exclude=i) for i in range(len(mode_chains))])
    if not np.isfinite(xi) or not np.all(np.isfinite(xi_jack)):
        raise RuntimeError("central or jackknife xi is non-finite")
    scaled = (nominal_rho[:, None] * np.concatenate(([xi], xi_jack))[None, :] ** 2).ravel() / flow_epsilon
    output_steps = np.unique(np.concatenate(([0], np.floor(scaled).astype(int), np.ceil(scaled).astype(int))))
    output_steps = output_steps[output_steps >= 0]
    times = output_steps * flow_epsilon
    p = manifest["parameters"]
    coupling_norm = float(p["beta_scaled"]) + float(p["beta1_scaled"])
    if abs(coupling_norm) < 1e-14:
        raise ValueError("beta_scaled + beta1_scaled is zero")
    flow_config = {"type": "covariant" if do_covariant else "normal",
                   "flow_t_ratio_list": nominal_rho.tolist(), "flow_epsilon": float(flow_epsilon),
                   "mass_a": float(flow_mass_a), "mass_z": float(flow_mass_z),
                   "integration_output_steps": output_steps.tolist(), "normalization": coupling_norm,
                   "simulation_signature": simulation_identity,
                   "tau_window_c": float(tau_window_c)}
    if compute_backend == "torch_cuda":
        flow_config["numerics"] = {key: execution_info[key]
                                   for key in ("backend", "engine", "dtype")}
    if os.path.exists(out_json):
        with open(out_json, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"legacy flow output cannot be validated/resumed: {out_json}")
        differences = _diff(old.get("flow_config", {}), flow_config, "flow_config")
        if differences:
            raise ValueError(f"parameters do not match existing flow result {out_json}:\n  " + "\n  ".join(differences))
        if old.get("status") == "ok":
            print(f"[skip] validated complete result: {out_json}")
            return
    _atomic_json(out_json, {"schema_version": SCHEMA_VERSION, "status": "running",
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "parameters": p, "L": L, "flow_config": flow_config,
                            "execution": execution_info})
    common = {"N": int(p["N"]), "L": L, "mod": int(p["mod"]),
              "beta": float(p["beta_scaled"]) / coupling_norm,
              "beta1": float(p["beta1_scaled"]) / coupling_norm,
              "alpha": float(p["alpha_scaled"]) / coupling_norm,
              "alpha1": float(p["alpha1_scaled"]) / coupling_norm,
              "flow_epsilon": float(flow_epsilon), "mass_a": float(flow_mass_a),
              "mass_z": float(flow_mass_z), "output_steps": output_steps,
              "do_covariant": bool(do_covariant), "mul": float(mul),
              "simulation_signature": simulation_identity,
              "require_config_signature": manifest.get("schema_version") == SCHEMA_VERSION,
              "compute_backend": compute_backend, "torch_device": torch_device,
              "gpu_batch_size": int(gpu_batch_size),
              "gpu_allow_oom_backoff": bool(gpu_allow_oom_backoff)}
    jobs, specs = [], []
    for chunk in chunks:
        if not os.path.exists(chunk["path"]):
            raise FileNotFoundError(f"missing configuration chunk: {chunk['path']}")
        source_hash = _sha256(chunk["path"])
        with np.load(chunk["path"], allow_pickle=False) as data:
            actual_count = len(data["z"])
            if "chain_id" in data.files:
                actual_chain = int(data["chain_id"])
            elif "rid" in data.files:
                actual_chain = int(data["rid"])
            else:
                actual_chain = int(chunk["chain_id"])
        expected_count = chunk.get("n_configurations")
        if ((expected_count is not None and actual_count != int(expected_count)) or
                actual_chain != int(chunk["chain_id"])):
            raise ValueError(f"configuration chunk identity/count mismatch: {chunk['path']}")
        batch_path = os.path.join(batch_dir, f"batch_{chunk['index']:08d}.npz")
        metadata = {"schema_version": SCHEMA_VERSION, "flow_config": flow_config,
                    "source_path": os.path.relpath(chunk["path"], os.path.dirname(manifest_path)),
                    "source_sha256": source_hash, "chain_id": actual_chain,
                    "n_configurations": actual_count, "output_steps": output_steps.tolist()}
        specs.append((batch_path, metadata))
        if not os.path.exists(batch_path):
            jobs.append(dict(common, config_path=chunk["path"], chain_id=actual_chain,
                             batch_path=batch_path, batch_metadata=_jsonable_canonical(metadata)))
        else:
            _load_batch(batch_path, metadata)
    if jobs and compute_backend == "numpy":
        worker_count = min(n_workers, len(jobs))
        for job in jobs:
            job["worker_count"] = worker_count
        lock = RLock()
        tqdm.set_lock(lock)
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_tqdm_worker, initargs=(lock,)) as pool:
            futures = [pool.submit(flow_chunk, job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"flow mul={mul:.3f}"):
                fut.result()
    elif jobs:
        print(f"[CUDA] using one GPU process on {execution_info['gpu_name']}; "
              f"requested batch size {gpu_batch_size}")
        for job in tqdm(jobs, total=len(jobs), desc=f"CUDA chunks mul={mul:.3f}"):
            flow_chunk_torch(job)
    loaded = [_load_batch(path, metadata) for path, metadata in specs]
    if compute_backend == "torch_cuda":
        execution_info["effective_batch_sizes"] = sorted({
            item["gpu_batch_size_used"] for item in loaded
            if item["gpu_batch_size_used"] is not None})
    chain_map = {int(cid): [] for cid in chain_ids}
    for result in loaded:
        if result["chain_id"] not in chain_map:
            raise ValueError(f"unknown chain id in flow batch: {result['chain_id']}")
        chain_map[result["chain_id"]].append(result)
    keys = ("E_action", "S00", "S10", "S01", "Q_z")
    chains = []
    for cid in chain_ids:
        parts = chain_map[int(cid)]
        if not parts:
            raise RuntimeError(f"chain {cid} has no flowed configurations")
        chains.append({key: np.concatenate([part[key] for part in parts], axis=0) for key in keys})
    trajectory_values = _observables(_select_chains(chains), L)
    trajectory_replicas = np.asarray([_observables(_select_chains(chains, i), L)
                                      for i in range(len(chains))])
    trajectory_errors = _jackknife_error(trajectory_replicas)
    target_times = nominal_rho * xi ** 2
    fixed_values = np.asarray([_fixed_observables(chains, times, target, L) for target in target_times])
    fixed_replicas = np.empty((len(chains), len(nominal_rho), 4))
    for omitted in range(len(chains)):
        for k, rho in enumerate(nominal_rho):
            fixed_replicas[omitted, k] = _fixed_observables(
                chains, times, rho * xi_jack[omitted] ** 2, L, omitted=omitted)
    fixed_errors = _jackknife_error(fixed_replicas)
    names = ("E_action", "chi_m", "xi", "chi_t")
    flow_data_path = os.path.join(npz_dir, f"mul{mul:.3f}_flow.npz")
    arrays = {"times": times, "output_steps": output_steps, "chain_ids": chain_ids,
              "n_configurations": np.asarray([len(c["E_action"]) for c in chains]),
              "xi_jackknife": xi_jack, "rho": nominal_rho, "target_times": target_times,
              "sqrt_rho": np.sqrt(nominal_rho), "sqrt_8t_over_L": np.sqrt(8 * target_times) / L,
              "trajectory_t": times, "trajectory_t_over_xi2": times / xi ** 2,
              "trajectory_sqrt_8t_over_L": np.sqrt(8 * times) / L,
              "flow_kind": np.asarray(sorted({x["flow_kind"] for x in loaded})),
              "batch_paths": np.asarray([os.path.relpath(path, out_root) for path, _ in specs])}
    for index, name in enumerate(names):
        arrays[name] = fixed_values[:, index]
        arrays[f"{name}_error"] = fixed_errors[:, index]
        arrays[f"trajectory_{name}"] = trajectory_values[:, index]
        arrays[f"trajectory_{name}_error"] = trajectory_errors[:, index]
    arrays["tE_action"] = target_times * arrays["E_action"]
    arrays["tE_action_error"] = target_times * arrays["E_action_error"]
    arrays["trajectory_tE_action"] = times * arrays["trajectory_E_action"]
    arrays["trajectory_tE_action_error"] = times * arrays["trajectory_E_action_error"]
    tmp = flow_data_path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, flow_data_path)
    xi_stats = corr_len_stats([c[0] for c in mode_chains], [c[1] for c in mode_chains],
                              [c[2] for c in mode_chains], L=L, c=tau_window_c)
    payload = {"schema_version": SCHEMA_VERSION, "status": "ok",
               "created_at": datetime.now().isoformat(timespec="seconds"), "parameters": p,
               "L": L, "xi": xi_stats["xi"], "flow_config": flow_config,
               "execution": execution_info,
               "flow_data_path": os.path.relpath(flow_data_path, out_root),
               "flow_batch_paths": arrays["batch_paths"],
               "observables": {name: {"mean": arrays[name], "error": arrays[f"{name}_error"]}
                               for name in names},
               "independent_runs": [{"chain_id": int(cid), "n_configurations": len(chains[i]["E_action"]),
                                     "action_violation_count": sum(x["action_violation_count"] for x in chain_map[int(cid)]),
                                     "max_action_increase": max(x["max_action_increase"] for x in chain_map[int(cid)])}
                                    for i, cid in enumerate(chain_ids)]}
    _atomic_json(out_json, payload)
    print(f"[ok] {out_json}")


def main():
    manifests = discover_manifests()
    if not manifests:
        print(f"[no data] no successful manifests under {parameter_folder('gf_data')}")
        return
    for mul, path, manifest in manifests:
        run_one_mul(mul, path, manifest)


if __name__ == "__main__":
    main()
