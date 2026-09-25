"""Generate resumable MC configurations for the gradient-flow study.

Edit the module-level parameters and run this file from the repository root.
Every logical chain owns an atomic checkpoint, so a restarted process pool may
assign that chain to any worker without repeating completed work.
"""

import hashlib
import json
import math
import os
import secrets
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func.func_CPN_HB import CPNSampler
from func.func_CPN_RefVil_HMC import CPN_RefVil_HMCSampler
from func.obs import compute_stats, corr_len_from_PP_corr_k, corr_len_stats, to_jsonable


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
mul_list = [0.8, 0.9]
L0 = 50
pilot_therm = 400
pilot_meas = 1000
pilot_run_times = 10
L_multiple = 6
therm = 2000
meas = 20000
meas_int = 4
flow_save_int = 20
configs_per_file = 100
run_times = 10
n_workers = 10
seed_base = None
heatbath_fraction = 0.4
mass_a = 1.0
mass_z = 1.0
s_step = 1.0
s_max = 100
pilot_epsilon = 0.02
pilot_n_leapfrog = 50
pilot_s_update_num = 3
epsilon = 0.01
n_leapfrog = 40
s_update_num = 3
tau_window_c = 5.0

SCHEMA_VERSION = 2
TOPOLOGY_LABELS = ("Q_U", "Q_z", "Q_s")


def mod_for(beta1_val):
    return 0 if abs(beta1_val) < 1e-8 else int(mod)


def sampler_kind_for(beta1_val, alpha_val):
    return "HB" if abs(beta1_val) < 1e-8 and abs(alpha_val) < 1e-8 else "RefVil_HMC"


def folder_path(mod_run):
    return os.path.join("gf_data", f"N{N}_mod{mod_run}",
                        f"beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}")


def mul_dir_path(mod_run, mul):
    return os.path.join(folder_path(mod_run), f"mul{mul:.3f}")


def manifest_path(mod_run, mul):
    return os.path.join(mul_dir_path(mod_run, mul), "manifest.json")


def _atomic_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=2)
    os.replace(tmp, path)


def _canonical(obj):
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _signature(parameters, run_config):
    comparable = {k: v for k, v in run_config.items() if k != "n_workers"}
    raw = _canonical({"parameters": parameters, "run_config": comparable})
    return hashlib.sha256(raw.encode()).hexdigest()


def _diff(stored, current, prefix=""):
    differences = []
    for key in sorted(set(stored) | set(current)):
        name = f"{prefix}.{key}" if prefix else key
        if key not in stored:
            differences.append(f"{name}: stored=<missing>, current={current[key]!r}")
        elif key not in current:
            differences.append(f"{name}: stored={stored[key]!r}, current=<missing>")
        elif isinstance(stored[key], dict) and isinstance(current[key], dict):
            differences.extend(_diff(stored[key], current[key], name))
        elif stored[key] != current[key]:
            differences.append(f"{name}: stored={stored[key]!r}, current={current[key]!r}")
    return differences


def _validate_existing(path, stored, parameters, run_config):
    schema = stored.get("schema_version")
    if schema not in (None, SCHEMA_VERSION):
        raise RuntimeError(f"{path} uses unsupported schema {schema!r}")
    if schema is None and stored.get("status") != "ok":
        raise RuntimeError(f"legacy partial/error run cannot be resumed: {path}")
    old_run = {k: v for k, v in stored.get("run_config", {}).items() if k != "n_workers"}
    new_run = {k: v for k, v in run_config.items() if k != "n_workers"}
    differences = _diff(stored.get("parameters", {}), parameters, "parameters")
    differences += _diff(old_run, new_run, "run_config")
    if differences:
        raise ValueError(f"parameters do not match existing run {path}:\n  " + "\n  ".join(differences))
    return schema is None


def _validate_legacy_complete_run(manifest_path, manifest, run_config):
    """Validate old completed files without treating them as resumable state."""
    base = os.path.dirname(manifest_path)
    measurement_name = manifest.get("measurements_path")
    chunks = manifest.get("config_chunks")
    if not measurement_name or not isinstance(chunks, list):
        raise RuntimeError(f"legacy manifest lacks completed-output inventory: {manifest_path}")
    measurement_path = os.path.join(base, measurement_name)
    if not os.path.isfile(measurement_path):
        raise FileNotFoundError(f"legacy measurement archive is missing: {measurement_path}")
    expected_measurements = (0 if run_config["meas"] <= 0 else
                             (run_config["meas"] - 1) // max(1, run_config["meas_int"]) + 1)
    expected_configs = (0 if run_config["meas"] <= 0 else
                        (run_config["meas"] - 1) // max(1, run_config["flow_save_int"]) + 1)
    with np.load(measurement_path, allow_pickle=False) as data:
        required = ("S00", "S10", "S01", "topological_charge", "chain_ids")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise RuntimeError(f"legacy measurement archive {measurement_path} lacks {missing}")
        if len(data["chain_ids"]) != run_config["run_times"]:
            raise RuntimeError(f"legacy measurement chain count is inconsistent: {measurement_path}")
        if set(map(int, data["chain_ids"])) != set(range(run_config["run_times"])):
            raise RuntimeError(f"legacy measurement chain IDs are inconsistent: {measurement_path}")
        for name in ("S00", "S10", "S01", "topological_charge"):
            if data[name].shape[:2] != (run_config["run_times"], expected_measurements):
                raise RuntimeError(f"legacy {name} shape is inconsistent in {measurement_path}")
    per_chain = {rid: 0 for rid in range(run_config["run_times"])}
    seen_paths = set()
    for chunk in chunks:
        path = os.path.join(base, chunk["path"])
        normalized_path = os.path.abspath(path)
        if normalized_path in seen_paths:
            raise RuntimeError(f"legacy manifest contains a duplicate configuration chunk: {path}")
        seen_paths.add(normalized_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"legacy configuration chunk is missing: {path}")
        with np.load(path, allow_pickle=False) as data:
            actual_count = len(data["z"])
            actual_chain = int(data["chain_id"] if "chain_id" in data.files else chunk["chain_id"])
        if actual_count != int(chunk["n_configurations"]) or actual_chain != int(chunk["chain_id"]):
            raise RuntimeError(f"legacy configuration chunk identity/count is inconsistent: {path}")
        if actual_chain not in per_chain:
            raise RuntimeError(f"legacy configuration chunk has unknown chain {actual_chain}: {path}")
        per_chain[actual_chain] += actual_count
    incorrect = {rid: count for rid, count in per_chain.items() if count != expected_configs}
    if incorrect:
        raise RuntimeError(f"legacy configuration counts are inconsistent: {incorrect}; expected {expected_configs}")


def _seed(run_entropy, mul, rid, generation, phase):
    token = f"{run_entropy}|{mul:.17g}|{rid}|{generation}|{phase}"
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")


def _make_sampler(params):
    kind = sampler_kind_for(params["beta1"], params["alpha"])
    if kind == "HB":
        sampler = CPNSampler(params["N"], params["L"], params["L"], params["beta"],
                             params["alpha1"], seed=params.get("seed"))
    else:
        sampler = CPN_RefVil_HMCSampler(
            params["N"], params["L"], params["L"], params["beta"], params["beta1"],
            params["alpha"], params["alpha1"], seed=params.get("seed"),
            epsilon=params["epsilon"], n_leapfrog=params["n_leapfrog"],
            mass_a=params["mass_a"], mass_z=params["mass_z"], s_step=params["s_step"],
            s_update_num=params["s_update_num"], s_max=params["s_max"])
    return sampler, kind


def _sweep(sampler, kind, params):
    if kind == "HB":
        sampler.sweep(heatbath_fraction=params["heatbath_fraction"])
    else:
        sampler.sweep(mod=params["mod"])


def topological_stats(topo_chains, L, c):
    volume = float(L * L)
    results = {}
    for index, label in enumerate(TOPOLOGY_LABELS):
        q_chains = [np.asarray(values, dtype=float)[:, index] for values in topo_chains]
        q_mean, q_tau, q_error, n_total = compute_stats(q_chains, c=c)
        chi_chains = [(q - q_mean) ** 2 / volume for q in q_chains]
        chi_mean, chi_tau, chi_error, chi_n_total = compute_stats(chi_chains, c=c)
        results[label] = {
            "Q": {"mean": q_mean, "tau": q_tau, "error": q_error, "n_total": n_total},
            "chi_t": {"mean": chi_mean, "tau": chi_tau, "error": chi_error,
                      "n_total": chi_n_total}}
    return results


def pilot_chain(params):
    sampler, kind = _make_sampler(params)
    rid = params["rid"]
    for _ in tqdm(range(params["pilot_therm"]), desc=f"pilot {rid} therm", leave=False):
        _sweep(sampler, kind, params)
    total = np.zeros((params["L"], params["L"]), dtype=float)
    for _ in tqdm(range(params["pilot_meas"]), desc=f"pilot {rid} meas", leave=False):
        _sweep(sampler, kind, params)
        total += sampler.PP_corr_k()
    rates = (float("nan"), float("nan")) if kind == "HB" else tuple(map(float, sampler.accept_rate))
    return rid, total, int(params["pilot_meas"]), rates[0], rates[1], kind


def _pilot_file(checkpoint_dir, rid):
    return os.path.join(checkpoint_dir, f"pilot_chain_{rid:04d}.npz")


def run_pilot(params):
    results, missing = [], []
    for rid in range(params["pilot_run_times"]):
        path = _pilot_file(params["checkpoint_dir"], rid)
        if not os.path.exists(path):
            missing.append(rid)
            continue
        with np.load(path, allow_pickle=False) as data:
            if str(data["config_signature"]) != params["config_signature"]:
                raise ValueError(f"pilot checkpoint parameters do not match: {path}")
            results.append((rid, data["S_sum"].copy(), int(data["count"]),
                            float(data["hmc_rate"]), float(data["metro_rate"]),
                            str(data["sampler_kind"])))
    if missing:
        with ProcessPoolExecutor(max_workers=min(params["n_workers"], len(missing))) as pool:
            futures = []
            for rid in missing:
                p = dict(params, rid=rid, seed=_seed(params["run_entropy"], params["mul"], rid, 0, "pilot"))
                futures.append(pool.submit(pilot_chain, p))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="pilot collecting"):
                result = fut.result()
                rid, S_sum, count, hmc_rate, metro_rate, kind = result
                path = _pilot_file(params["checkpoint_dir"], rid)
                tmp = path + ".tmp.npz"
                np.savez_compressed(tmp, config_signature=np.asarray(params["config_signature"]),
                                    S_sum=S_sum, count=np.asarray(count), hmc_rate=np.asarray(hmc_rate),
                                    metro_rate=np.asarray(metro_rate), sampler_kind=np.asarray(kind))
                os.replace(tmp, path)
                results.append(result)
    results.sort(key=lambda x: x[0])
    S_total = sum((x[1] for x in results), np.zeros((params["L"], params["L"])))
    n_total = sum(x[2] for x in results)
    xi_x, xi_y, xi = corr_len_from_PP_corr_k(S_total / n_total, params["L"])
    hmc = [x[3] for x in results if np.isfinite(x[3])]
    metro = [x[4] for x in results if np.isfinite(x[4])]
    return {"xi": xi, "xi_x": xi_x, "xi_y": xi_y,
            "accept_rate_hmc": float(np.mean(hmc)) if hmc else None,
            "accept_rate_metro": float(np.mean(metro)) if metro else None,
            "sampler_kind": results[0][5]}


def _checkpoint_path(params):
    return os.path.join(params["checkpoint_dir"], f"chain_{params['rid']:04d}.npz")


def _save_chain_checkpoint(params, sampler, kind, state):
    keys = ("phase", "thermalization_sweeps_completed", "production_sweeps_completed",
            "n_configurations", "restart_generation", "config_chunks")
    metadata = {key: state[key] for key in keys}
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION), "config_signature": np.asarray(params["config_signature"]),
        "metadata": np.asarray(_canonical(metadata)), "z": np.asarray(sampler.z),
        "a": np.angle(sampler.U) if kind == "HB" else np.asarray(sampler.a),
        "S00": np.asarray(state["S00"]), "S10": np.asarray(state["S10"]),
        "S01": np.asarray(state["S01"]),
        "topological_charge": np.asarray(state["topological_charge"], dtype=int).reshape((-1, 3)),
        "accepted_hmc": np.asarray(getattr(sampler, "accepted_hmc", 0)),
        "attempted_hmc": np.asarray(getattr(sampler, "attempted_hmc", 0)),
        "accepted_metro": np.asarray(getattr(sampler, "accepted_metro", 0)),
        "attempted_metro": np.asarray(getattr(sampler, "attempted_metro", 0)),
        "sampler_kind": np.asarray(kind), "rid": np.asarray(params["rid"]),
        "L": np.asarray(params["L"]), "N": np.asarray(params["N"])}
    if kind != "HB":
        arrays["s"] = np.asarray(sampler.s, dtype=int)
    path = _checkpoint_path(params)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _load_chain_checkpoint(params, sampler, kind):
    path = _checkpoint_path(params)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != SCHEMA_VERSION or str(data["config_signature"]) != params["config_signature"]:
            raise ValueError(f"chain checkpoint parameters do not match: {path}")
        if int(data["rid"]) != params["rid"] or int(data["L"]) != params["L"] or str(data["sampler_kind"]) != kind:
            raise ValueError(f"chain checkpoint identity does not match: {path}")
        state = json.loads(str(data["metadata"]))
        sampler.z = data["z"].copy()
        if kind == "HB":
            sampler.U = np.exp(1j * data["a"])
        else:
            sampler.a = data["a"].copy()
            sampler.s = data["s"].copy()
            sampler._sync_U_from_a()
            for name in ("accepted_hmc", "attempted_hmc", "accepted_metro", "attempted_metro"):
                setattr(sampler, name, int(data[name]))
        state.update(S00=data["S00"].tolist(), S10=data["S10"].tolist(), S01=data["S01"].tolist(),
                     topological_charge=data["topological_charge"].tolist())
    return state


def measurement_chain(params):
    rid = params["rid"]
    sampler, kind = _make_sampler(params)
    state = _load_chain_checkpoint(params, sampler, kind)
    if state is None:
        state = {"phase": "thermalization", "thermalization_sweeps_completed": 0,
                 "production_sweeps_completed": 0, "n_configurations": 0,
                 "restart_generation": 0, "config_chunks": [], "S00": [], "S10": [],
                 "S01": [], "topological_charge": []}
    else:
        state["restart_generation"] += 1
    np.random.seed(_seed(params["run_entropy"], params["mul"], rid,
                         state["restart_generation"], state["phase"]))
    checkpoint_interval = max(1, int(params["flow_save_int"]))
    start = int(state["thermalization_sweeps_completed"])
    for completed in tqdm(range(start, params["therm"]), desc=f"chain {rid} therm", leave=False):
        _sweep(sampler, kind, params)
        state["thermalization_sweeps_completed"] = completed + 1
        if (completed + 1) % checkpoint_interval == 0 or completed + 1 == params["therm"]:
            _save_chain_checkpoint(params, sampler, kind, state)
    state["phase"] = "production"
    _save_chain_checkpoint(params, sampler, kind, state)

    z_chunk, a_chunk, s_chunk = [], [], []
    meas_int_local = max(1, int(params["meas_int"]))
    save_int_local = max(1, int(params["flow_save_int"]))
    chunk_size = int(params["configs_per_file"])
    if chunk_size <= 0:
        raise ValueError("configs_per_file must be positive")

    def flush_chunk():
        if not z_chunk:
            return
        first_local = state["n_configurations"] - len(z_chunk)
        global_index = params["global_config_offset"] + first_local
        path = os.path.join(params["npz_dir"], f"config_{global_index:08d}.npz")
        archive = {"z": np.asarray(z_chunk), "a": np.asarray(a_chunk),
                   "sampler_kind": np.asarray(kind), "chain_id": np.asarray(rid),
                   "first_config_index": np.asarray(first_local), "L": np.asarray(params["L"]),
                   "N": np.asarray(params["N"]), "config_signature": np.asarray(params["config_signature"])}
        if kind != "HB":
            archive["s"] = np.asarray(s_chunk, dtype=int)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **archive)
        os.replace(tmp, path)
        state["config_chunks"].append({"path": os.path.relpath(path, params["mul_dir"]),
                                       "chain_id": rid, "n_configurations": len(z_chunk)})
        z_chunk.clear()
        a_chunk.clear()
        s_chunk.clear()
        _save_chain_checkpoint(params, sampler, kind, state)

    start = int(state["production_sweeps_completed"])
    for i in tqdm(range(start, params["meas"]), desc=f"chain {rid} meas", leave=False):
        _sweep(sampler, kind, params)
        if i % meas_int_local == 0:
            S = sampler.PP_corr_k()
            state["S00"].append(float(S[0, 0]))
            state["S10"].append(float(S[1, 0]))
            state["S01"].append(float(S[0, 1]))
            state["topological_charge"].append(np.asarray(sampler.topo_charge(), dtype=int).tolist())
        if i % save_int_local == 0:
            z_chunk.append(np.array(sampler.z, copy=True))
            if kind == "HB":
                a_chunk.append(np.angle(sampler.U))
            else:
                a_chunk.append(np.array(sampler.a, copy=True))
                s_chunk.append(np.array(sampler.s, copy=True))
            state["n_configurations"] += 1
        state["production_sweeps_completed"] = i + 1
        if len(z_chunk) >= chunk_size:
            flush_chunk()
    flush_chunk()
    expected_measurements = 0 if params["meas"] <= 0 else (params["meas"] - 1) // meas_int_local + 1
    expected_configs = 0 if params["meas"] <= 0 else (params["meas"] - 1) // save_int_local + 1
    if (state["production_sweeps_completed"] != params["meas"] or len(state["S00"]) != expected_measurements
            or state["n_configurations"] != expected_configs):
        raise RuntimeError(f"chain {rid} completion counts are inconsistent")
    state["phase"] = "complete"
    _save_chain_checkpoint(params, sampler, kind, state)
    rates = (None, None) if kind == "HB" else tuple(map(float, sampler.accept_rate))
    return _chain_result(params, state, kind, rates)


def _chain_result(params, state, kind, rates):
    return {"rid": params["rid"], "config_chunks": state["config_chunks"], "sampler_kind": kind,
            "n_measurements": len(state["S00"]), "n_configurations": state["n_configurations"],
            "S00": np.asarray(state["S00"]), "S10": np.asarray(state["S10"]),
            "S01": np.asarray(state["S01"]),
            "topological_charge": np.asarray(state["topological_charge"], dtype=int).reshape((-1, 3)),
            "accept_rate_hmc": rates[0], "accept_rate_metro": rates[1]}


def _load_complete_chain(params):
    sampler, kind = _make_sampler(dict(params, seed=0))
    state = _load_chain_checkpoint(params, sampler, kind)
    if state is None or state.get("phase") != "complete":
        return None
    expected_m = 0 if params["meas"] <= 0 else (params["meas"] - 1) // max(1, params["meas_int"]) + 1
    expected_c = 0 if params["meas"] <= 0 else (params["meas"] - 1) // max(1, params["flow_save_int"]) + 1
    if state["production_sweeps_completed"] != params["meas"] or len(state["S00"]) != expected_m or state["n_configurations"] != expected_c:
        raise RuntimeError(f"complete checkpoint has inconsistent counts: {_checkpoint_path(params)}")
    rates = (None, None) if kind == "HB" else tuple(map(float, sampler.accept_rate))
    return _chain_result(params, state, kind, rates)


def _settings(mul):
    mod_run = mod_for(beta1)
    parameters = {"N": N, "mul": mul, "mod": mod_run, "beta": beta, "beta1": beta1,
                  "alpha": alpha, "alpha1": alpha1, "beta_scaled": beta * mul,
                  "beta1_scaled": beta1 * mul, "alpha_scaled": alpha * mul, "alpha1_scaled": alpha1 * mul}
    run_config = {"L0": L0, "L_multiple": L_multiple, "pilot_therm": pilot_therm,
                  "pilot_meas": pilot_meas, "pilot_run_times": pilot_run_times, "therm": therm,
                  "meas": meas, "meas_int": meas_int, "flow_save_int": flow_save_int,
                  "configs_per_file": configs_per_file, "run_times": run_times, "n_workers": n_workers,
                  "seed_base": seed_base, "heatbath_fraction": heatbath_fraction,
                  "pilot_epsilon": pilot_epsilon, "pilot_n_leapfrog": pilot_n_leapfrog,
                  "pilot_s_update_num": pilot_s_update_num, "epsilon": epsilon,
                  "n_leapfrog": n_leapfrog, "mass_a": mass_a, "mass_z": mass_z, "s_step": s_step,
                  "s_update_num": s_update_num, "s_max": s_max, "tau_window_c": tau_window_c}
    return mod_run, parameters, run_config


def run_one_mul(mul):
    mod_run, parameters, run_config = _settings(mul)
    out_dir = mul_dir_path(mod_run, mul)
    out_manifest = manifest_path(mod_run, mul)
    npz_dir = os.path.join(out_dir, "npz")
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    signature = _signature(parameters, run_config)
    existing = None
    if os.path.exists(out_manifest):
        with open(out_manifest, encoding="utf-8") as fh:
            existing = json.load(fh)
        is_legacy = _validate_existing(out_manifest, existing, parameters, run_config)
        if existing.get("status") == "ok":
            if is_legacy:
                _validate_legacy_complete_run(out_manifest, existing, run_config)
                print(f"[skip] validated complete legacy run: {out_manifest}")
            else:
                print(f"[skip] validated complete run: {out_manifest}")
            return out_manifest, "skip"
    elif ((os.path.isdir(npz_dir) and any(os.scandir(npz_dir))) or
          (os.path.isdir(checkpoint_dir) and any(os.scandir(checkpoint_dir)))):
        raise RuntimeError(
            f"unverifiable partial output exists below {out_dir} without a schema-{SCHEMA_VERSION} "
            "manifest; relocate or remove it before running"
        )
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_entropy = int(existing["run_entropy"]) if existing else int(seed_base if seed_base is not None else secrets.randbits(63))
    created_at = existing.get("created_at", datetime.now().isoformat(timespec="seconds")) if existing else datetime.now().isoformat(timespec="seconds")
    base = {"schema_version": SCHEMA_VERSION, "parameters": parameters, "run_config": run_config,
            "config_signature": signature, "run_entropy": run_entropy, "created_at": created_at}
    pilot = existing.get("pilot") if existing else None
    L = existing.get("L") if existing else None
    _atomic_json(out_manifest, {**base, "status": "running", "phase": "pilot" if pilot is None else "production",
                                **({"pilot": pilot, "L": L} if pilot is not None else {})})
    try:
        if pilot is None:
            p = parameters
            pilot_params = {"N": N, "L": L0, "beta": p["beta_scaled"], "beta1": p["beta1_scaled"],
                            "alpha": p["alpha_scaled"], "alpha1": p["alpha1_scaled"], "mod": mod_run,
                            "pilot_therm": pilot_therm, "pilot_meas": pilot_meas,
                            "pilot_run_times": pilot_run_times, "n_workers": n_workers,
                            "epsilon": pilot_epsilon, "n_leapfrog": pilot_n_leapfrog,
                            "mass_a": mass_a, "mass_z": mass_z, "s_step": s_step,
                            "s_update_num": pilot_s_update_num, "s_max": s_max,
                            "heatbath_fraction": heatbath_fraction, "checkpoint_dir": checkpoint_dir,
                            "config_signature": signature, "run_entropy": run_entropy, "mul": mul}
            pilot = run_pilot(pilot_params)
            if not np.isfinite(pilot["xi"]) or pilot["xi"] <= 0:
                raise RuntimeError(f"pilot xi is not finite/positive: {pilot['xi']}")
            if pilot["xi"] > L0 // 2:
                raise RuntimeError(f"pilot xi={pilot['xi']:.4f} exceeds L0//2={L0 // 2}; increase L0")
            L = max(int(L_multiple * math.ceil(pilot["xi"])), 6)
            _atomic_json(out_manifest, {**base, "status": "running", "phase": "production", "pilot": pilot, "L": L})
        print(f"[mul={mul:.3f}] sampler={pilot['sampler_kind']} xi_pilot={pilot['xi']:.4f} L={L}")
        p = parameters
        common = {"N": N, "L": int(L), "beta": p["beta_scaled"], "beta1": p["beta1_scaled"],
                  "alpha": p["alpha_scaled"], "alpha1": p["alpha1_scaled"], "mod": mod_run,
                  "therm": therm, "meas": meas, "meas_int": meas_int, "flow_save_int": flow_save_int,
                  "epsilon": epsilon, "n_leapfrog": n_leapfrog, "mass_a": mass_a, "mass_z": mass_z,
                  "s_step": s_step, "s_update_num": s_update_num, "s_max": s_max,
                  "heatbath_fraction": heatbath_fraction, "mul_dir": out_dir, "npz_dir": npz_dir,
                  "checkpoint_dir": checkpoint_dir, "configs_per_file": configs_per_file,
                  "config_signature": signature, "run_entropy": run_entropy, "mul": mul}
        configs_per_chain = 0 if meas <= 0 else (meas - 1) // max(1, flow_save_int) + 1
        chains, pending = [], []
        for rid in range(run_times):
            params = dict(common, rid=rid, global_config_offset=rid * configs_per_chain,
                          seed=_seed(run_entropy, mul, rid, 0, "initial"))
            complete = _load_complete_chain(params)
            (pending if complete is None else chains).append(params if complete is None else complete)
        if pending:
            with ProcessPoolExecutor(max_workers=min(n_workers, len(pending))) as pool:
                futures = [pool.submit(measurement_chain, p) for p in pending]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"collecting mul={mul:.3f}"):
                    chains.append(fut.result())
        chains.sort(key=lambda c: c["rid"])
        measurement_path = os.path.join(npz_dir, "measurements.npz")
        tmp = measurement_path + ".tmp.npz"
        np.savez_compressed(tmp, S00=np.stack([c["S00"] for c in chains]), S10=np.stack([c["S10"] for c in chains]),
                            S01=np.stack([c["S01"] for c in chains]),
                            topological_charge=np.stack([c["topological_charge"] for c in chains]),
                            chain_ids=np.asarray([c["rid"] for c in chains], dtype=int),
                            sampler_kind=np.asarray(chains[0]["sampler_kind"]), L=np.asarray(L), N=np.asarray(N),
                            meas_int=np.asarray(meas_int), config_signature=np.asarray(signature))
        os.replace(tmp, measurement_path)
        xi_stats = corr_len_stats([c["S00"] for c in chains], [c["S10"] for c in chains],
                                  [c["S01"] for c in chains], L=L, c=tau_window_c)
        topo_stats = topological_stats([c["topological_charge"] for c in chains], L=L, c=tau_window_c)
        payload = {**base, "status": "ok", "completed_at": datetime.now().isoformat(timespec="seconds"),
                   "L": L, "pilot": pilot, "xi_results": xi_stats, "topological_results": topo_stats,
                   "measurements_path": os.path.relpath(measurement_path, out_dir),
                   "config_chunks": [chunk for chain in chains for chunk in chain["config_chunks"]],
                   "independent_runs": [{k: v for k, v in c.items()
                                         if k not in ("S00", "S10", "S01", "topological_charge", "config_chunks")}
                                        for c in chains]}
        _atomic_json(out_manifest, payload)
        status = "ok"
    except Exception as exc:
        payload = {**base, "status": "error", "phase": "resumable",
                   **({"pilot": pilot, "L": L} if pilot is not None else {}),
                   "error": {"message": str(exc), "traceback": traceback.format_exc()}}
        _atomic_json(out_manifest, payload)
        print(f"[error] {out_manifest}")
        raise
    print(f"[ok] {out_manifest}")
    return out_manifest, "ok"


def main():
    for mul in mul_list:
        run_one_mul(float(mul))


if __name__ == "__main__":
    main()
