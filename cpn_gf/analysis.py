from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_config
from .io import atomic_json, atomic_npz
from .stats import analyze_flow, analyze_unflowed


def load_chain_values(run_dir, chains):
    run_dir = Path(run_dir)
    parts = sorted((run_dir / "observations").glob("flow_*.npz"))
    if not parts:
        raise RuntimeError(f"no flow chunks found below {run_dir}")
    per_chain = [[] for _ in range(chains)]
    per_chain_qs = [[] for _ in range(chains)]
    has_qs = None
    output_steps = times = None
    for path in parts:
        with np.load(path, allow_pickle=False) as data:
            if "values" in data.files:
                values = data["values"]
                if values.shape[-1] == 5:
                    values = np.concatenate((values, np.full(values.shape[:-1] + (1,), np.nan)), axis=-1)
            else:
                values = np.stack([data[name] for name in
                                   ("E_action", "S00", "S10", "S01", "Q_z", "Q_U")], axis=-1)
            if values.shape[0] != chains:
                raise RuntimeError(f"chain count mismatch in {path}")
            steps = data["output_steps"]
            if output_steps is None:
                output_steps = steps.copy()
                times = data["times"].copy()
            elif not np.array_equal(output_steps, steps):
                raise RuntimeError(f"flow time grid mismatch in {path}")
            for chain in range(chains):
                per_chain[chain].append(values[chain])
                if "Q_s" in data.files:
                    per_chain_qs[chain].append(data["Q_s"][chain])
            current_has_qs = "Q_s" in data.files
            if has_qs is None:
                has_qs = current_has_qs
            elif has_qs != current_has_qs:
                raise RuntimeError(f"inconsistent Q_s presence in {path}")
    qs = ([np.concatenate(x, axis=0) for x in per_chain_qs] if has_qs else None)
    return [np.concatenate(x, axis=0) for x in per_chain], qs, times, output_steps, parts


def analyze_run(run_dir, write=True):
    run_dir = Path(run_dir)
    with (run_dir / "manifest.json").open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    with np.load(run_dir / "scale.npz", allow_pickle=False) as scale:
        xi, xi_loo = float(scale["xi"]), scale["xi_loo"].copy()
    rho, L = np.asarray(manifest["flow"]["rho"], dtype=float), int(manifest["L"])
    chains = int(manifest["chains"])
    chain_values, chain_qs, times, output_steps, parts = load_chain_values(run_dir, chains)
    values, errors, target_times = analyze_flow(chain_values, times, rho, xi, xi_loo, L)
    unflowed, unflowed_errors, qs = analyze_unflowed(chain_values, chain_qs, L)
    names = ("E_action", "chi_m", "xi", "Q_z_mean", "chi_t_Q_z", "Q_U_mean", "chi_t_Q_U")
    result = {"rho": rho, "target_times": target_times, "times": times,
              "output_steps": output_steps,
              "n_samples_per_chain": np.asarray([len(x) for x in chain_values]),
              "chunk_paths": np.asarray([str(p.relative_to(run_dir)) for p in parts])}
    for index, name in enumerate(names):
        result[name] = values[:, index]
        result[f"{name}_error"] = errors[:, index]
    result["tE_action"] = target_times * result["E_action"]
    result["tE_action_error"] = target_times * result["E_action_error"]
    for index, name in enumerate(names):
        result[f"unflowed_{name}"] = np.asarray(unflowed[index])
        result[f"unflowed_{name}_error"] = np.asarray(unflowed_errors[index])
    result["unflowed_Q_s_applicable"] = np.asarray(qs is not None)
    if qs is not None:
        for key, value in qs.items():
            result[f"unflowed_Q_s_{key}"] = np.asarray(value)
    rel = np.abs(result["tE_action_error"] / result["tE_action"])
    xi_error = float(np.sqrt((len(xi_loo) - 1.0) / len(xi_loo)
                             * np.sum((xi_loo - xi_loo.mean()) ** 2)))
    summary = {"xi_scale": xi, "xi_scale_error": xi_error,
               "n_samples_per_chain": result["n_samples_per_chain"],
               "maximum_tE_relative_error": float(np.nanmax(rel)),
               "all_tE_relative_errors": rel,
               "converged": bool(np.all(np.isfinite(rel)) and
                                  np.all(rel <= manifest["sampling"]["relative_error"]))}
    summary["unflowed"] = {
        name: {"mean": result[f"unflowed_{name}"],
               "error": result[f"unflowed_{name}_error"]}
        for name in names if name != "E_action"
    }
    summary["unflowed"]["Q_s"] = ({"applicable": False} if qs is None else
                                      {"applicable": True, **qs})
    summary["flowed"] = {
        name: {"mean": result[name], "error": result[f"{name}_error"]}
        for name in names
    }
    summary["flowed"]["tE_action"] = {
        "mean": result["tE_action"], "error": result["tE_action_error"]}
    if write:
        atomic_npz(run_dir / "results.npz", **result)
        atomic_json(run_dir / "results.json", summary)
        _plot_run(run_dir, result)
    return result, summary


def _plot_run(run_dir, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.errorbar(result["rho"], result["tE_action"], yerr=result["tE_action_error"],
                marker="o", capsize=3)
    ax.set(xlabel=r"$\rho=t/\xi^2$", ylabel=r"$t\langle E(t)\rangle$")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "tE_action_vs_rho.png", dpi=180)
    plt.close(fig)


def analyze_path(path):
    """Analyze one mul run or every mul run in an experiment directory."""
    path = Path(path)
    if (path / "manifest.json").is_file():
        with (path / "manifest.json").open(encoding="utf-8") as fh:
            phase = json.load(fh).get("phase")
        if phase in ("pilot_complete", "pilot_error"):
            raise RuntimeError("pilot-only runs have no production observations to analyze")
        return {path.name: analyze_run(path, write=True)[1]}
    runs = []
    for run in sorted(path.glob("mul_*")):
        manifest_path = run / "manifest.json"
        if not manifest_path.is_file():
            continue
        with manifest_path.open(encoding="utf-8") as fh:
            phase = json.load(fh).get("phase")
        if phase not in ("pilot_complete", "pilot_error"):
            runs.append(run)
    if not runs:
        raise RuntimeError(f"no production runs found below {path}")
    summaries = {run.name: analyze_run(run, write=True)[1] for run in runs}
    _continuum_analysis(path, runs, summaries)
    atomic_json(path / "analysis.json", summaries)
    return summaries


def _continuum_analysis(root, runs, summaries):
    """Linear 1/xi^2 extrapolation of tE at fixed rho across mul runs."""
    from scipy import odr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root_config = Path(root) / "config.toml"
    configured_threshold = (float(load_config(root_config)["analysis"]["min_t_over_a2_for_fit"])
                            if root_config.is_file() else None)
    entries = []
    for run in runs:
        with (run / "manifest.json").open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        with np.load(run / "results.npz", allow_pickle=False) as data:
            entries.append((float(manifest["model"]["mul"]), summaries[run.name],
                            {k: data[k].copy() for k in ("rho", "target_times", "tE_action",
                                                         "tE_action_error")}))
    reference = entries[0][2]["rho"]
    if any(not np.array_equal(item[2]["rho"], reference) for item in entries[1:]):
        raise RuntimeError("cannot perform continuum analysis with different rho grids")
    minimum_flow_time = (configured_threshold if configured_threshold is not None else
                         float(json.loads((runs[0] / "manifest.json").read_text(encoding="utf-8"))
                               .get("analysis", {}).get("min_t_over_a2_for_fit", 1.0)))
    out, plot_dir = {}, Path(root) / "plots" / "continuum"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for index, rho in enumerate(reference):
        xi = np.asarray([item[1]["xi_scale"] for item in entries])
        xi_err = np.asarray([item[1]["xi_scale_error"] for item in entries])
        y = np.asarray([item[2]["tE_action"][index] for item in entries])
        yerr = np.asarray([item[2]["tE_action_error"][index] for item in entries])
        target = np.asarray([item[2]["target_times"][index] for item in entries])
        x, xerr = 1 / xi ** 2, 2 * xi_err / xi ** 3
        finite = (np.isfinite(x) & np.isfinite(xerr) & np.isfinite(y)
                  & np.isfinite(yerr) & (xerr > 0) & (yerr > 0))
        mask = finite & (target >= minimum_flow_time)
        fit = None
        if mask.sum() >= 3:
            model = odr.Model(lambda p, xx: p[0] * xx + p[1])
            seed = np.polyfit(x[mask], y[mask], 1)
            result = odr.ODR(odr.RealData(x[mask], y[mask], sx=xerr[mask], sy=yerr[mask]),
                             model, beta0=seed).run()
            fit = {"slope": float(result.beta[0]), "continuum": float(result.beta[1]),
                   "slope_error": float(result.sd_beta[0]),
                   "continuum_error": float(result.sd_beta[1])}
        out[f"rho_{rho:.6f}"] = {"rho": float(rho), "n_points": int(mask.sum()),
                                  "min_t_over_a2_for_fit": minimum_flow_time,
                                  "fit": fit, "inverse_xi2": x,
                                  "inverse_xi2_error": xerr, "observable": y,
                                  "observable_error": yerr}
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        excluded = finite & ~mask
        if excluded.any():
            ax.errorbar(x[excluded], y[excluded], xerr=xerr[excluded], yerr=yerr[excluded],
                        fmt="o", mfc="none", color="0.55", capsize=3,
                        label=rf"excluded: $t/a^2<{minimum_flow_time:g}$")
        if mask.any():
            ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask],
                        fmt="o", capsize=3, label="fit data")
        if fit is not None:
            xx = np.linspace(0, 1.05 * np.nanmax(x), 200)
            ax.plot(xx, fit["slope"] * xx + fit["continuum"])
        ax.set(xlabel=r"$1/\xi^2$", ylabel=r"$t\langle E(t)\rangle$",
               title=rf"$\rho={rho:.3f}$")
        ax.grid(alpha=0.25)
        if excluded.any():
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"tE_rho_{rho:.3f}.png", dpi=180)
        plt.close(fig)
    atomic_json(Path(root) / "continuum_fits.json", out)
