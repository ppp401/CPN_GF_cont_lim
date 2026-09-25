"""Plot and fit gradient-flow continuum-limit observables."""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import odr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from func.obs import to_jsonable


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
min_t_over_a2_for_fit = 1.0


def mod_for(beta1_val):
    return 0 if abs(beta1_val) < 1e-8 else int(mod)


def parameter_folder(root):
    return os.path.join(
        root,
        f"N{N}_mod{mod_for(beta1)}",
        f"beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}",
    )


def flow_result_folder(root):
    flow_name = "covariant" if do_covariant else "normal"
    return os.path.join(parameter_folder(root), flow_name)


def load_entries():
    paths = glob.glob(os.path.join(flow_result_folder("gf_results"), "mul*.json"))
    wanted = None if mul_list is None else [float(x) for x in mul_list]
    entries = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("status") != "ok":
            continue
        if "flow_data_path" in data:
            npz_path = os.path.join(os.path.dirname(path), data["flow_data_path"])
            with np.load(npz_path, allow_pickle=False) as arrays:
                data["fixed_rho"] = {
                    "rho": arrays["rho"].tolist(),
                    "t_over_a2": arrays["target_times"].tolist(),
                    "tE_action": arrays["tE_action"].tolist(),
                    "tE_action_error": arrays["tE_action_error"].tolist(),
                }
        mul = float(data["parameters"]["mul"])
        if wanted is not None and not any(np.isclose(mul, x, atol=1e-9, rtol=0) for x in wanted):
            continue
        entries.append((mul, data))
    return sorted(entries, key=lambda item: item[0])


def _fit_odr(x, y, sx, sy):
    def linear(par, xx):
        return par[0] * xx + par[1]

    seed = np.polyfit(x, y, 1)
    result = odr.ODR(
        odr.RealData(x, y, sx=sx, sy=sy), odr.Model(linear), beta0=seed
    ).run()
    slope, intercept = result.beta
    slope_err, intercept_err = result.sd_beta
    residual = y - linear(result.beta, x)
    chi2 = float(np.sum((residual / sy) ** 2))
    dof = max(0, len(x) - 2)
    return {
        "slope": float(slope), "slope_error": float(slope_err),
        "continuum": float(intercept), "continuum_error": float(intercept_err),
        "chi2": chi2, "dof": dof,
    }


def _title():
    return (
        rf"$N={N}$, mod={mod_for(beta1)}, $\beta={beta:g}$, "
        rf"$\beta_1={beta1:g}$, $\alpha={alpha:g}$, $\alpha_1={alpha1:g}$"
    )


def plot_collapse(entries, out_dir, key, error_key, ylabel, stem, continuum_fits=None):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for mul, data in entries:
        block = data["fixed_rho"]
        rho = np.asarray(block["rho"], dtype=float)
        values = np.asarray(block[key], dtype=float)
        errors = np.asarray(block[error_key], dtype=float)
        ax.errorbar(rho, values, yerr=errors, marker="o", capsize=2, label=rf"mul={mul:g}")
    if continuum_fits is not None:
        continuum = [item for item in continuum_fits.values() if item["fit"] is not None]
        if continuum:
            rho = np.asarray([item["rho"] for item in continuum], dtype=float)
            values = np.asarray([item["fit"]["continuum"] for item in continuum], dtype=float)
            errors = np.asarray([item["fit"]["continuum_error"] for item in continuum], dtype=float)
            order = np.argsort(rho)
            ax.errorbar(
                rho[order], values[order], yerr=errors[order], fmt="k-s",
                capsize=2, linewidth=1.5, markersize=4,
                label=r"continuum extrapolation",
            )
    ax.set_xlabel(r"$\rho=t/\xi^2$")
    ax.set_ylabel(ylabel)
    ax.set_title("GF curve collapse: " + _title())
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{stem}_vs_t_over_xi2.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {path}")


def continuum_fits(entries, out_dir, key, error_key, ylabel, stem):
    os.makedirs(out_dir, exist_ok=True)
    reference_rho = np.asarray(entries[0][1]["fixed_rho"]["rho"], dtype=float)
    fits = {}
    for index, rho in enumerate(reference_rho):
        muls, xi, xi_err, y, y_err, t_values = [], [], [], [], [], []
        for mul, data in entries:
            block = data["fixed_rho"]
            rho_here = np.asarray(block["rho"], dtype=float)
            match = np.where(np.isclose(rho_here, rho, atol=1e-12, rtol=0))[0]
            if match.size != 1:
                continue
            j = int(match[0])
            muls.append(mul)
            xi.append(float(data["xi"]["mean"]))
            xi_err.append(float(data["xi"]["error"]))
            y.append(float(block[key][j]))
            y_err.append(float(block[error_key][j]))
            t_values.append(float(block["t_over_a2"][j]))

        muls = np.asarray(muls)
        xi, xi_err = np.asarray(xi), np.asarray(xi_err)
        y, y_err, t_values = np.asarray(y), np.asarray(y_err), np.asarray(t_values)
        x = 1.0 / xi ** 2
        x_err = 2.0 * xi_err / xi ** 3
        finite = np.isfinite(x) & np.isfinite(x_err) & np.isfinite(y) & np.isfinite(y_err)
        fit_mask = finite & (y_err > 0) & (x_err > 0) & (t_values >= min_t_over_a2_for_fit)

        fit = None
        if np.count_nonzero(fit_mask) >= 3:
            fit = _fit_odr(x[fit_mask], y[fit_mask], x_err[fit_mask], y_err[fit_mask])
        fits[f"rho_{rho:.6f}"] = {
            "rho": float(rho), "sqrt_rho": float(np.sqrt(rho)),
            "fit_threshold_t_over_a2": min_t_over_a2_for_fit,
            "n_points": int(np.count_nonzero(fit_mask)), "fit": fit,
            "mul": muls, "xi": xi, "xi_error": xi_err,
            "inverse_xi2": x, "inverse_xi2_error": x_err,
            "observable": y, "observable_error": y_err, "t_over_a2": t_values,
        }

        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        excluded = finite & ~fit_mask
        if np.any(excluded):
            ax.errorbar(
                x[excluded], y[excluded], xerr=x_err[excluded], yerr=y_err[excluded],
                fmt="o", mfc="none", color="0.55", capsize=2, label="shown, excluded from fit",
            )
        if np.any(fit_mask):
            ax.errorbar(
                x[fit_mask], y[fit_mask], xerr=x_err[fit_mask], yerr=y_err[fit_mask],
                fmt="o", capsize=2, label="fit points",
            )
        if fit is not None:
            xmax = max(float(np.max(x[finite])), 1e-12)
            xx = np.linspace(0.0, 1.05 * xmax, 200)
            yy = fit["continuum"] + fit["slope"] * xx
            ax.plot(
                xx, yy, "-",
                label=(
                    rf"$F_0={fit['continuum']:.5g}\pm{fit['continuum_error']:.2g}$" + "\n"
                    rf"$c={fit['slope']:.5g}\pm{fit['slope_error']:.2g}$"
                ),
            )
        ax.set_xlabel(r"$1/\xi^2$")
        ax.set_ylabel(ylabel)
        ax.set_title(rf"$t/\xi^2={rho:.3f}$: " + _title())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{stem}_rho{rho:.3f}_continuum.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] {path}")
    return fits


def main():
    entries = load_entries()
    if not entries:
        print(f"[no data] no GF result JSON under {flow_result_folder('gf_results')}")
        return
    out_dir = flow_result_folder("gf_plots")
    os.makedirs(out_dir, exist_ok=True)
    continuum_dir = os.path.join(out_dir, "continuum")

    action_fits = continuum_fits(
        entries, continuum_dir, "tE_action", "tE_action_error",
        r"$t\langle E_{\rm action}(t)\rangle$", "tE_action",
    )
    plot_collapse(
        entries, out_dir, "tE_action", "tE_action_error",
        r"$t\langle E_{\rm action}(t)\rangle$", "tE_action", action_fits,
    )
    summary = {
        "parameters": {"N": N, "mod": mod_for(beta1), "beta": beta, "beta1": beta1,
                       "alpha": alpha, "alpha1": alpha1},
        "flow_type": "covariant" if do_covariant else "normal",
        "min_t_over_a2_for_fit": min_t_over_a2_for_fit,
        "action": action_fits,
    }
    summary_path = os.path.join(out_dir, "continuum_fits.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(summary), fh, indent=2)
    print(f"[ok] {summary_path}")


if __name__ == "__main__":
    main()
