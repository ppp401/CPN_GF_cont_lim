"""
Observable utilities for the RefVil CPN continuum-limit study.

- ``corr_len_from_PP_corr_k``: second-moment correlation length from the
  Fourier-space PP structure factor returned by
  ``CPN_RefVil_HMCSampler.PP_corr_k``.
- ``compute_stats``: mean / integrated-autocorr-time / error of the mean
  for a list of independent per-chain sample arrays.
- ``to_jsonable``: recursive numpy -> JSON converter.
"""

import numpy as np

from cpn_gf.autocorr import integrated_autocorr_time


def corr_len_from_PP_corr_k(S, L):
    """
    Second-moment correlation length from the k-space PP correlator S.

    Parameters
    ----------
    S : ndarray, shape (Lx, Ly)
        Output of ``CPN_RefVil_HMCSampler.PP_corr_k`` (already averaged over
        the ensemble if desired).
    L : int
        Linear lattice size (used for the lattice momentum k1 = 2*pi/L).

    Returns
    -------
    xi_x, xi_y, xi : float
        Correlation length along x, y, and their mean.
    """
    S = np.asarray(S)
    S00 = S[0, 0]
    S10 = S[1, 0]
    S01 = S[0, 1]

    if S10 <= 0 or S01 <= 0 or S00 <= 0:
        return float("nan"), float("nan"), float("nan")

    denom = 2.0 * np.sin(np.pi / L)
    xi_x = np.sqrt(max(S00 / S10 - 1.0, 0.0)) / denom
    xi_y = np.sqrt(max(S00 / S01 - 1.0, 0.0)) / denom
    return float(xi_x), float(xi_y), float(0.5 * (xi_x + xi_y))


def compute_stats(samples_list, c=5.0):
    """
    Aggregate a list of independent per-chain sample arrays.

    Each entry of ``samples_list`` is a 1D array of measurements from one
    independent chain (same observable, same length is NOT required). The
    integrated autocorrelation time is computed per chain along the sample
    axis and then averaged; the error is the autocorrelation-corrected
    standard error of the mean.

    Returns
    -------
    mean : float
    tau  : float     (mean integrated autocorrelation time)
    error: float
    n_total : int    (total number of samples across chains)
    """
    flat = np.concatenate([np.asarray(s, dtype=float) for s in samples_list])
    n_total = flat.size
    if n_total < 2:
        return float(flat.mean()), float("nan"), float("nan"), int(n_total)
    if np.all(flat == flat[0]):
        return float(flat[0]), 0.5, 0.0, int(n_total)

    taus = []
    for s in samples_list:
        s = np.asarray(s, dtype=float)
        if s.size >= 2:
            taus.append(0.5 if np.all(s == s[0])
                        else float(integrated_autocorr_time(s, c=c)))
    tau = float(np.mean(taus)) if taus else float("nan")

    std = float(np.std(flat, ddof=1))
    if not np.isfinite(tau) or tau <= 0:
        n_eff = n_total
    else:
        n_eff = n_total / (2.0 * tau)
    error = std / np.sqrt(n_eff)

    return float(flat.mean()), tau, error, int(n_total)


def _stats_block(flat, c):
    """mean / tau / error of the mean for one flat 1D array."""
    n_total = flat.size
    if n_total < 2:
        return {"mean": float(flat.mean()), "tau": float("nan"),
                "error": float("nan"), "n_total": int(n_total)}
    taus = [float(integrated_autocorr_time(flat, c=c))]
    tau = float(np.mean(taus))
    std = float(np.std(flat, ddof=1))
    n_eff = n_total / (2.0 * tau) if (np.isfinite(tau) and tau > 0) else n_total
    return {"mean": float(flat.mean()), "tau": tau,
            "error": std / np.sqrt(n_eff), "n_total": int(n_total)}


def corr_len_stats(S00_chains, S10_chains, S01_chains, L, c=5.0):
    """
    Correlation-length statistics from per-sample S[0,0], S[1,0], S[0,1].

    The mean correlation length is computed from the *means* of the structure
    factor modes (xi_x = sqrt(<S00>/<S10> - 1) / (2 sin(pi/L)) etc.), which is
    the unbiased estimator for the second-moment correlation length. Errors are
    propagated from err(<S00>), err(<S10>), err(<S01>) via the chain rule,
    including the cross-covariances Cov(<S00>,<S10>) and Cov(<S00>,<S01>)
    estimated from the sample Pearson correlations.

    The integrated autocorrelation time of xi is computed directly on the
    per-sample xi series (with non-finite entries dropped), giving a direct
    measure of the xi observable's autocorrelation.

    Parameters
    ----------
    S00_chains, S10_chains, S01_chains : list of 1D arrays
        Per-chain per-sample values of S[0,0], S[1,0], S[0,1].
    L : int
        Linear lattice size.
    c : float
        Window cutoff for ``integrated_autocorr_time``.

    Returns
    -------
    dict with keys "xi_x", "xi_y", "xi", "chi_m", "S00", "S10", "S01".
    Each value is itself a dict {mean, tau, error, n_total}.
    """
    S00 = np.concatenate([np.asarray(s, dtype=float) for s in S00_chains])
    S10 = np.concatenate([np.asarray(s, dtype=float) for s in S10_chains])
    S01 = np.concatenate([np.asarray(s, dtype=float) for s in S01_chains])
    n_total = S00.size

    s00 = _stats_block(S00, c)
    s10 = _stats_block(S10, c)
    s01 = _stats_block(S01, c)

    m00, m10, m01 = s00["mean"], s10["mean"], s01["mean"]
    e00, e10, e01 = s00["error"], s10["error"], s01["error"]

    denom = 2.0 * np.sin(np.pi / L)

    # ---- xi_x mean and chain-rule error (depends on S00, S10) ----
    if m10 > 0 and m00 / m10 - 1.0 > 0:
        xi_x_mean = float(np.sqrt(m00 / m10 - 1.0) / denom)
        g = np.sqrt(m00 / m10 - 1.0)
        d_xi_x_d_S00 = 1.0 / (2.0 * g * m10 * denom)
        d_xi_x_d_S10 = -m00 / (2.0 * g * m10**2 * denom)
        rho_01 = _pearson(S00, S10)
        cov_00_10 = rho_01 * e00 * e10
        err_xi_x_sq = (d_xi_x_d_S00**2) * (e00**2) \
            + (d_xi_x_d_S10**2) * (e10**2) \
            + 2.0 * d_xi_x_d_S00 * d_xi_x_d_S10 * cov_00_10
        err_xi_x = float(np.sqrt(max(err_xi_x_sq, 0.0)))
    else:
        xi_x_mean = float("nan")
        err_xi_x = float("nan")

    # ---- xi_y mean and chain-rule error (depends on S00, S01) ----
    if m01 > 0 and m00 / m01 - 1.0 > 0:
        xi_y_mean = float(np.sqrt(m00 / m01 - 1.0) / denom)
        g = np.sqrt(m00 / m01 - 1.0)
        d_xi_y_d_S00 = 1.0 / (2.0 * g * m01 * denom)
        d_xi_y_d_S01 = -m00 / (2.0 * g * m01**2 * denom)
        rho_02 = _pearson(S00, S01)
        cov_00_01 = rho_02 * e00 * e01
        err_xi_y_sq = (d_xi_y_d_S00**2) * (e00**2) \
            + (d_xi_y_d_S01**2) * (e01**2) \
            + 2.0 * d_xi_y_d_S00 * d_xi_y_d_S01 * cov_00_01
        err_xi_y = float(np.sqrt(max(err_xi_y_sq, 0.0)))
    else:
        xi_y_mean = float("nan")
        err_xi_y = float("nan")

    # ---- xi mean = (xi_x + xi_y)/2 ; option (b): xi_x, xi_y independent ----
    if np.isfinite(xi_x_mean) and np.isfinite(xi_y_mean):
        xi_mean = float(0.5 * (xi_x_mean + xi_y_mean))
        err_xi = float(0.5 * np.sqrt(err_xi_x**2 + err_xi_y**2))
    else:
        xi_mean = float("nan")
        err_xi = float("nan")

    # ---- tau on per-sample xi series (filter non-finite) ----
    ratio_x = S00 / np.maximum(S10, 1e-300) - 1.0
    ratio_y = S00 / np.maximum(S01, 1e-300) - 1.0
    xi_x_full = np.where(ratio_x > 0, np.sqrt(np.where(ratio_x > 0, ratio_x, 0.0)), np.nan) / denom
    xi_y_full = np.where(ratio_y > 0, np.sqrt(np.where(ratio_y > 0, ratio_y, 0.0)), np.nan) / denom
    xi_x_series = xi_x_full[np.isfinite(xi_x_full)]
    xi_y_series = xi_y_full[np.isfinite(xi_y_full)]

    tau_xi_x = _tau_of_series(xi_x_series, c)
    tau_xi_y = _tau_of_series(xi_y_series, c)
    # for xi = (xi_x+xi_y)/2 we need the per-sample average where both are finite
    both_finite = np.isfinite(xi_x_full) & np.isfinite(xi_y_full)
    if both_finite.sum() >= 2:
        xi_avg_series = 0.5 * (xi_x_full[both_finite] + xi_y_full[both_finite])
        tau_xi = _tau_of_series(xi_avg_series, c)
    else:
        tau_xi = float("nan")

    return {
        "xi_x": {"mean": xi_x_mean, "tau": tau_xi_x, "error": err_xi_x, "n_total": int(n_total)},
        "xi_y": {"mean": xi_y_mean, "tau": tau_xi_y, "error": err_xi_y, "n_total": int(n_total)},
        "xi":   {"mean": xi_mean,   "tau": tau_xi,   "error": err_xi,   "n_total": int(n_total)},
        # chi_m is exactly S[0,0]
        "chi_m": dict(s00),
        "S00": dict(s00),
        "S10": dict(s10),
        "S01": dict(s01),
    }


def _pearson(a, b):
    """Sample Pearson correlation, with NaN if undefined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    am = a - a.mean()
    bm = b - b.mean()
    denom = np.sqrt(np.sum(am**2) * np.sum(bm**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(am * bm) / denom)


def _tau_of_series(x, c):
    """Integrated autocorr time on a finite 1D series; NaN if too few points."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float("nan")
    return float(integrated_autocorr_time(x, c=c))


def to_jsonable(obj):
    """Recursively convert numpy types/arrays to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj
