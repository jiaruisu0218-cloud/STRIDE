# dataset_analyzer.py
# ------------------------------------------------------------
# Usage:
#   1) Modify DATA_PATH below
#   2) python dataset_analyzer.py
#
# Supported dataset formats:
#   - CSV: should contain columns for features + target
#   - NPZ: should contain arrays: X, y (or inputs, outputs)
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# =============== USER CONFIG (modify here) ===================
# ============================================================
DATA_PATH = r"F:\projrct\LLM-SR-main\LLM-SR-main\data\oscillator2\train.csv"   # edit path for your dataset
TARGET_COL = None  # target column name; None -> last column
FEATURE_COLS = None  # feature columns; None -> all but last
TOP_K_TERMS = 5  # max leading terms in hint output
PRINT_FULL_ARRAYS = False  # print full arrays for debugging
STLSQ_THRESHOLD = 0.001  # STLSQ coefficient shrinkage threshold
STLSQ_MAX_ITER = 100  # STLSQ max iterations
REGRESSION_METHOD = "stlsq"  # "stlsq" / "lasso" / "auto"
INCLUDE_NONLIN_CROSS = True  # add nonlinear cross terms (e.g. sin(x_i)*x_j)
ALLOW_SELF_CROSS = True  # allow self-interaction terms (e.g. sin(x_i)*x_i)
SELF_CROSS_MAX = 100  # max self-interaction terms per variable
BOOTSTRAP_RUNS = 100  # bootstrap / subsample runs
BOOTSTRAP_MODE = "subsample"  # "bootstrap" (with replacement) / "subsample"
SUBSAMPLE_FRAC = 0.6  # subsample fraction (subsample mode)
FREQ_THRESHOLD = 0.8  # frequency threshold to include term in hint
# ============================================================
SELECT_EPS = 1e-8  # min |coefficient| to count as selected
EVIDENCE_FILTER_ENABLED = False  # enable evidence-based filtering
PERIODIC_PEAK_RATIO = 4.0  # trig evidence: spectrum peak / median ratio
PERIODIC_MIN_SAMPLES = 2000  # trig evidence: minimum samples
EXP_CORR_THRESHOLD = 0.7  # exp evidence legacy threshold (deprecated)
EXP_DELTA_R2 = 0.2  # exp evidence: semi-log R^2 gain vs linear
LOG_INV_IQR_CORR = 0.5  # log/inv evidence: IQR growth vs |x|
HEAVY_TAIL_RATIO = 2.5  # log/inv evidence: heavy-tail ratio


# ============================================================
# Robust stats
# ============================================================
def _robust_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "std": np.nan, "p01": np.nan, "p50": np.nan, "p99": np.nan, "iqr": np.nan}
    p01, p50, p99 = np.percentile(x, [1, 50, 99])
    q25, q75 = np.percentile(x, [25, 75])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x) + 1e-12),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "iqr": float((q75 - q25) + 1e-12),
    }


# ============================================================
# Scale analysis
# ============================================================
def _scale_analysis(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"y": _robust_stats(y), "x": {}}
    for j, name in enumerate(var_names):
        out["x"][name] = _robust_stats(X[:, j])

    y_mean = out["y"]["mean"]
    y_std = out["y"]["std"]
    out["derived"] = {
        "y_is_near_zero_mean": bool(abs(y_mean) <= 0.05 * y_std),
        "suggest_bias_term": bool(abs(y_mean) > 0.05 * y_std),
        "x_scales": {name: float(out["x"][name]["std"]) for name in var_names},
    }
    return out


# ============================================================
# Symmetry analysis
# ============================================================
def _symmetry_score_by_sign_flip(
    X: np.ndarray,
    y: np.ndarray,
    j: int,
    bins: int = 25,
) -> Dict[str, Any]:
    x = X[:, j]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    yy = y[finite]
    if x.size < 200:
        return {"ok": False, "reason": "too_few_points"}

    ax = np.abs(x)
    edges = np.quantile(ax, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 6:
        return {"ok": False, "reason": "degenerate_bins"}

    def bin_mean(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        counts = []
        for k in range(edges.size - 1):
            m = mask & (ax >= edges[k]) & (ax < edges[k + 1])
            if np.sum(m) < 10:
                means.append(np.nan)
                counts.append(0)
            else:
                means.append(np.mean(yy[m]))
                counts.append(int(np.sum(m)))
        return np.asarray(means), np.asarray(counts)

    mp, cp = bin_mean(x > 0)
    mn, cn = bin_mean(x < 0)

    valid = np.isfinite(mp) & np.isfinite(mn) & (cp >= 10) & (cn >= 10)
    if np.sum(valid) < 5:
        return {"ok": False, "reason": "insufficient_symmetric_coverage"}

    denom = np.nanmean(np.abs(np.r_[mp[valid], mn[valid]])) + 1e-12
    odd_err = float(np.nanmean(np.abs(mp[valid] + mn[valid])) / denom)
    even_err = float(np.nanmean(np.abs(mp[valid] - mn[valid])) / denom)

    verdict = "odd_like" if odd_err < even_err else "even_like"
    strength = float(abs(even_err - odd_err) / (min(even_err, odd_err) + 1e-12))

    out = {
        "ok": True,
        "odd_err": odd_err,
        "even_err": even_err,
        "verdict": verdict,
        "strength": strength,
    }
    if PRINT_FULL_ARRAYS:
        out["bin_edges_absx"] = edges.tolist()
        out["mean_pos"] = mp.tolist()
        out["mean_neg"] = mn.tolist()
        out["count_pos"] = cp.tolist()
        out["count_neg"] = cn.tolist()
    return out


def _symmetry_analysis(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for j, name in enumerate(var_names):
        out[name] = _symmetry_score_by_sign_flip(X, y, j=j)
    return out


# ============================================================
# Evidence checks for nonlinear terms
# ============================================================
def _spectral_peak_ratio(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < PERIODIC_MIN_SAMPLES:
        return 0.0
    y = y - np.mean(y)
    spec = np.fft.rfft(y)
    power = np.abs(spec) ** 2
    if power.size < 3:
        return 0.0
    power[0] = 0.0
    peak = float(np.max(power))
    median = float(np.median(power[1:]) + 1e-12)
    return peak / median


def _periodicity_evidence(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_var": {}, "any": False}
    for j, vn in enumerate(var_names):
        x = X[:, j]
        finite = np.isfinite(x) & np.isfinite(y)
        if np.sum(finite) < PERIODIC_MIN_SAMPLES:
            out["by_var"][vn] = {"ok": False, "ratio": 0.0}
            continue
        idx = np.argsort(x[finite])
        yy = y[finite][idx]
        ratio = _spectral_peak_ratio(yy)
        ok = ratio >= PERIODIC_PEAK_RATIO
        out["by_var"][vn] = {"ok": bool(ok), "ratio": float(ratio)}
        out["any"] = out["any"] or ok
    return out


def _exp_evidence(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_var": {}, "any": False}
    logy = np.log(np.abs(y) + 1e-12)
    for j, vn in enumerate(var_names):
        x = X[:, j]
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(logy)
        if np.sum(finite) < 50:
            out["by_var"][vn] = {"ok": False, "r2_lin": 0.0, "r2_semilog": 0.0, "delta_r2": 0.0}
            continue
        xv = x[finite]
        yv = y[finite]
        ly = logy[finite]
        r2_lin = float(np.corrcoef(xv, yv)[0, 1]) ** 2
        r2_semilog = float(np.corrcoef(xv, ly)[0, 1]) ** 2
        delta_r2 = r2_semilog - r2_lin
        ok = delta_r2 >= EXP_DELTA_R2
        out["by_var"][vn] = {
            "ok": bool(ok),
            "r2_lin": r2_lin,
            "r2_semilog": r2_semilog,
            "delta_r2": delta_r2,
        }
        out["any"] = out["any"] or ok
    return out


def _iqr_growth_evidence(x: np.ndarray, y: np.ndarray, bins: int = 6) -> bool:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    yy = y[finite]
    if x.size < 200:
        return False
    ax = np.abs(x)
    edges = np.quantile(ax, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 4:
        return False
    centers = []
    iqrs = []
    for k in range(edges.size - 1):
        m = (ax >= edges[k]) & (ax < edges[k + 1])
        if np.sum(m) < 10:
            continue
        q25, q75 = np.percentile(yy[m], [25, 75])
        centers.append(0.5 * (edges[k] + edges[k + 1]))
        iqrs.append(q75 - q25)
    if len(centers) < 4:
        return False
    corr = float(np.corrcoef(np.asarray(centers), np.asarray(iqrs))[0, 1])
    return corr >= LOG_INV_IQR_CORR


def _log_inv_evidence(X: np.ndarray, y: np.ndarray, var_names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_var": {}, "any": False, "heavy_tail": False}
    ys = _robust_stats(y)
    denom = (ys["p50"] - ys["p01"]) + 1e-12
    tail_ratio = (ys["p99"] - ys["p50"]) / denom
    out["heavy_tail"] = bool(tail_ratio >= HEAVY_TAIL_RATIO)
    for j, vn in enumerate(var_names):
        ok = _iqr_growth_evidence(X[:, j], y)
        out["by_var"][vn] = {"ok": bool(ok)}
        out["any"] = out["any"] or ok
    return out


def _extract_vars_in_term(term: str, var_names: List[str]) -> List[str]:
    hits = []
    for vn in var_names:
        if re.search(rf"\\b{re.escape(vn)}\\b", term):
            hits.append(vn)
    return hits


def _evidence_ok_for_vars(vars_in_term: List[str], by_var: Dict[str, Any], any_ok: bool) -> bool:
    if not vars_in_term:
        return bool(any_ok)
    return any(by_var.get(v, {}).get("ok", False) for v in vars_in_term)


def _term_allowed_with_evidence(
    term: str,
    var_names: List[str],
    trig_ev: Dict[str, Any],
    exp_ev: Dict[str, Any],
    loginv_ev: Dict[str, Any],
) -> bool:
    lower = term.lower()
    vars_in_term = _extract_vars_in_term(term, var_names)
    if "sin(" in lower or "cos(" in lower:
        return _evidence_ok_for_vars(vars_in_term, trig_ev["by_var"], trig_ev["any"])
    if "exp(" in lower:
        return _evidence_ok_for_vars(vars_in_term, exp_ev["by_var"], exp_ev["any"])
    if "log(" in lower or "inv(" in lower:
        if loginv_ev.get("heavy_tail", False):
            return True
        return _evidence_ok_for_vars(vars_in_term, loginv_ev["by_var"], loginv_ev["any"])
    return True

# ============================================================
# Feature dictionary builder
# ============================================================
def _build_dictionary(
    X: np.ndarray,
    var_names: List[str],
    max_degree: int = 3,
    include_abs: bool = True,
    include_cross: bool = True,
    include_nonlin: bool = True,
    include_nonlin_cross: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    feats = []
    names = []
    n, d = X.shape

    nonlin_feats = []
    nonlin_names = []

    for j, vn in enumerate(var_names):
        x = X[:, j]
        feats.append(x)
        names.append(vn)
        for p in range(2, max_degree + 1):
            feats.append(x ** p)
            names.append(f"{vn}**{p}")
        if include_abs:
            feats.append(np.abs(x) * x)
            names.append(f"abs({vn})*{vn}")
        if include_nonlin:
            x_clip = np.clip(x, -20.0, 20.0)
            sinx = np.sin(x)
            cosx = np.cos(x)
            expx = np.exp(x_clip)
            logx = np.log(np.abs(x) + 1e-12)
            invx = np.where(np.abs(x) > 1e-12, 1.0 / x, 0.0)

            feats.append(sinx)
            names.append(f"sin({vn})")
            feats.append(cosx)
            names.append(f"cos({vn})")
            feats.append(expx)
            names.append(f"exp({vn})")
            feats.append(logx)
            names.append(f"log(abs({vn})+eps)")
            feats.append(invx)
            names.append(f"inv({vn})")

            nonlin_feats.append(
                {
                    "var": vn,
                    "sin": sinx,
                    "cos": cosx,
                    "exp": expx,
                    "log": logx,
                    "inv": invx,
                }
            )
            nonlin_names.append(
                {
                    "var": vn,
                    "sin": f"sin({vn})",
                    "cos": f"cos({vn})",
                    "exp": f"exp({vn})",
                    "log": f"log(abs({vn})+eps)",
                    "inv": f"inv({vn})",
                }
            )

    if include_cross and d >= 2:
        for i in range(d):
            for j in range(i + 1, d):
                xi = X[:, i]
                xj = X[:, j]
                feats.append(xi * xj)
                names.append(f"{var_names[i]}*{var_names[j]}")
                if max_degree >= 3:
                    feats.append((xi ** 2) * xj)
                    names.append(f"{var_names[i]}**2*{var_names[j]}")
                    feats.append(xi * (xj ** 2))
                    names.append(f"{var_names[i]}*{var_names[j]}**2")

    if include_nonlin and include_nonlin_cross and d >= 2:
        for i in range(d):
            for j in range(d):
                if i == j:
                    if not ALLOW_SELF_CROSS or SELF_CROSS_MAX <= 0:
                        continue
                    nli = nonlin_feats[i]
                    xi = X[:, i]
                    self_terms = [
                        (nli["sin"] * xi, f"{nonlin_names[i]['sin']}*{var_names[i]}"),
                        (nli["cos"] * xi, f"{nonlin_names[i]['cos']}*{var_names[i]}"),
                        (nli["exp"] * xi, f"{nonlin_names[i]['exp']}*{var_names[i]}"),
                        (nli["log"] * xi, f"{nonlin_names[i]['log']}*{var_names[i]}"),
                        (nli["inv"] * xi, f"{nonlin_names[i]['inv']}*{var_names[i]}"),
                        (nli["sin"] * nli["cos"], f"{nonlin_names[i]['sin']}*{nonlin_names[i]['cos']}"),
                    ]
                    added = 0
                    for feat, name in self_terms:
                        feats.append(feat)
                        names.append(name)
                        added += 1
                        if added >= SELF_CROSS_MAX:
                            break
                    continue
                xi = X[:, i]
                nli = nonlin_feats[i]
                nlj = nonlin_feats[j]

                feats.append(nli["sin"] * xj)
                names.append(f"{nonlin_names[i]['sin']}*{var_names[j]}")
                feats.append(nli["cos"] * xj)
                names.append(f"{nonlin_names[i]['cos']}*{var_names[j]}")
                feats.append(nli["exp"] * xj)
                names.append(f"{nonlin_names[i]['exp']}*{var_names[j]}")
                feats.append(nli["log"] * xj)
                names.append(f"{nonlin_names[i]['log']}*{var_names[j]}")
                feats.append(nli["inv"] * xj)
                names.append(f"{nonlin_names[i]['inv']}*{var_names[j]}")

                feats.append(nli["sin"] * nlj["cos"])
                names.append(f"{nonlin_names[i]['sin']}*{nonlin_names[j]['cos']}")

    Phi = np.column_stack(feats).astype(float)

    good = np.isfinite(Phi).all(axis=0) & (np.std(Phi, axis=0) > 1e-12)
    Phi = Phi[:, good]
    names = [nm for nm, g in zip(names, good) if g]
    return Phi, names


def _ridge_closed_form(Phi: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    mu = Phi.mean(axis=0, keepdims=True)
    sd = Phi.std(axis=0, keepdims=True) + 1e-12
    Z = (Phi - mu) / sd
    y0 = y - np.mean(y)

    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    b = Z.T @ y0
    w = np.linalg.solve(A, b)
    return w


def _stlsq(
    Z: np.ndarray,
    y0: np.ndarray,
    threshold: float,
    max_iter: int,
) -> np.ndarray:
    """
    Sequential Thresholded Least Squares (STLSQ) on standardized features.
    """
    w = np.linalg.lstsq(Z, y0, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(w) < threshold
        if not np.any(small):
            break
        w[small] = 0.0
        keep = ~small
        if np.sum(keep) == 0:
            break
        w_keep = np.linalg.lstsq(Z[:, keep], y0, rcond=None)[0]
        w = np.zeros_like(w)
        w[keep] = w_keep
    return w


def _fit_weights(
    Z: np.ndarray,
    y0: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, str]:
    method = method.lower()
    if method not in {"stlsq", "lasso", "auto"}:
        method = "stlsq"

    if method in {"lasso", "auto"}:
        try:
            from sklearn.linear_model import LassoCV

            model = LassoCV(cv=5, n_alphas=30, max_iter=5000, random_state=0)
            model.fit(Z, y0)
            return model.coef_, "lasso_cv"
        except Exception:
            if method == "lasso":
                raise

    return _stlsq(Z, y0, threshold=STLSQ_THRESHOLD, max_iter=STLSQ_MAX_ITER), "stlsq"


def _bootstrap_selection_frequency(
    Z: np.ndarray,
    y0: np.ndarray,
    *,
    runs: int,
    mode: str,
    subsample_frac: float,
    method: str,
) -> np.ndarray:
    n = Z.shape[0]
    if n < 10 or runs <= 0:
        return np.zeros(Z.shape[1], dtype=float)

    rng = np.random.default_rng(0)
    counts = np.zeros(Z.shape[1], dtype=float)
    mode = mode.lower()
    for _ in range(runs):
        if mode == "bootstrap":
            idx = rng.integers(0, n, size=n)
        else:
            k = max(1, int(n * subsample_frac))
            idx = rng.choice(n, size=k, replace=False)
        w, _ = _fit_weights(Z[idx], y0[idx], method=method)
        counts += (np.abs(w) > SELECT_EPS).astype(float)
    return counts / float(runs)


def _dominant_factor_analysis(
    X: np.ndarray,
    y: np.ndarray,
    var_names: List[str],
    top_k: int = 8,
) -> Dict[str, Any]:
    Phi, feat_names = _build_dictionary(
        X,
        var_names,
        max_degree=3,
        include_abs=True,
        include_cross=True,
        include_nonlin=True,
        include_nonlin_cross=INCLUDE_NONLIN_CROSS,
    )
    finite = np.isfinite(Phi).all(axis=1) & np.isfinite(y)
    Phi = Phi[finite]
    yy = y[finite]
    if Phi.shape[0] < 200 or Phi.shape[1] < 3:
        return {"ok": False, "reason": "too_small_after_filtering", "Phi_shape": list(Phi.shape)}

    mu = Phi.mean(axis=0, keepdims=True)
    sd = Phi.std(axis=0, keepdims=True) + 1e-12
    Z = (Phi - mu) / sd
    yy0 = yy - np.mean(yy)

    freq = _bootstrap_selection_frequency(
        Z,
        yy0,
        runs=BOOTSTRAP_RUNS,
        mode=BOOTSTRAP_MODE,
        subsample_frac=SUBSAMPLE_FRAC,
        method=REGRESSION_METHOD,
    )

    try:
        weights, method = _fit_weights(Z, yy0, method=REGRESSION_METHOD)
    except Exception:
        weights = _ridge_closed_form(Phi, yy, lam=1e-3)
        method = "ridge_closed_form"

    imp = np.abs(weights)
    idx = np.argsort(imp)[::-1]
    top = []
    allowed = freq >= FREQ_THRESHOLD
    allowed_idx = [i for i in idx if allowed[int(i)]]
    if not allowed_idx:
        return {
            "ok": False,
            "reason": "no_terms_above_frequency_threshold",
            "method": method,
            "Phi_shape": list(Phi.shape),
            "freq_threshold": float(FREQ_THRESHOLD),
        }

    if EVIDENCE_FILTER_ENABLED:
        trig_ev = _periodicity_evidence(X, y, var_names)
        exp_ev = _exp_evidence(X, y, var_names)
        loginv_ev = _log_inv_evidence(X, y, var_names)
    else:
        trig_ev = {"by_var": {}, "any": True}
        exp_ev = {"by_var": {}, "any": True}
        loginv_ev = {"by_var": {}, "any": True, "heavy_tail": True}
    for k in allowed_idx[: min(top_k, len(allowed_idx))]:
        term = feat_names[int(k)]
        if EVIDENCE_FILTER_ENABLED and not _term_allowed_with_evidence(
            term, var_names, trig_ev, exp_ev, loginv_ev
        ):
            continue
        top.append(
            {
                "term": term,
                "importance": float(imp[int(k)]),
                "weight": float(weights[int(k)]),
                "frequency": float(freq[int(k)]),
            }
        )

    if not top:
        return {
            "ok": False,
            "reason": "no_terms_after_evidence_filter",
            "method": method,
            "Phi_shape": list(Phi.shape),
            "freq_threshold": float(FREQ_THRESHOLD),
            "evidence": {
                "trig": trig_ev,
                "exp": exp_ev,
                "log_inv": loginv_ev,
            },
        }

    return {
        "ok": True,
        "method": method,
        "top_terms": top,
        "Phi_shape": list(Phi.shape),
        "freq_threshold": float(FREQ_THRESHOLD),
        "evidence": {
            "trig": trig_ev,
            "exp": exp_ev,
            "log_inv": loginv_ev,
        },
    }


def _format_prompt_hint(
    scale: Dict[str, Any],
    symmetry: Dict[str, Any],
    dominant: Dict[str, Any],
    var_names: List[str],
) -> str:
    lines = []
    lines.append("[DATA_HINT]")

    y = scale.get("y", {})
    derived = scale.get("derived", {})
    lines.append(
        f"y_mean={y.get('mean', np.nan):.4g}, y_std={y.get('std', np.nan):.4g}, "
        f"y_p01={y.get('p01', np.nan):.4g}, y_p99={y.get('p99', np.nan):.4g}"
    )
    lines.append(f"bias_term_suggested={bool(derived.get('suggest_bias_term', False))}")

    sym_msgs = []
    for vn in var_names:
        info = symmetry.get(vn, {})
        if not info or not info.get("ok", False):
            continue
        sym_msgs.append(f"{vn}:{info['verdict']} (odd_err={info['odd_err']:.3g}, even_err={info['even_err']:.3g})")
    if sym_msgs:
        lines.append("symmetry=" + "; ".join(sym_msgs))

    if dominant.get("ok", False):
        top = dominant.get("top_terms", [])
        dom = [t["term"] for t in top[:6]]
        lines.append("dominant_terms=" + ", ".join(dom))

    sig_vars = ", ".join(var_names + ["params", "np"])
    lines.append(f"constraint=use_only_signature_vars({sig_vars}); do_not_invent_external_vars")
    lines.append("format=require_return_statement; output must be executable function body")
    return "\n".join(lines) + "\n"


# ============================================================
# Main API
# ============================================================
@dataclass
class DataHint:
    scale: Dict[str, Any]
    symmetry: Dict[str, Any]
    dominant_factors: Dict[str, Any]
    prompt_hint: str


def analyze_io_dataset(
    X: np.ndarray,
    y: np.ndarray,
    var_names: Optional[List[str]] = None,
    *,
    top_k: int = 8,
) -> DataHint:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]

    if var_names is None:
        var_names = [f"x{i}" for i in range(X.shape[1])]

    scale = _scale_analysis(X, y, var_names)
    symmetry = _symmetry_analysis(X, y, var_names)
    dominant = _dominant_factor_analysis(X, y, var_names, top_k=top_k)
    prompt_hint = _format_prompt_hint(scale, symmetry, dominant, var_names)

    return DataHint(scale=scale, symmetry=symmetry, dominant_factors=dominant, prompt_hint=prompt_hint)


# ============================================================
# Dataset loading
# ============================================================
def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_PATH not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "X" in data and "y" in data:
            X, y = data["X"], data["y"]
        elif "inputs" in data and "outputs" in data:
            X, y = data["inputs"], data["outputs"]
        else:
            raise KeyError(f"NPZ must contain X,y or inputs,outputs. keys={list(data.keys())}")
        var_names = [f"x{i}" for i in range(X.shape[1])]
        return X, y, var_names

    if ext == ".csv":
        # Rule: all columns except last are X; last column is y.
        try:
            import pandas as pd

            df = pd.read_csv(path)
            if df.shape[1] < 2:
                raise ValueError(f"CSV must have at least 2 columns (X..., y). Got {df.shape[1]}")

            X_df = df.iloc[:, :-1]
            y_ser = df.iloc[:, -1]

            X = X_df.to_numpy(dtype=float)
            y = y_ser.to_numpy(dtype=float).ravel()

            # keep feature names if header exists
            var_names = [str(c) for c in X_df.columns.tolist()]
            return X, y, var_names

        except ImportError:
            # fallback without pandas; requires header
            arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
            names = arr.dtype.names
            if not names or len(names) < 2:
                raise RuntimeError("CSV load failed: need header and >=2 columns")

            xcols = list(names[:-1])
            ycol = names[-1]
            X = np.column_stack([arr[c] for c in xcols]).astype(float)
            y = np.asarray(arr[ycol], dtype=float).ravel()
            return X, y, xcols

    raise ValueError(f"Unsupported dataset extension: {ext}. Use .csv or .npz")

# ============================================================
# Entrypoint
# ============================================================
def main() -> None:
    X, y, var_names = load_dataset(DATA_PATH)

    print("============================================================")
    print("[Dataset loaded]")
    print(f"path: {DATA_PATH}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"variables: {var_names}")
    print("============================================================\n")

    hint = analyze_io_dataset(X, y, var_names=var_names, top_k=TOP_K_TERMS)

    # Print dict nicely
    obj = asdict(hint)
    print("============================================================")
    print("[Analysis Dictionary]")
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    print("============================================================\n")

    print("============================================================")
    print("[Prompt Hint]")
    print(hint.prompt_hint)
    print("============================================================\n")


if __name__ == "__main__":
    main()
