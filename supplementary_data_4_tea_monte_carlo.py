# -*- coding: utf-8 -*-
"""
Process cost Monte Carlo (N=10000), linear $ axis, with:
- A) Normal-fit overlay scaled to COUNTS per bin (linear, X covers ALL samples)
- B) Overlapped histograms (COUNTS) + normal fits scaled to counts (linear, full X)
- C) Standardized density (z-score) — KDE curves only (density domain)
- D) CDF overlay (linear, full X)

Assumes per-line-item uncertainty via CVs. Line items are sampled lognormally
(positive support; total cost tends to near-normal by CLT).
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# ---- Config ----
# =========================
N_ITER = 10000            # Monte Carlo iterations
SEED = 42
FIG_DPI = 360
FONT_FAMILY = "Arial"     # Will fall back if missing
TITLE_SIZE = 18
LABEL_SIZE = 20
TICK_SIZE  = 18
LEGEND_SIZE = 20
NBINS = 60                # Histogram bins (linear axis)

# Nature-like palette
NATURE_PALETTE = [
    "#4C78A8",  # blue
    "#54A24B",  # green
    "#F58518",  # orange
    "#B279A2",  # purple (spare)
    "#72B7B2",  # teal (spare)
]

# Nominal line-item costs (USD)
base_costs = {
    "Solvent-based": {
        "butyl acetate": 26.145,
        "tartatic acid": 5.47,  # keep name as provided
        "diethyl ether (99.95%)": 163.76,
        "electricity, high voltage": 4.08,
        "water, deionised": 71.50,
        "municipal solid waste": 0.796,
        "Chemical waste, regulated": 121.67,
    },
    "Pyrolysis": {
        "electricity, high voltage": 43.56,
        "hydrochloric acid (30%)": 0.26,
        "nitrogen, liquid": 15.05,
        "water, deionised": 0.92,
    },
    "IC-FJH": {
        "electricity, high voltage": 14.31,
        "nitrogen, liquid": 3.8,
        "sodium hydroxide (50%)": 1.76,
        "water, deionised": 2.11,
        "municipal solid waste": 0.212,
        "Chemical waste, regulated": 0.0042,
    },
}

# Per-item coefficients of variation (CV)
cv_by_item = {
    # solvents/organics
    "diethyl ether (99.95%)": 0.10,
    "butyl acetate": 0.10,
    "tartatic acid": 0.10,
    # utilities
    "electricity, high voltage": 0.10,
    "water, deionised": 0.10,
    "nitrogen, liquid": 0.15,
    # acids/bases (commodity)
    "hydrochloric acid (30%)": 0.15,
    "sodium hydroxide (50%)": 0.15,
    # wastes
    "municipal solid waste": 0.25,
    "Chemical waste, regulated": 0.25,
}
DEFAULT_CV = 0.20

# Line-item distribution: lognormal (positive), or truncated normal if disabled
USE_LOGNORMAL_LINEITEM = True

# =========================
# ---- Helpers ----
# =========================
rng = np.random.default_rng(SEED)

def set_style():
    mpl.rcParams["figure.dpi"] = FIG_DPI
    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["axes.titlesize"] = TITLE_SIZE
    mpl.rcParams["axes.labelsize"] = LABEL_SIZE
    mpl.rcParams["xtick.labelsize"] = TICK_SIZE
    mpl.rcParams["ytick.labelsize"] = TICK_SIZE
    mpl.rcParams["legend.fontsize"] = LEGEND_SIZE
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.grid"] = False

def draw_lognormal(mean_val, cv, size):
    """Sample from a lognormal with given mean and CV."""
    sigma2 = np.log(1.0 + cv**2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean_val) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=sigma, size=size)

def draw_truncnorm_pos(mean_val, cv, size, max_tries=1000):
    """Positive truncated normal by rejection sampling."""
    std = cv * mean_val
    out = np.empty(size)
    i = 0
    while i < size and max_tries > 0:
        need = size - i
        cand = rng.normal(loc=mean_val, scale=std, size=need)
        cand = cand[cand > 0]
        take = min(len(cand), need)
        if take > 0:
            out[i:i+take] = cand[:take]
            i += take
        max_tries -= 1
    if i < size:  # extreme fallback
        out[i:] = np.finfo(float).eps
    return out

def sample_line_item(method, item, mean_val, N):
    """Sample one line item using per-item CV (fallback to DEFAULT_CV)."""
    cv = cv_by_item.get(item, DEFAULT_CV)
    if USE_LOGNORMAL_LINEITEM:
        return draw_lognormal(mean_val, cv, N)
    else:
        return draw_truncnorm_pos(mean_val, cv, N)

def simulate_totals(base_costs, N):
    totals = {}
    for method, items in base_costs.items():
        total = np.zeros(N)
        for name, mean_val in items.items():
            total += sample_line_item(method, name, mean_val, N)
        totals[method] = total
    return totals

def normal_pdf(x, mu, sigma):
    sigma = max(float(sigma), np.finfo(float).tiny)
    return (1.0 / (np.sqrt(2.0*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def normal_cdf(x, mu, sigma):
    sigma = max(float(sigma), np.finfo(float).tiny)
    z = (x - mu) / (sigma * np.sqrt(2.0))
    # erf-based CDF to avoid SciPy dependency
    return 0.5 * (1.0 + (2/np.sqrt(np.pi)) * np.vectorize(lambda t: np.math.erf(t))(z))

def summarize_series(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

# Simple Gaussian KDE (no SciPy), Silverman's rule for bandwidth
def kde_gaussian(grid, samples):
    samples = np.asarray(samples)
    n = len(samples)
    if n <= 1:
        return np.zeros_like(grid)
    std = np.std(samples, ddof=1)
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = q75 - q25
    sigma = std if iqr <= 0 else min(std, iqr / 1.34)
    if sigma <= 0:
        sigma = 1.0
    h = 0.9 * sigma * n ** (-1/5)   # Silverman's rule
    if h <= 0:
        h = 1.06 * (std if std > 0 else 1.0) * n ** (-1/5)
    u = (grid[:, None] - samples[None, :]) / h
    dens = np.exp(-0.5 * u**2).sum(axis=1) / (n * h * np.sqrt(2*np.pi))
    return dens

# =========================
# ---- Simulate ----
# =========================
set_style()
totals = simulate_totals(base_costs, N_ITER)
df_samples = pd.DataFrame(totals)
df_samples.index.name = "iteration"

summary_rows = [pd.Series(summarize_series(df_samples[c]), name=c) for c in df_samples.columns]
summary = pd.DataFrame(summary_rows)
df_samples.to_csv("cost_sim_samples.csv")
summary.to_csv("cost_summary.csv")

# Global X range (cover all samples)
GLOBAL_MIN = float(np.min(df_samples.values))
GLOBAL_MAX = float(np.max(df_samples.values))
pad = 0.02 * (GLOBAL_MAX - GLOBAL_MIN) if GLOBAL_MAX > GLOBAL_MIN else 1.0
X_MIN = max(0.0, GLOBAL_MIN - pad)
X_MAX = GLOBAL_MAX + pad

# Shared histogram edges and bin width
BIN_EDGES = np.linspace(X_MIN, X_MAX, NBINS + 1)
BIN_WIDTH = BIN_EDGES[1] - BIN_EDGES[0]

# =========================
# ---- Fig A: Normal-fit overlay (scaled to COUNTS per bin) ----
# =========================
x = np.linspace(X_MIN, X_MAX, 1200)

plt.close("all")
figA, axA = plt.subplots(figsize=(9, 5.2))
for i, m in enumerate(df_samples.columns):
    mu = summary.loc[m, "mean"]
    sd = summary.loc[m, "std"]
    # Convert pdf to expected counts per bin: N * pdf * bin_width
    y_counts = N_ITER * normal_pdf(x, mu, sd) * BIN_WIDTH
    axA.plot(x, y_counts, label=f"{m} (μ={mu:,.0f}, σ={sd:,.0f})",
             lw=2.2, color=NATURE_PALETTE[i % len(NATURE_PALETTE)])
axA.set_title(f"Total Cost — Normal Fits Scaled to Counts per Bin (N={N_ITER})")
axA.set_xlabel("Total cost (USD)")
axA.set_ylabel("Frequency (counts per bin)")
axA.set_xlim(X_MIN, X_MAX)
axA.legend(frameon=False)
figA.tight_layout()
figA.savefig("A_cost_normalfit_counts.svg")
figA.savefig("A_cost_normalfit_counts.png")

# =========================
# ---- Fig B: Overlapped HIST (Counts) + Normal Fits (Counts) ----
# =========================
figB, axB = plt.subplots(figsize=(9, 5.2))
for i, m in enumerate(df_samples.columns):
    color = NATURE_PALETTE[i % len(NATURE_PALETTE)]
    data = df_samples[m].values
    # Histogram: counts
    hist, _ = np.histogram(data, bins=BIN_EDGES, density=False)
    centers = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
    axB.fill_between(centers, hist, step="mid", alpha=0.28, color=color, label=m)
    # Normal fit curve: counts per bin
    mu = summary.loc[m, "mean"]
    sd = summary.loc[m, "std"]
    xh = np.linspace(X_MIN, X_MAX, 1200)
    fit_counts = N_ITER * normal_pdf(xh, mu, sd) * BIN_WIDTH
    axB.plot(xh, fit_counts, lw=2.2, color=color)

axB.set_title(f"Overlapped Histograms (Counts) + Normal Fits (Counts) (N={N_ITER})")
axB.set_xlabel("Total cost (USD)")
axB.set_ylabel("Frequency (counts per bin)")
axB.set_xlim(X_MIN, X_MAX)
axB.legend(frameon=False)
figB.tight_layout()
figB.savefig("B_cost_hist_overlapped_counts.svg")
figB.savefig("B_cost_hist_overlapped_counts.png")

# =========================
# ---- Fig C: Standardized density (z-score) — KDE ONLY (density) ----
# =========================
z_min, z_max = -4.5, 4.5
z_grid = np.linspace(z_min, z_max, 1200)

figC, axC = plt.subplots(figsize=(9, 5.2))
for i, m in enumerate(df_samples.columns):
    color = NATURE_PALETTE[i % len(NATURE_PALETTE)]
    mu = summary.loc[m, "mean"]
    sd = summary.loc[m, "std"]
    sd_safe = sd if sd > 0 else 1.0
    z_samples = (df_samples[m].values - mu) / sd_safe
    dens = kde_gaussian(z_grid, z_samples)
    axC.plot(z_grid, dens, lw=2.2, color=color, label=m)

axC.set_title("Standardized Density (z-score) — Empirical KDE (curves only)")
axC.set_xlabel("z-score  ((Total cost − μ) / σ)")
axC.set_ylabel("Density")
axC.legend(frameon=False)
figC.tight_layout()
figC.savefig("C_cost_kde_zscore.svg")
figC.savefig("C_cost_kde_zscore.png")

# =========================
# ---- Fig D: CDF overlay (linear, full X) ----
# =========================
figD, axD = plt.subplots(figsize=(9, 5.2))
for i, m in enumerate(df_samples.columns):
    color = NATURE_PALETTE[i % len(NATURE_PALETTE)]
    data = np.sort(df_samples[m].values)
    n = len(data)
    y_ecdf = np.arange(1, n + 1) / n
    axD.step(data, y_ecdf, where="post", color=color, lw=2.0, alpha=0.9, label=m)

    # Overlay normal CDF with sample mean/std
    mu = summary.loc[m, "mean"]
    sd = summary.loc[m, "std"]
    x_cdf = np.linspace(X_MIN, X_MAX, 800)
    y_cdf = normal_cdf(x_cdf, mu, sd)
    axD.plot(x_cdf, y_cdf, color=color, lw=1.6, ls="--", alpha=0.9)

axD.set_title(f"CDF Overlay (empirical steps + normal CDF; N={N_ITER})")
axD.set_xlabel("Total cost (USD)")
axD.set_ylabel("Cumulative probability")
axD.set_xlim(X_MIN, X_MAX)
axD.set_ylim(0, 1)

# Put legend OUTSIDE on the right to avoid any overlap with lines
# Reserve room on the right, then anchor the legend there.
figD.subplots_adjust(right=0.78)
leg = axD.legend(
    loc="center left", bbox_to_anchor=(1.002, 0.5), frameon=False,
    borderaxespad=0.0, handlelength=2.2, handletextpad=0.8, labelspacing=1.0
)

figD.tight_layout()
figD.savefig("D_cost_cdf_overlay.svg")
figD.savefig("D_cost_cdf_overlay.png")

print("\nSaved files:")
print("  - cost_sim_samples.csv")
print("  - cost_summary.csv")
print("  - A_cost_normalfit_counts.(svg|png)")
print("  - B_cost_hist_overlapped_counts.(svg|png)")
print("  - C_cost_kde_zscore.(svg|png)")
print("  - D_cost_cdf_overlay.(svg|png)")

