# -*- coding: utf-8 -*-
"""
Supporting Data Script — Sb recovery & purity mapping with small-data augmentation
- Gaussian Process Regression (GPR) for Sb recovery and Sb purity
- Optional PCHIP row-wise augmentation along log(heating rate)
- Leave-One-Out Cross-Validation (LOOCV) evaluation
- Heatmaps (x = heating rate, y = temperature)
- Feasible region: recovery >= 89 AND purity >= 97
Author: (Your Name)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C, Matern, RationalQuadratic, WhiteKernel
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.interpolate import PchipInterpolator

# --------------------------
# Config
# --------------------------
SEED = 0
AUGMENT = True          # enable in-row PCHIP augmentation (no extrapolation)
POINTS_PER_ROW = 4      # synthetic points added per row (excluding originals)
N_RATE = 240            # x-grid samples for plotting
N_TEMP = 240            # y-grid samples for plotting
THR_REC = 89.0
THR_PUR = 97.0

# Show original sample points on heatmaps? (set False to omit white circles)
DRAW_RAW_POINTS = False

# --------------------------
# 1) Original data
# --------------------------
temps_rows = np.array([1500, 1200, 900, 600], dtype=float)  # rows = temperature
rates_rows = np.array([
    [ 545.52, 16250, 59091],   # T=1500
    [ 277.78,  6000, 33333],   # T=1200
    [ 218.59,  5040, 20628],   # T= 900
    [  78.22,  2912, 13367],   # T= 600
], dtype=float)

rec_rows = np.array([
    [78.37723314, 90.14484226, 90.79704621],
    [75.78649810, 89.29170674, 90.11856070],
    [69.45524512, 84.81446985, 89.11453254],
    [65.83353832, 82.11248918, 86.81711698],
], dtype=float)

pur_rows = np.array([
    [97.64952215, 69.00631717, 88.86128414],
    [98.24367361, 90.04083043, 94.95283501],
    [97.83421398, 97.99548579, 97.77089252],
    [98.38635465, 98.11068096, 98.13947694],
], dtype=float)

# --------------------------
# 2) Assemble sample points
# --------------------------
def to_samples(temps_rows, rates_rows, target_rows):
    X_list, y_list = [], []
    for i, T in enumerate(temps_rows):
        for j in range(rates_rows.shape[1]):
            X_list.append([np.log10(rates_rows[i, j]), T])
            y_list.append(target_rows[i, j])
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float)

X_rec_raw, y_rec_raw = to_samples(temps_rows, rates_rows, rec_rows)
X_pur_raw, y_pur_raw = to_samples(temps_rows, rates_rows, pur_rows)

# --------------------------
# 3) Row-wise PCHIP augmentation (no extrapolation)
# --------------------------
def augment_by_row_pchip(temps_rows, rates_rows, target_rows, points_per_row=4):
    Xa, ya = [], []
    for i, T in enumerate(temps_rows):
        r = rates_rows[i, :]
        y = target_rows[i, :]
        xr = np.log10(r)
        order = np.argsort(xr)
        xr = xr[order]; y = y[order]
        f = PchipInterpolator(xr, y, extrapolate=False)
        xr_new = np.linspace(xr.min(), xr.max(), points_per_row + 2)[1:-1]
        y_new = f(xr_new)
        mask = np.isfinite(y_new)
        xr_new = xr_new[mask]; y_new = y_new[mask]
        for xv, yv in zip(xr_new, y_new):
            Xa.append([xv, T]); ya.append(yv)
    return np.array(Xa, dtype=float), np.array(ya, dtype=float)

if AUGMENT:
    X_rec_aug, y_rec_aug = augment_by_row_pchip(temps_rows, rates_rows, rec_rows, POINTS_PER_ROW)
    X_pur_aug, y_pur_aug = augment_by_row_pchip(temps_rows, rates_rows, pur_rows, POINTS_PER_ROW)
    X_rec = np.vstack([X_rec_raw, X_rec_aug]); y_rec = np.hstack([y_rec_raw, y_rec_aug])
    X_pur = np.vstack([X_pur_raw, X_pur_aug]); y_pur = np.hstack([y_pur_raw, y_pur_aug])
else:
    X_rec, y_rec = X_rec_raw.copy(), y_rec_raw.copy()
    X_pur, y_pur = X_pur_raw.copy(), y_pur_raw.copy()

print(f"[Data] recovery samples: {len(y_rec)}, purity samples: {len(y_pur)} (raw=12, augmented={'on' if AUGMENT else 'off'})")

# --------------------------
# 4) GPR model (robust kernel + light noise)
# --------------------------
kernel_stable = C(1.0, (1e-3, 1e3)) * (
    Matern(length_scale=[0.5, 0.5], nu=1.5, length_scale_bounds=(0.1, 5.0)) +
    RationalQuadratic(alpha=1.0, length_scale=0.5,
                      alpha_bounds=(1e-3, 1e3), length_scale_bounds=(0.1, 5.0))
) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

def make_gpr():
    return GaussianProcessRegressor(
        kernel=kernel_stable,
        normalize_y=True,
        alpha=1e-6,
        n_restarts_optimizer=12,
        random_state=SEED
    )

xsc_rec = MinMaxScaler()
xsc_pur = MinMaxScaler()

# --------------------------
# 5) LOOCV evaluation (raw 12 points only)
# --------------------------
def loocv_minmax(X_raw, y_raw):
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(X_raw):
        xsc = MinMaxScaler()
        Xtr = xsc.fit_transform(X_raw[tr])
        Xte = xsc.transform(X_raw[te])
        model = make_gpr()
        model.fit(Xtr, y_raw[tr])
        preds.append(model.predict(Xte)[0]); trues.append(y_raw[te][0])
    return mean_absolute_error(trues, preds), r2_score(trues, preds)

mae_rec, r2_rec = loocv_minmax(X_rec_raw, y_rec_raw)
mae_pur, r2_pur = loocv_minmax(X_pur_raw, y_pur_raw)
print(f"[LOOCV] Sb-recovery: MAE={mae_rec:.2f}, R2={r2_rec:.3f}")
print(f"[LOOCV] Sb-purity : MAE={mae_pur:.2f}, R2={r2_pur:.3f}")

# --------------------------
# 6) Fit final models
# --------------------------
Xs_rec = xsc_rec.fit_transform(X_rec)
gpr_rec = make_gpr(); gpr_rec.fit(Xs_rec, y_rec)
print("[Kernel-Rec]", gpr_rec.kernel_)

Xs_pur = xsc_pur.fit_transform(X_pur)
gpr_pur = make_gpr(); gpr_pur.fit(Xs_pur, y_pur)
print("[Kernel-Pur]", gpr_pur.kernel_)

# --------------------------
# 7) Prediction grid (no extrapolation beyond min/max)
# --------------------------
rate_min = np.min(rates_rows)
rate_max = np.max(rates_rows)
temp_min = np.min(temps_rows)
temp_max = np.max(temps_rows)

rate_dense = np.logspace(np.log10(rate_min), np.log10(rate_max), N_RATE)
temp_dense = np.linspace(temp_min, temp_max, N_TEMP)
RR, TT = np.meshgrid(rate_dense, temp_dense)
XX = np.column_stack([np.log10(RR).ravel(), TT.ravel()])

Rec_hat = gpr_rec.predict(xsc_rec.transform(XX)).reshape(TT.shape)
Pur_hat = gpr_pur.predict(xsc_pur.transform(XX)).reshape(TT.shape)

# --------------------------
# 8) Heatmaps (raw points optional)
# --------------------------
extent = [np.log10(rate_dense.min()), np.log10(rate_dense.max()),
          temp_dense.min(), temp_dense.max()]

def set_axes(ax):
    ax.set_xlabel("Heating rate (K/s)")
    ax.set_ylabel("Temperature (°C)")

    # Show only 10^2, 10^3, 10^4 ticks when within [rate_min, rate_max]
    desired = np.array([1e2, 1e3, 1e4], dtype=float)
    in_range = desired[(desired >= rate_min) & (desired <= rate_max)]
    if in_range.size == 0:
        # Fallback: geometric spacing if none are in range
        xticks_vals = np.geomspace(rate_min, rate_max, 4)
        ax.set_xticks(np.log10(xticks_vals))
        ax.set_xticklabels([f"{v:.0f}" for v in xticks_vals])
    else:
        ax.set_xticks(np.log10(in_range))
        ax.set_xticklabels([rf"$10^{int(np.log10(v))}$" for v in in_range])

def overlay_raw_points(ax):
    # Kept for potential use; not called by default
    for i, T in enumerate(temps_rows):
        for j in range(rates_rows.shape[1]):
            ax.scatter(np.log10(rates_rows[i, j]), T, s=26,
                       edgecolor='k', facecolor='w', linewidth=0.7)

# Recovery map (white contour, no label)
plt.figure(figsize=(7.0, 5.2))
im1 = plt.imshow(Rec_hat, origin='lower', aspect='auto', extent=extent,
                 vmin=min(y_rec_raw), vmax=max(y_rec_raw))
set_axes(plt.gca()); plt.title("Sb recovery (%) — GP")
if DRAW_RAW_POINTS:
    overlay_raw_points(plt.gca())
plt.contour(np.log10(rate_dense), temp_dense, Rec_hat,
            levels=[THR_REC], colors='white', linewidths=1.8)
plt.colorbar(im1); plt.tight_layout()

# Purity map (white contour, no label)
plt.figure(figsize=(7.0, 5.2))
im2 = plt.imshow(Pur_hat, origin='lower', aspect='auto', extent=extent,
                 vmin=min(y_pur_raw), vmax=max(y_pur_raw))
set_axes(plt.gca()); plt.title("Sb purity (%) — GP")
if DRAW_RAW_POINTS:
    overlay_raw_points(plt.gca())
plt.contour(np.log10(rate_dense), temp_dense, Pur_hat,
            levels=[THR_PUR], colors='white', linewidths=1.8)
plt.colorbar(im2); plt.tight_layout()

# --------------------------
# 9) Feasible region
# --------------------------
mask = (Rec_hat >= THR_REC) & (Pur_hat >= THR_PUR)

plt.figure(figsize=(7.0, 5.2))
im3 = plt.imshow(mask.astype(float), origin='lower', aspect='auto',
                 extent=extent, vmin=0, vmax=1)
set_axes(plt.gca()); plt.title(f"Feasible region (Rec≥{THR_REC:.0f}, Pur≥{THR_PUR:.0f})")
cs1 = plt.contour(np.log10(rate_dense), temp_dense, Rec_hat, levels=[THR_REC], linewidths=1.2)
cs2 = plt.contour(np.log10(rate_dense), temp_dense, Pur_hat, levels=[THR_PUR], linewidths=1.2)
plt.clabel(cs1, fmt={THR_REC: "Rec=89"}, inline=True, fontsize=9)
plt.clabel(cs2, fmt={THR_PUR: "Pur=97"}, inline=True, fontsize=9)
if DRAW_RAW_POINTS:
    overlay_raw_points(plt.gca())
plt.colorbar(im3, ticks=[0, 1]); plt.tight_layout(); plt.show()

# --------------------------
# 10) Range output
# --------------------------
if np.any(mask):
    feas_rates = RR[mask]; feas_temps = TT[mask]
    print("\n[Feasible region]")
    print("  Rate (K/s): [%.0f, %.0f]" % (feas_rates.min(), feas_rates.max()))
    print("  Temp (°C):  [%.0f, %.0f]" % (feas_temps.min(), feas_temps.max()))
    best_idx = np.nanargmax(np.where(mask, Rec_hat, np.nan))
    ti, ri = np.unravel_index(best_idx, Rec_hat.shape)
    print("  Example best (within feasible): rate=%.0f K/s, temp=%.0f °C | Rec=%.2f, Pur=%.2f"
          % (RR[ti, ri], TT[ti, ri], Rec_hat[ti, ri], Pur_hat[ti, ri]))
else:
    print("\n[No feasible region found under current thresholds and model]. "
          "Try adjusting thresholds, enabling augmentation, or adding experiments near boundaries.")


