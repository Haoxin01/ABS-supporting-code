## -*- coding: utf-8 -*-
"""
Binary feasibility map on (Voltage, Time) for C = 132 mF (fixed features + monotone T)
- Data: latest table (all 132 mF). Times: 1 s, 0.2 s, 0.05 s (50 ms has three rows: 1200/900/600 °C)
- Models: two GPRs with X = [log10(V), log10(t)]  (drop logE/logP to avoid non-physical bias)
- Physics prior: for each fixed V, enforce T(V, ·) monotone non-decreasing vs time via isotonic regression
- Heatmap (x = Voltage, y = Time) shows 1 if Temp ∈ [TEMP_MIN, TEMP_MAX] and Rate ∈ [RATE_MIN, RATE_MAX], else 0
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as Ck, Matern, RationalQuadratic, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.isotonic import IsotonicRegression

# --------------------------
# Feasible (Temp, Rate) region from model-1
# --------------------------
TEMP_MIN, TEMP_MAX = 600.0, 1206.0
RATE_MIN, RATE_MAX = 12858.0, 59091.0

# --------------------------
# Latest data (all C = 132 mF)
# V,t columns: 1 s, 0.2 s, 0.05 s (three rows)
# --------------------------
# 1 s
V_1s = np.array([190, 180, 150, 100], dtype=float)
t_1s = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
T_1s = np.array([1500, 1200,  900,  600], dtype=float)
R_1s = np.array([16250,  6001, 5040, 1912], dtype=float)

# 0.2 s
V_02 = np.array([220, 190, 160, 115], dtype=float)
t_02 = np.array([0.2, 0.2, 0.2, 0.2], dtype=float)
T_02 = np.array([1500, 1200,  900,  600], dtype=float)
R_02 = np.array([14375,  8182, 4800, 3739], dtype=float)

# 0.05 s
V_005 = np.array([285, 230, 190, 170], dtype=float)
t_005 = np.array([0.05, 0.05, 0.05, 0.05], dtype=float)
T_005 = np.array([1500, 1200,  900,  600], dtype=float)
R_005 = np.array([59091, 33332, 20628,  6367], dtype=float)

# Stack to arrays (11 samples)
V_arr = np.concatenate([V_1s,  V_02,  V_005])
t_arr = np.concatenate([t_1s,  t_02,  t_005])
T_arr = np.concatenate([T_1s,  T_02,  T_005])
R_arr = np.concatenate([R_1s,  R_02,  R_005])

# --------------------------
# Features: ONLY [log10(V), log10(t)]
# --------------------------
def make_features(V, t):
    return np.column_stack([np.log10(np.clip(V, 1e-9, None)),
                            np.log10(np.clip(t, 1e-9, None))])

X_raw = make_features(V_arr, t_arr)

# --------------------------
# GPR models (shared kernel)
# --------------------------
kernel = Ck(1.0, (1e-3, 1e3)) * (
    Matern(length_scale=[0.6, 0.6], nu=1.5, length_scale_bounds=(0.05, 5.0))
  + RationalQuadratic(alpha=1.0, length_scale=0.6,
                      alpha_bounds=(1e-3, 1e3), length_scale_bounds=(0.05, 5.0))
) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

def make_gpr():
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-6,
        n_restarts_optimizer=12, random_state=0
    )

# LOOCV (using original 11 points)
def loocv(X, y):
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(X):
        sc = MinMaxScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        gpr = make_gpr(); gpr.fit(Xtr, y[tr])
        preds.append(gpr.predict(Xte)[0]); trues.append(y[te][0])
    return mean_absolute_error(trues, preds), r2_score(trues, preds)

mae_T, r2_T = loocv(X_raw, T_arr)
mae_R, r2_R = loocv(X_raw, R_arr)
print(f"[LOOCV] Temperature: MAE={mae_T:.1f} °C, R2={r2_T:.3f}")
print(f"[LOOCV] Heat rate  : MAE={mae_R:.0f} K/s, R2={r2_R:.3f}")

## Fit final models (use ALL data)
# --------------------------
scaler = MinMaxScaler()
Xs = scaler.fit_transform(X_raw)

gpr_T = make_gpr()
gpr_R = make_gpr()
gpr_T.fit(Xs, T_arr)
gpr_R.fit(Xs, R_arr)

# --------------------------
# Build (V, t) grid for heatmap
# --------------------------
Nv, Nt = 220, 220  # 网格分辨率：越大越细，但计算越慢
V_grid = np.linspace(V_arr.min(), V_arr.max(), Nv)
t_grid = np.logspace(np.log10(t_arr.min()), np.log10(t_arr.max()), Nt)

VV, TT = np.meshgrid(V_grid, t_grid)  # shape: (Nt, Nv)

# Predict on grid
Xg_raw = make_features(VV.ravel(), TT.ravel())
Xg = scaler.transform(Xg_raw)

T_pred = gpr_T.predict(Xg).reshape(TT.shape)  # (Nt, Nv)
R_pred = gpr_R.predict(Xg).reshape(TT.shape)  # (Nt, Nv)

# --------------------------
# Physics prior: for each fixed V, enforce T(V,·) monotone non-decreasing vs time
# --------------------------
T_mono = np.empty_like(T_pred)
ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
for j in range(Nv):  # each column is fixed V
    T_mono[:, j] = ir.fit_transform(t_grid, T_pred[:, j])

# --------------------------
# Binary feasibility matrix (1 feasible, 0 infeasible)
# --------------------------
feasible = (
    (T_mono >= TEMP_MIN) & (T_mono <= TEMP_MAX) &
    (R_pred >= RATE_MIN) & (R_pred <= RATE_MAX)
).astype(int)

mask = (feasible == 1)

if not mask.any():
    print("No feasible (V,t) points predicted on this grid.")
else:
    # 1) 可行点的“整体包络范围”(bounding box)
    rows, cols = np.where(mask)
    V_feas_min, V_feas_max = V_grid[cols].min(), V_grid[cols].max()
    t_feas_min, t_feas_max = t_grid[rows].min(), t_grid[rows].max()

    print(f"[Predicted feasible region bounding box]")
    print(f"  V: {V_feas_min:.2f} – {V_feas_max:.2f} V")
    print(f"  t: {t_feas_min:.4f} – {t_feas_max:.4f} s")

    # 2) （可选）分别输出 V 和 t 的“可行区间段”（可能有多个不连续段）
    def contiguous_intervals(flag, axis_values):
        idx = np.where(flag)[0]
        if idx.size == 0:
            return []
        cuts = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[cuts + 1]]
        ends   = np.r_[idx[cuts], idx[-1]]
        return [(axis_values[s], axis_values[e]) for s, e in zip(starts, ends)]

    V_any = mask.any(axis=0)   # 某个V列是否存在任意t可行
    t_any = mask.any(axis=1)   # 某个t行是否存在任意V可行

    V_intervals = contiguous_intervals(V_any, V_grid)
    t_intervals = contiguous_intervals(t_any, t_grid)

    print("[Feasible V intervals]")
    for a, b in V_intervals:
        print(f"  {a:.2f} – {b:.2f} V")

    print("[Feasible t intervals]")
    for a, b in t_intervals:
        print(f"  {a:.4f} – {b:.4f} s")

print("V_grid shape:", V_grid.shape)
print("t_grid shape:", t_grid.shape)
print("feasible matrix shape (Nt, Nv):", feasible.shape)
print("Feasible fraction:", feasible.mean())

# --------------------------
# Plot heatmap (binary)
# --------------------------
plt.figure(figsize=(8.5, 5.5))
m = plt.pcolormesh(V_grid, t_grid, feasible, shading="auto")
plt.xlabel("Voltage (V)")
plt.ylabel("Time (s)")
plt.title("Feasibility map (1 = feasible)")
plt.colorbar(m, label="Feasible (0/1)")
plt.tight_layout()
plt.show()

# --------------------------
# Save outputs (optional)
# --------------------------
np.save("feasible_matrix.npy", feasible)
np.savetxt("V_grid.csv", V_grid, delimiter=",")
np.savetxt("t_grid.csv", t_grid, delimiter=",")
np.savetxt("feasible_matrix.csv", feasible, fmt="%d", delimiter=",")

