#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-D transient heat conduction (explicit FTCS) in Ar with carbon paper as Dirichlet T_hot(t).
- Reads Excel with two columns: Time (e.g., '000:00:16.606'), Temperature (°C).
- Scenario A: SINGLE layer carbon paper (0.37 mm thick), domain length 5 cm.
  Outputs temperature vs time at y = (carbon top surface + 0.5/1/2 mm),
  and a front-view heat map when the carbon temperature first reaches 911 °C.
- Scenario B: DOUBLE layers (two carbon papers separated by 1.0 mm gap).
  Outputs temperature vs time at the mid-gap (0.5 mm from each layer),
  and heat map at the same 911 °C trigger.
- EXTRA: For DOUBLE layer, also save heat maps at t = 0, 4, 8, 12, 16, 20, 24 ms.

All plots use Arial and are saved as PNG; probe time series saved as CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------- USER SETTINGS -----------------------
excel_path = Path("carbon_temp_curve_ABS.xlsx")   # .xlsx or .xls

L = 0.05           # domain size in x & y (m) — 5 cm square
N = 301            # grid points per side
cp_th = 0.000370   # carbon paper thickness (m) = 0.37 mm
cp_w  = 0.015      # carbon paper width along x (m) = 1.5 cm
gap_double = 0.001 # double-layer gap (m) = 1.0 mm

# Argon
k_argon   = 0.0177   # W/m/K
rho_argon = 1.6      # kg/m3
cp_argon  = 520.0    # J/kg/K

# Carbon paper
k_carbon   = 0.25
rho_carbon = 1700.0
cp_carbon  = 710.0

# Robin boundary
T_inf = 25.0
h_eff = 10.0

# Numerics
Fo = 0.4
font_family = "Arial"

# Probes for SINGLE layer (distances from carbon TOP surface, meters)
probe_offsets_m = [0.0005, 0.0010, 0.0020]   # 0.5/1.0/2.0 mm
T_trigger = 911.0    # °C — trigger temp for snapshot

# EXTRA: specified double-layer snapshot times (s) — added 0.024 s
requested_times_double = [0.0, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024]

# Unified heatmap colorbar range (°C)
VMIN = 0.0
VMAX = 1000.0
# ------------------------------------------------------------

# Global font scaling (other styles unchanged)
plt.rcParams["font.family"] = font_family
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["figure.titlesize"] = 18

def parse_time_to_seconds(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.floating)):
        return float(s)
    s = str(s).strip()
    parts = s.split(":")
    if len(parts) == 3:
        try:
            H = int(parts[0]); M = int(parts[1]); S = float(parts[2])
            return H*3600.0 + M*60.0 + S
        except Exception:
            pass
    try:
        return pd.to_timedelta(s).total_seconds()
    except Exception as e:
        raise ValueError(f"Unrecognized time format: {s}") from e

def read_curve(path: Path):
    ext = path.suffix.lower()
    engine = None
    if ext == ".xlsx":
        engine = "openpyxl"
    elif ext == ".xls":
        try:
            import xlrd  # noqa
            engine = "xlrd"
        except Exception as e:
            raise ImportError('Reading ".xls" needs xlrd<2.0: pip install "xlrd<2.0"') from e
    if not path.exists():
        raise FileNotFoundError(f"Excel not found: {path}")
    df = pd.read_excel(path, engine=engine)
    if df.shape[1] < 2:
        raise ValueError("Excel needs at least 2 columns: Time, Temperature(°C).")
    t_sec = df.iloc[:,0].apply(parse_time_to_seconds).astype(float).to_numpy()
    T_deg = pd.to_numeric(df.iloc[:,1], errors="coerce").to_numpy()
    mask = np.isfinite(t_sec) & np.isfinite(T_deg)
    t_sec, T_deg = t_sec[mask], T_deg[mask]
    order = np.argsort(t_sec)
    return t_sec[order], T_deg[order]

def find_first_crossing_time(t_curve, T_curve, T_thr):
    above = T_curve >= T_thr
    if not np.any(above):
        return None
    i = np.argmax(above)
    if i == 0:
        return t_curve[0]
    t0, t1 = t_curve[i-1], t_curve[i]
    T0, T1 = T_curve[i-1], T_curve[i]
    if T1 == T0:
        return t1
    frac = (T_thr - T0) / (T1 - T0)
    return t0 + frac * (t1 - t0)

# ---------- Grid & materials ----------
dx = L / (N - 1)
x = np.linspace(0.0, L, N)
y = np.linspace(0.0, L, N)

# single-layer indices (centered)
ix0 = int((L/2 - cp_w/2) / dx)
ix1 = int((L/2 + cp_w/2) / dx)
iy_mid = N // 2
iy0 = int(iy_mid - (cp_th/2) / dx)
iy1 = int(iy_mid + (cp_th/2) / dx)

# double-layer indices
gap_cells = max(1, int(round(gap_double / dx)))
iy0_b = iy_mid - gap_cells//2 - int(round(cp_th/dx)) - 1
iy1_b = iy0_b + int(round(cp_th/dx))
iy0_t = iy1_b + gap_cells + 1
iy1_t = iy0_t + int(round(cp_th/dx))

def apply_carbon(k, rho, cp, iy0, iy1, ix0, ix1):
    k[iy0:iy1+1, ix0:ix1+1]   = k_carbon
    rho[iy0:iy1+1, ix0:ix1+1] = rho_carbon
    cp[iy0:iy1+1, ix0:ix1+1]  = cp_carbon

# ---------- Read curve ----------
t_curve, T_curve = read_curve(excel_path)
t_end = t_curve[-1]
t_911 = find_first_crossing_time(t_curve, T_curve, T_trigger)

# ---------- Time step ----------
alpha_argon = k_argon/(rho_argon*cp_argon)
alpha_carbon = k_carbon/(rho_carbon*cp_carbon)
alpha_max = max(alpha_argon, alpha_carbon)
dt = Fo * dx * dx / (4.0 * alpha_max)
n_steps = int(np.ceil(t_end / dt))

print(f"[Grid] N={N}  dx={dx*1e3:.3f} mm")
print(f"[Time] dt={dt*1e3:.3f} ms, steps≈{n_steps:,} up to t_end={t_end:.3f} s")
print(f"[Trigger] T_hot first reaches {T_trigger} °C at t≈{t_911:.6f} s" if t_911 is not None else
      "[Trigger] T_hot never reaches 911 °C within provided curve.")

def run_case(single_or_double="single", requested_times=None):
    # fields
    kM = np.full((N, N), k_argon)
    rM = np.full((N, N), rho_argon)
    cM = np.full((N, N), cp_argon)

    if single_or_double == "single":
        apply_carbon(kM, rM, cM, iy0, iy1, ix0, ix1)
        cp_masks = [np.zeros((N, N), dtype=bool)]
        cp_masks[0][iy0:iy1+1, ix0:ix1+1] = True
        iy_top = iy1
        # labels unified to mm: 0.5mm/1.0mm/2.0mm
        probe_js = [iy_top + int(round(d/dx)) for d in probe_offsets_m]
        probe_labels = [f"{d*1e3:.1f}mm" for d in probe_offsets_m]
    else:
        # double: bottom + top
        apply_carbon(kM, rM, cM, iy0_b, iy1_b, ix0, ix1)
        apply_carbon(kM, rM, cM, iy0_t, iy1_t, ix0, ix1)
        mask_b = np.zeros((N, N), dtype=bool)
        mask_t = np.zeros((N, N), dtype=bool)
        mask_b[iy0_b:iy1_b+1, ix0:ix1+1] = True
        mask_t[iy0_t:iy1_t+1, ix0:ix1+1] = True
        cp_masks = [mask_b, mask_t]
        # mid-gap probe
        j_mid = (iy1_b + iy0_t) // 2
        probe_js = [j_mid]
        probe_labels = ["mid_gap_0.5mm"]

    aM = kM / (rM * cM)
    T = np.full((N, N), T_inf, dtype=float)
    t_list = []
    probes = [[] for _ in probe_js]
    T_hot_series = []  # record source carbon-paper temperature (interpolated to solver time steps)

    took_911 = False
    T_snapshot = None
    t_snapshot = None

    # Prepare snapshot container for requested times (enabled only if requested_times is provided)
    snapshots_by_time = {}
    req_times_sorted = []
    if requested_times:
        # Filter times within the solution range and sort ascending
        req_times_sorted = sorted([t for t in requested_times if 0.0 <= t <= t_end + 1e-12])
        pending_flags = {t: True for t in req_times_sorted}

    cx = (ix0 + ix1) // 2
    steps = int(np.ceil(t_end / dt))
    for n in range(steps + 1):
        Th = float(np.interp(t_list[-1] if t_list else 0.0, t_curve, T_curve))
        # Dirichlet on carbon
        for m in cp_masks:
            T[m] = Th

        # FTCS
        T_in = T.copy()
        lap = (T_in[2:,1:-1] + T_in[:-2,1:-1] + T_in[1:-1,2:] + T_in[1:-1,:-2]
               - 4.0*T_in[1:-1,1:-1]) / (dx*dx)
        T[1:-1,1:-1] = T_in[1:-1,1:-1] + dt * aM[1:-1,1:-1] * lap

        for m in cp_masks:
            T[m] = Th

        # Robin BC
        T[:,0]  = (kM[:,0]*T[:,1]   + h_eff*dx*T_inf) / (kM[:,0]  + h_eff*dx)
        T[:,-1] = (kM[:,-1]*T[:,-2] + h_eff*dx*T_inf) / (kM[:,-1] + h_eff*dx)
        T[0,:]  = (kM[0,:]*T[1,:]   + h_eff*dx*T_inf) / (kM[0,:]  + h_eff*dx)
        T[-1,:] = (kM[-1,:]*T[-2,:] + h_eff*dx*T_inf) / (kM[-1]   + h_eff*dx)

        # record
        t_cur = 0.0 if not t_list else (t_list[-1] + dt)
        t_list.append(t_cur)
        T_hot_series.append(Th)
        for p, j in enumerate(probe_js):
            jj = np.clip(j, 0, N-1)
            probes[p].append(T[jj, cx])

        if (not took_911) and (t_911 is not None) and (t_cur >= t_911):
            T_snapshot = T.copy()
            t_snapshot = t_cur
            took_911 = True

        # If we reach/exceed a requested time, save a snapshot at that time (double-layer use)
        if requested_times:
            for t_req in req_times_sorted:
                if pending_flags[t_req] and (t_cur >= t_req):
                    snapshots_by_time[t_req] = T.copy()
                    pending_flags[t_req] = False

    # Also return times that could not be captured (beyond t_end) with value None
    if requested_times:
        for t_req in requested_times:
            if t_req not in snapshots_by_time:
                snapshots_by_time[t_req] = None

    return dict(
        T_snapshot=T_snapshot,
        t_arr=np.array(t_list),
        probes=[np.array(v) for v in probes],
        probe_labels=probe_labels,
        t_snapshot=t_snapshot,
        T_hot_series=np.array(T_hot_series),
        snapshots_by_time=snapshots_by_time
    )

# ---------- Run SINGLE ----------
res_single = run_case("single")
# CSV (single layer): time, T_hot, T_at_0.5mm/1.0mm/2.0mm; also output alias T_at_0p5mm
single_df = pd.DataFrame({"time_s": res_single["t_arr"],
                          "T_hot_degC": res_single["T_hot_series"]})
for lab, arr in zip(res_single["probe_labels"], res_single["probes"]):
    single_df[f"T_at_{lab}_degC"] = arr
# Friendly alias for the 0.5 mm location
if "T_at_0.5mm_degC" in single_df.columns:
    single_df["T_at_0p5mm_degC"] = single_df["T_at_0.5mm_degC"]
single_df.to_csv("single_layer_probes.csv", index=False)

# Heat map at 911 °C (single layer) — unified color scale 0–1000 °C
if res_single["T_snapshot"] is not None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(res_single["T_snapshot"], origin="lower", cmap="inferno",
                   extent=[0, L*100, 0, L*100], vmin=VMIN, vmax=VMAX)
    fig.colorbar(im, ax=ax, label="Temperature (°C)")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(f"Single layer — heat map at T_hot≈{T_trigger:.0f} °C (t≈{res_single['t_snapshot']:.3f} s)")
    fig.tight_layout(); fig.savefig("heatmap_single_at_911C.png", dpi=300); plt.close(fig)

# ---------- Run DOUBLE ----------
res_double = run_case("double", requested_times=requested_times_double)

# CSV (double layer): time, T_hot, mid-gap 0.5 mm
double_df = pd.DataFrame({
    "time_s": res_double["t_arr"],
    "T_hot_degC": res_double["T_hot_series"],
    "T_mid_gap_0p5mm_degC": res_double["probes"][0]
})
double_df.to_csv("double_layer_mid_gap_probe.csv", index=False)

# Heat map at 911 °C (double layer) — unified color scale 0–1000 °C
if res_double["T_snapshot"] is not None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(res_double["T_snapshot"], origin="lower", cmap="inferno",
                   extent=[0, L*100, 0, L*100], vmin=VMIN, vmax=VMAX)
    fig.colorbar(im, ax=ax, label="Temperature (°C)")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(f"Double layer — heat map at T_hot≈{T_trigger:.0f} °C (t≈{res_double['t_snapshot']:.3f} s)")
    fig.tight_layout(); fig.savefig("heatmap_double_at_911C.png", dpi=300); plt.close(fig)

# Double layer heatmaps at specified times — unified color scale 0–1000 °C
for t_req, snap in res_double["snapshots_by_time"].items():
    if snap is None:
        print(f"[Warn] Requested t={t_req:.3f} s is beyond the available time range (no snapshot).")
        continue
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(snap, origin="lower", cmap="inferno",
                   extent=[0, L*100, 0, L*100], vmin=VMIN, vmax=VMAX)
    fig.colorbar(im, ax=ax, label="Temperature (°C)")
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title(f"Double layer — heat map at t={t_req*1e3:.0f} ms")
    fig.tight_layout()
    fname = f"heatmap_double_at_{int(round(t_req*1e3))}ms.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)

print("\nSaved:")
print(" - single_layer_probes.csv  (contains T_hot and the 0.5/1.0/2.0 mm curves, plus an alias column 0p5mm)")
print(" - double_layer_mid_gap_probe.csv  (contains T_hot and the mid-gap 0.5 mm curve)")
print(" - heatmap_single_at_911C.png")
print(" - heatmap_double_at_911C.png")
print(" - heatmap_double_at_0ms.png, 4ms, 8ms, 12ms, 16ms, 20ms, 24ms (if a time exceeds data duration, the corresponding file is not generated)")

