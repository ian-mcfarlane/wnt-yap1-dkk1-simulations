#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite comparison figure — atrophy (A,B,C,D) vs healthy boxplots overlay.

Layout (single row):
 A | cbar | B | C | D

A : Bound WNT (C) kymograph (atrophy, rep0)
cbar : colourbar for A (own gridspec column; never overlaps B)
B : time-aggregated division-position histogram (atrophy; ONLY divisions with t>=100)
C : Nuclear YAP quartile boxplots: atrophy (t>=100; y-positions compressed by atrophic length) vs healthy (t>=0)
D : Bound DKK1 quartile boxplots: atrophy (t>=100; y-positions compressed by atrophic length) vs healthy (t>=0)

Key requirements implemented:
- Atrophy boxplot *positions* use final_length_max computed from t>=100 only.
- Atrophy boxplot *data* use t>=100 only (and rolling windows exclude t<100).
- Healthy boxplot data use t>=0.
- Colourbar cannot overlap histogram (separate gridspec column).
"""
import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# -----------------------------
# Plot style
# -----------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 12,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

doc_width  = 5.78851
golden     = (np.sqrt(5) - 1) / 2
row_height = doc_width * golden * 0.6
FIGSIZE = (doc_width, row_height * 1.05)

# -----------------------------
# Data paths / selection
# -----------------------------
BASE_DIR_ATROPHY = "./outputs_params_refined/rapid_slough"

TARGET = dict(beta=1.5, kappa=1350, a_step=0.0, b_step=0.7, phiD=61.7, Wdiv=0.0098)
BASE_DIR_TMPL = "./outputs_params_refined/baseline/healthy"

KAPPA    = float(TARGET["kappa"])
W_DIV  = float(TARGET["Wdiv"])
A_STEP = float(TARGET["a_step"])
B_STEP = float(TARGET["b_step"])
PHI    = float(TARGET["phiD"])

TMIN_ATROPHY = 100.0  # atrophy homeostasis cutoff
WINDOW_DAYS  = 2.5

# -----------------------------
# Dimensional scales
# -----------------------------
LENGTH_SCALE_UM = 600.0
TIME_LABEL = r"Time, $t$ (days)"
POSITION_LABEL = r"Position, $x$ ($\mu\mathrm{m}$)"

FORCE_SCALE_NN = 49.05 * LENGTH_SCALE_UM
CONCENTRATION_LABEL = "Concentration\n(nM)"
FRACTION_LABEL = "Fraction"
OCCUPANCY_LABEL = "Receptor\nOccupancy"

# -----------------------------
# Filename parsing helpers
# -----------------------------
FN_RE = re.compile(
    r"kappa(?P<kappa>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)_"
    r"phiD(?P<phiD>-?\d+(?:\.\d+)?)_"
    r"Wdiv(?P<Wdiv>-?\d+(?:\.\d+)?)_"
    r"a(?P<a>-?\d+(?:\.\d+)?)_"
    r"b(?P<b>-?\d+(?:\.\d+)?)_"
    r"rep(?P<rep>\d+)\.pkl$"
)

def parse_from_name(fn):
    m = FN_RE.search(os.path.basename(fn))
    if not m:
        return None
    d = m.groupdict()
    return dict(
        kappa=float(d["kappa"]),
        phiD=float(d["phiD"]),
        Wdiv=float(d["Wdiv"]),
        a_step=float(d["a"]),
        b_step=float(d["b"]),
        rep=int(d["rep"]),
    )

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Kymograph helpers (same logic as your other scripts)
# -----------------------------
def detect_new_node(old_x, xs_k):
    if len(xs_k) != len(old_x) + 1:
        raise ValueError("Expected xs_k to have one more node than old_x.")
    tree = cKDTree(old_x[:, None])
    dists, _ = tree.query(xs_k[:, None], k=1)
    new_idx = int(np.argmax(dists))
    return xs_k[new_idx], new_idx

def build_segments(all_t, all_xs, all_values, division_positions=None):
    if division_positions is None:
        division_positions = []
    seg_times, seg_xedges, seg_vals = [], [], []
    if len(all_t) == 0:
        return seg_times, seg_xedges, seg_vals

    cur_t, cur_x, cur_v = [all_t[0]], [all_xs[0]], [all_values[0]]
    prev_N = len(all_xs[0]) - 1
    div_idx = 0

    for t_k, xs_k, v_k in zip(all_t[1:], all_xs[1:], all_values[1:]):
        N_k = len(xs_k) - 1
        if N_k != prev_N:
            T = np.array(cur_t)
            X = np.vstack(cur_x)
            V = np.vstack(cur_v)
            seg_times.append(T)
            seg_xedges.append(np.vstack([X, X[-1:]]))
            seg_vals.append(V)

            old_x = cur_x[-1]
            old_v = cur_v[-1]

            if N_k > prev_N:  # division
                if div_idx < len(division_positions):
                    p = float(division_positions[div_idx]); div_idx += 1
                    new_node = p * xs_k[-1]
                    xs_prev_with_new = np.sort(np.concatenate([old_x, [new_node]]))
                    insert_pos = np.searchsorted(old_x, new_node)
                    parent_cell = insert_pos - 1
                    if 0 <= parent_cell < prev_N:
                        v_prev_with_split = np.insert(old_v, parent_cell + 1, old_v[parent_cell])
                    else:
                        xs_prev_with_new = xs_k.copy()
                        v_prev_with_split = v_k.copy()
                else:
                    try:
                        new_node, insert_pos = detect_new_node(old_x, xs_k)
                        xs_prev_with_new = np.sort(np.concatenate([old_x, [new_node]]))
                        parent_cell = insert_pos - 1
                        if 0 <= parent_cell < prev_N:
                            v_prev_with_split = np.insert(old_v, parent_cell + 1, old_v[parent_cell])
                        else:
                            v_prev_with_split = v_k.copy()
                    except Exception:
                        xs_prev_with_new = xs_k.copy()
                        v_prev_with_split = v_k.copy()

                cur_t = [cur_t[-1], t_k]
                cur_x = [xs_prev_with_new, xs_k * LENGTH_SCALE_UM]  # convert to physical units
                cur_v = [v_prev_with_split, v_k]

            else:  # sloughing
                prev_removed = old_x[:-1]
                prev_v = old_v[:-1] if len(old_v) > 1 else old_v
                cur_t = [cur_t[-1], t_k]
                cur_x = [prev_removed, xs_k * LENGTH_SCALE_UM]
                cur_v = [prev_v, v_k]

            prev_N = N_k
        else:
            cur_t.append(t_k)
            cur_x.append(xs_k * LENGTH_SCALE_UM)  # convert to physical units
            cur_v.append(v_k)

    T = np.array(cur_t)
    X = np.vstack(cur_x)
    V = np.vstack(cur_v)
    seg_times.append(T)
    seg_xedges.append(np.vstack([X, X[-1:]]))
    seg_vals.append(V)

    ok = [i for i, V in enumerate(seg_vals) if V.size > 0 and np.isfinite(V).all()]
    return ([seg_times[i] for i in ok],
            [seg_xedges[i] for i in ok],
            [seg_vals[i] for i in ok])

def extract_division_events_from_lengths(all_t, num_cells, lengths, division_positions):
    """Map fractional division positions -> absolute x using the length at that timepoint."""
    t_ev, x_ev = [], []
    if len(all_t) == 0:
        return np.array([]), np.array([])
    prev_N = int(num_cells[0]) if len(num_cells) else 0
    idx = 0
    for k in range(1, len(all_t)):
        N_k = int(num_cells[k])
        if N_k == prev_N + 1 and idx < len(division_positions):
            t_ev.append(all_t[k])
            x_ev.append(float(division_positions[idx]) * float(lengths[k]))
            idx += 1
        prev_N = N_k
    return np.asarray(t_ev, float), np.asarray(x_ev, float)

# -----------------------------
# Atrophy: load only matching (kappa,phi,Wdiv,a,b) files
# -----------------------------
def find_atrophy_files(base_dir, phi):
    patt = os.path.join(
        base_dir,
        f"*kappa{KAPPA:.2e}_phiD{phi:.1f}_Wdiv{W_DIV:.4f}_a{A_STEP:.2f}_b{B_STEP:.2f}_rep*.pkl"
    )
    print("Looking for atrophy files with pattern:", patt)
    return sorted(glob.glob(patt))

def prep_atrophy_row(base_dir, phi):
    files = find_atrophy_files(base_dir, phi)
    if not files:
        return None

    rep0_path = None
    for f in files:
        md = parse_from_name(f)
        if md and md["rep"] == 0:
            rep0_path = f
            break
    if rep0_path is None:
        rep0_path = files[0]

    d0 = load_pkl(rep0_path)
    all_t0 = np.asarray(d0.get("all_t", []), float)
    all_x0 = d0.get("all_xs", [])
    all_C0 = d0.get("all_C", [])
    divpos0 = d0.get("division_positions", [])
    segs_C = build_segments(all_t0, all_x0, all_C0, divpos0)

    t_div_all, x_div_all = [], []
    times_all, lengths_all = [], []
    sims = []
    for fp in files:
        try:
            sim = load_pkl(fp)
        except Exception:
            continue
        sims.append(sim)

        tt = np.asarray(sim.get("all_t", []), float)
        nn = np.asarray(sim.get("num_cells", []), int)
        ll = np.asarray(sim.get("lengths", []), float)

        if tt.size and ll.size:
            times_all.append(tt)
            lengths_all.append(ll)

        td, xd = extract_division_events_from_lengths(tt, nn, ll, sim.get("division_positions", []))
        if td.size:
            t_div_all.append(td)
            x_div_all.append(xd)

    return dict(
        sims=sims,
        segs_C=segs_C,
        t0=all_t0,
        lengths0=np.asarray(d0.get("lengths", []), float),
        t_div=(np.concatenate(t_div_all) if t_div_all else np.array([])),
        x_div=(np.concatenate(x_div_all) if x_div_all else np.array([])),
        times_all=times_all,
        lengths_all=lengths_all
    )

# -----------------------------
# Healthy helper: load_all_matching (for pooled baseline sims)
# -----------------------------
def load_all_matching(dir_suff, target=TARGET, atol=1e-9, rtol=1e-9):
    outdir = BASE_DIR_TMPL.format(dir_suff, target["beta"])
    files = sorted(glob.glob(os.path.join(outdir, "*.pkl")))
    sims = []
    for fn in files:
        md = parse_from_name(fn)
        ok = False
        if md is not None:
            ok = (np.isclose(md["kappa"], target["kappa"], atol=atol, rtol=rtol) and
                  np.isclose(md["phiD"], target["phiD"], atol=atol, rtol=rtol) and
                  np.isclose(md["Wdiv"], target["Wdiv"], atol=atol, rtol=rtol) and
                  np.isclose(md["a_step"], target["a_step"], atol=atol, rtol=rtol) and
                  np.isclose(md["b_step"], target["b_step"], atol=atol, rtol=rtol))
        if not ok:
            # fallback: key_parameters
            try:
                sim = load_pkl(fn)
            except Exception:
                continue
            kp = sim.get("key_parameters", {})
            if kp:
                ok = (np.isclose(kp.get("kappa_spring0", np.nan), target["kappa"], atol=atol, rtol=rtol) and
                      np.isclose(kp.get("phi_D", np.nan),        target["phiD"], atol=atol, rtol=rtol) and
                      np.isclose(kp.get("W_div_switch", np.nan), target["Wdiv"], atol=atol, rtol=rtol) and
                      np.isclose(kp.get("a_step", np.nan),       target["a_step"], atol=atol, rtol=rtol) and
                      np.isclose(kp.get("b_step", np.nan),       target["b_step"], atol=atol, rtol=rtol))
            if not ok:
                continue
            sims.append(sim)
            continue

        try:
            sims.append(load_pkl(fn))
        except Exception:
            continue

    return sims

# -----------------------------
# Quartile rolling helpers (time-filtered correctly)
# -----------------------------
def _first_key(d, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

def _quartile_means_for_snapshot(xs_k, vals_k):
    """Return 4 means over fractional quartiles [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]."""
    if xs_k is None or len(xs_k) < 2:
        return [np.nan] * 4
    Lk = float(xs_k[-1])
    if Lk <= 0:
        return [np.nan] * 4
    centres = 0.5 * (np.asarray(xs_k[:-1]) + np.asarray(xs_k[1:]))
    frac = centres / Lk
    vals = np.asarray(vals_k, float)

    edges = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]
    out = []
    for j, (lo, hi) in enumerate(edges):
        if j < 3:
            m = (frac >= lo) & (frac < hi)
        else:
            m = (frac >= lo) & (frac <= hi)
        out.append(np.nanmean(vals[m]) if np.any(m) else np.nan)
    return out

def rolling_quartile_bins(all_t, all_xs, series, window_days=2.5, t_min=0.0):
    """
    Centred rolling averages of quartile means, pooled as samples.
    Enforces:
      - anchor times ti must satisfy ti >= t_min
      - windows only use samples with t >= t_min (so atrophy never mixes in t<100)
    """
    if series is None or len(all_t) == 0:
        return [np.array([]) for _ in range(4)]

    all_t = np.asarray(all_t, float)

    qmeans = []
    for xs_k, vals_k in zip(all_xs, series):
        qmeans.append(_quartile_means_for_snapshot(xs_k, vals_k))
    qmeans = np.asarray(qmeans, float)  # (nt,4)

    halfW = 0.5 * float(window_days)
    bins = [[] for _ in range(4)]

    for i, ti in enumerate(all_t):
        if ti < t_min:
            continue
        lo, hi = ti - halfW, ti + halfW
        m = (all_t >= lo) & (all_t <= hi) & (all_t >= t_min)
        if not np.any(m):
            continue
        win_mean = np.nanmean(qmeans[m, :], axis=0)
        for j in range(4):
            if np.isfinite(win_mean[j]):
                bins[j].append(win_mean[j])

    return [np.asarray(b, float) if len(b) else np.array([]) for b in bins]

def finalise_bins(bins_list):
    out = []
    for b in bins_list:
        out.append(np.asarray(b, float) if len(b) else np.array([np.nan]))
    return out

def collect_bins_Y_I(sims, t_min=0.0, window_days=2.5):
    """Return pooled rolling bins for Y and I (4 arrays each)."""
    Y_bins_all = [[] for _ in range(4)]
    I_bins_all = [[] for _ in range(4)]

    for sim in sims:
        all_t  = sim.get("all_t", [])
        all_xs = sim.get("all_xs", [])
        all_L0s = sim.get("all_L0s", [])
        all_I  = _first_key(sim, "all_DL", "all_I")
        all_Y  = _first_key(sim, "all_YAP", "all_Y", "all_Y2")

        # get receptor concentration, given by L0/L, where L = x_i-x_{i-1} is the cell length, and L0 is the reference length
        all_Rs = [np.asarray(L0_k, float) / np.diff(xs_k) for xs_k, L0_k in zip(all_xs, all_L0s)]

        # scale DL and C by receptor concentration to get occupancy
        if all_I is not None:
            all_I = [np.asarray(v, float) / R_k for v, R_k in zip(all_I, all_Rs)]


        bins_I = rolling_quartile_bins(all_t, all_xs, all_I, window_days=window_days, t_min=t_min) if all_I is not None else [np.array([])] * 4
        bins_Y = rolling_quartile_bins(all_t, all_xs, all_Y, window_days=window_days, t_min=t_min) if all_Y is not None else [np.array([])] * 4

        for j in range(4):
            if bins_I[j].size:
                I_bins_all[j].extend(bins_I[j].tolist())
            if bins_Y[j].size:
                Y_bins_all[j].extend(bins_Y[j].tolist())

    return finalise_bins(Y_bins_all), finalise_bins(I_bins_all)

# -----------------------------
# Length scaling helpers
# -----------------------------
def max_length_over_all_time(times_all, lengths_all, fallback=1.0):
    mx = 0.0
    for tt, ll in zip(times_all, lengths_all):
        ll = np.asarray(ll, float)
        if ll.size:
            mx = max(mx, float(np.nanmax(ll)))
    return mx if mx > 0 else float(fallback)

def max_length_post_tmin(times_all, lengths_all, t_min=100.0, fallback=1.0):
    """
    This is the critical quantity for the atrophy boxplot y-position compression.
    Uses ONLY data with t >= t_min (falls back to whole series if no such times exist).
    """
    mx = 0.0
    for tt, ll in zip(times_all, lengths_all):
        tt = np.asarray(tt, float)
        ll = np.asarray(ll, float)
        if tt.size == 0 or ll.size == 0:
            continue
        n = min(tt.size, ll.size)
        tt = tt[:n]; ll = ll[:n]
        m = tt >= t_min
        if np.any(m):
            mx = max(mx, float(np.nanmax(ll[m])))
        else:
            mx = max(mx, float(np.nanmax(ll)))
    return mx if mx > 0 else float(fallback)

# -----------------------------
# Load data
# -----------------------------
row_at = prep_atrophy_row(BASE_DIR_ATROPHY, PHI)
if row_at is None:
    raise RuntimeError("No atrophy data found. Check BASE_DIR_ATROPHY and parameter filters.")

healthy_sims = load_all_matching("Y2")
if not healthy_sims:
    raise RuntimeError("No healthy data found. Check BASE_DIR_TMPL and TARGET.")

# pooled bins (data selection is correct here)
Y_at, I_at = collect_bins_Y_I(row_at["sims"], t_min=TMIN_ATROPHY, window_days=WINDOW_DAYS)
Y_h,  I_h  = collect_bins_Y_I(healthy_sims,   t_min=0.0,          window_days=WINDOW_DAYS)

# kymograph normalisation
def _minmax(seg_vals):
    vmin, vmax = np.inf, -np.inf
    for V in seg_vals:
        if V.size:
            vmin = min(vmin, float(np.nanmin(V)))
            vmax = max(vmax, float(np.nanmax(V)))
    return vmin, vmax

vmin_at, vmax_at = _minmax(row_at["segs_C"][2]) if row_at["segs_C"] else (0.0, 1.0)
normC = Normalize(vmin=vmin_at, vmax=vmax_at)

# axes limits for A/B (show full trajectory, so use max over all time)
tmax = float(row_at["t0"][-1]) if row_at["t0"].size else 0.0
xmax_plot = max_length_over_all_time(row_at["times_all"], row_at["lengths_all"], fallback=1.0)

# critical: atrophy length used for compressing quartile *positions* (post-100 only)
final_length_max = max_length_post_tmin(row_at["times_all"], row_at["lengths_all"], t_min=TMIN_ATROPHY, fallback=1.0)
print("xmax_plot (all time) =", xmax_plot)
print("final_length_max (t>=100) =", final_length_max)

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)

# 5 columns: A | cbar | B | C | D  (prevents any overlap by construction)
widths = [1.0, 0.06, 0.22, 1.0, 1.0]
gs = GridSpec(1, 5, figure=fig, width_ratios=widths, wspace=0.08)

axA  = fig.add_subplot(gs[0, 0])
axCB = fig.add_subplot(gs[0, 1])
axB  = fig.add_subplot(gs[0, 2])
axC  = fig.add_subplot(gs[0, 3])
axD  = fig.add_subplot(gs[0, 4])

# ---- Panel A: kymograph (atrophy) ----
last_pcm = None
for T, Xedges, V in zip(*row_at["segs_C"]):
    Ni1 = Xedges.shape[1]
    Tedges = np.vstack([T[:, None] * np.ones((1, Ni1)), T[-1] * np.ones((1, Ni1))])
    last_pcm = axA.pcolormesh(Tedges, Xedges, V, cmap="inferno",
                              shading="flat", norm=normC,
                              rasterized=True, antialiased=False)

axA.set_title("Bound WNT\n" + r"(active $\beta$-catenin)")
axA.set_xlabel(r"Time, $t$")
axA.set_ylabel(r"Position, $x$")

# axes limits for A/B (physical length in um)
tmax = float(row_at["t0"][-1]) if row_at["t0"].size else 0.0
xmax_plot = max_length_over_all_time(row_at["times_all"], row_at["lengths_all"], fallback=1.0) * LENGTH_SCALE_UM

# critical: atrophy length used for compressing quartile positions
# now in physical units
final_length_max = max_length_post_tmin(
    row_at["times_all"],
    row_at["lengths_all"],
    t_min=TMIN_ATROPHY,
    fallback=1.0,
) * LENGTH_SCALE_UM

print("xmax_plot (all time, um) =", xmax_plot)
print("final_length_max (t>=100, um) =", final_length_max)

# ---- Panel A: kymograph ----
axA.set_title("Bound WNT\n" + r"(active $\beta$-catenin)")
axA.set_xlabel(TIME_LABEL)
axA.set_ylabel(POSITION_LABEL)
axA.set_xlim(0.0, tmax)
axA.set_ylim(0.0, xmax_plot)
axA.grid(alpha=0.2)

# colourbar (dedicated axis; cannot overlap B)
if last_pcm is not None:
    cb = fig.colorbar(last_pcm, cax=axCB)
    axCB.yaxis.set_ticks_position("right")
    axCB.yaxis.set_label_position("right")

# ---- Panel B: division frequency histogram (atrophy, ONLY t>=100 divisions) ----
Tdiv = np.asarray(row_at.get("t_div", np.array([])), float)
Xdiv = np.asarray(row_at.get("x_div", np.array([])), float)

if Tdiv.size and Xdiv.size:
    m = Tdiv >= TMIN_ATROPHY
    Xlate = Xdiv[m] * LENGTH_SCALE_UM
else:
    Xlate = np.array([])


if Xlate.size:
    counts, edges = np.histogram(Xlate, bins=40, range=(0.0, xmax_plot))
    centres = 0.5 * (edges[:-1] + edges[1:])
    height = edges[1] - edges[0]
    axB.barh(
        centres, counts, height=0.9 * height,
        align="center", color="C0", edgecolor="C0", alpha=0.75
    )
else:
    axB.text(0.5, 0.5, "no divisions", ha="center", va="center", transform=axB.transAxes)

axB.set_title("Division\nfrequency")
axB.set_xlabel("Frequency")
axB.set_ylim(0.0, xmax_plot)
axB.set_yticks([])
axB.grid(alpha=0.2, axis="x")

# ---- compute boxplot positions: no offset (place both sets at the quartile centres) ----
quart_centres = np.array([0.125, 0.375, 0.625, 0.875])

# positions are now dimensional
pos_at = quart_centres * final_length_max
pos_h = quart_centres * LENGTH_SCALE_UM


def draw_overlay_boxplots(ax, bins_at, bins_h, title, xlabel, add_legend=False):
    bp_at = ax.boxplot(
        bins_at, vert=False, positions=pos_at, widths=0.1*LENGTH_SCALE_UM,
        manage_ticks=False, showmeans=False, showfliers=False, patch_artist=True,
        whiskerprops=dict(alpha=0.9, color="C1", linewidth=1.0, zorder=3),
        capprops=dict(alpha=0.9, color="C1", linewidth=1.0, zorder=3),
        boxprops=dict(alpha=0.60, facecolor="C1", edgecolor="C1", zorder=3)
    )
    for med in bp_at.get("medians", []):
        med.set_color("C1")
        med.set_linewidth(1.0)

    bp_h = ax.boxplot(
        bins_h, vert=False, positions=pos_h, widths=0.1*LENGTH_SCALE_UM,
        manage_ticks=False, showmeans=False, showfliers=False, patch_artist=True,
        whiskerprops=dict(alpha=0.9, color="C0", linewidth=1.0, zorder=2),
        capprops=dict(alpha=0.9, color="C0", linewidth=1.0, zorder=2),
        boxprops=dict(alpha=0.45, facecolor="C0", edgecolor="C0", zorder=2)
    )
    for med in bp_h.get("medians", []):
        med.set_color("C0")
        med.set_linewidth(1.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xlim(None, None)
    ax.set_ylim(0.0, xmax_plot)

    # no repeated position labelling here; the kymograph carries it
    # ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(alpha=0.25)

    if add_legend:
        handles = [
            Patch(facecolor="C1", edgecolor="C1", alpha=0.6, label="Atrophy"),
            Patch(facecolor="C0", edgecolor="C0", alpha=0.45, label="Healthy"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True)
    

draw_overlay_boxplots(axC, Y_at, Y_h, title="Nuclear YAP", xlabel=FRACTION_LABEL, add_legend=True)
draw_overlay_boxplots(axD, I_at, I_h, title="Bound DKK1", xlabel=OCCUPANCY_LABEL, add_legend=False)

axA.yaxis.set_major_locator(MaxNLocator(nbins=5))
axA.set_ylabel(POSITION_LABEL)

# Save
save_dir = "./plots"
os.makedirs(save_dir, exist_ok=True)
outfile = os.path.join(save_dir, "fig7.pdf")
fig.savefig(outfile, dpi=300, bbox_inches="tight")
print("Saved:", outfile)