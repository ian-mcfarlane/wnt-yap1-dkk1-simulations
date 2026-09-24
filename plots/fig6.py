import os
import re
import glob
import pickle
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from scipy.spatial import cKDTree

# -----------------------------
# Plot style and size
# -----------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 12,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

doc_width = 5.78851
golden = (np.sqrt(5) - 1) / 2
row_height = doc_width * golden * 0.6

# -----------------------------
# Parameters and dimensional scales
# -----------------------------
TARGET = dict(beta=1.5, kappa=1350, a_step=0.0, b_step=0.7, phiD=61.7, Wdiv=0.0098)

def fmt_phiD(phiD):
    return str(int(phiD)) if float(phiD).is_integer() else str(phiD)

# BASE_DIR = (
#     "./outputs_m_continuous_Ksigma_changed/baseline/"
#     f"baseline_phiD{fmt_phiD(TARGET['phiD'])}/"
# )

BASE_DIR = (
    "./outputs_params_refined/baseline/healthy"
)

LENGTH_SCALE_UM = 600.0
FREE_WNT_SCALE_NM = 140.0
FORCE_SCALE_NN = 49.05 * LENGTH_SCALE_UM  # 1350 nN/um * 600 um

TIME_LABEL = r"Time, $t$ (days)"
POSITION_LABEL = r"Position, $x$ ($\mu\mathrm{m}$)"
LENGTH_LABEL = ""
RELATIVE_POSITION_LABEL = r"Relative position, $x/L$"

FORCE_LABEL = "Force (nN)"
FRACTION_LABEL = "Fraction"
OCCUPANCY_LABEL = "Receptor\nOccupancy"
CONCENTRATION_LABEL = "Concentration\n(nM)"

# -----------------------------
# Filename parser
# -----------------------------
FN_RE = re.compile(
    r"kappa(?P<kappa>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)_"
    r"phiD(?P<phiD>-?\d+(?:\.\d+)?)_"
    r"Wdiv(?P<Wdiv>-?\d+(?:\.\d+)?)_"
    r"a(?P<a>-?\d+(?:\.\d+)?)_"
    r"b(?P<b>-?\d+(?:\.\d+)?)_"
    r"rep(?P<rep>\d+)\.pkl$"
)

def parse_from_name(filename):
    match = FN_RE.search(os.path.basename(filename))
    if match is None:
        return None

    fields = match.groupdict()
    return {
        "kappa": float(fields["kappa"]),
        "phiD": float(fields["phiD"]),
        "Wdiv": float(fields["Wdiv"]),
        "a_step": float(fields["a"]),
        "b_step": float(fields["b"]),
        "rep": int(fields["rep"]),
    }

# -----------------------------
# Data loading
# -----------------------------
def load_all_matching(target=TARGET, atol=1e-9, rtol=1e-9):
    files = sorted(glob.glob(os.path.join(BASE_DIR, "*.pkl")))
    sims, meta = [], []
    
    for filename in files:
        parsed = parse_from_name(filename)
        matches_target = False

        if parsed is not None:
            matches_target = (
                np.isclose(parsed["kappa"], target["kappa"], atol=atol, rtol=rtol) and
                np.isclose(parsed["phiD"], target["phiD"], atol=atol, rtol=rtol) and
                np.isclose(parsed["Wdiv"], target["Wdiv"], atol=atol, rtol=rtol) and
                np.isclose(parsed["a_step"], target["a_step"], atol=atol, rtol=rtol) and
                np.isclose(parsed["b_step"], target["b_step"], atol=atol, rtol=rtol)
            )

        try:
            with open(filename, "rb") as f:
                sim = pickle.load(f)
        except Exception:
            continue

        if not matches_target:
            key_parameters = sim.get("key_parameters", {})
            if not key_parameters:
                continue

            matches_target = (
                np.isclose(key_parameters.get("kappa", np.nan), target["kappa"], atol=atol, rtol=rtol) and
                np.isclose(key_parameters.get("phi_D", np.nan), target["phiD"], atol=atol, rtol=rtol) and
                np.isclose(key_parameters.get("W_div_switch", np.nan), target["Wdiv"], atol=atol, rtol=rtol) and
                np.isclose(key_parameters.get("a_step", np.nan), target["a_step"], atol=atol, rtol=rtol) and
                np.isclose(key_parameters.get("b_step", np.nan), target["b_step"], atol=atol, rtol=rtol)
            )

        if not matches_target:
            continue

        rep = parsed["rep"] if parsed is not None else -1
        sims.append(sim)
        meta.append({"path": filename, "rep": rep})

    return sims, meta

# -----------------------------
# Kymograph helpers
# -----------------------------
def detect_new_node(old_x, xs_k, ratio_threshold=4.0):
    if len(xs_k) != len(old_x) + 1:
        raise ValueError("Expected xs_k to have one more node than old_x.")

    tree = cKDTree(old_x[:, None])
    dists, _ = tree.query(xs_k[:, None], k=1)
    new_idx = int(np.argmax(dists))

    sorted_dists = np.sort(dists)[::-1]
    if len(sorted_dists) > 1 and sorted_dists[0] < ratio_threshold * sorted_dists[1]:
        pass

    return xs_k[new_idx], new_idx

def build_segments(all_t, all_xs, all_values, division_positions=None):
    if division_positions is None:
        division_positions = []

    seg_times, seg_xedges, seg_vals = [], [], []

    current_t = [all_t[0]]
    current_x = [all_xs[0]]
    current_v = [all_values[0]]

    previous_n_cells = len(all_xs[0]) - 1
    division_index = 0

    for t_k, xs_k, v_k in zip(all_t[1:], all_xs[1:], all_values[1:]):
        current_n_cells = len(xs_k) - 1

        if current_n_cells != previous_n_cells:
            T = np.array(current_t)
            X = np.vstack(current_x)
            V = np.vstack(current_v)

            seg_times.append(T)
            seg_xedges.append(np.vstack([X, X[-1:]]))
            seg_vals.append(V)

            old_x = current_x[-1]
            old_v = current_v[-1]

            if current_n_cells > previous_n_cells:
                # Cell division
                if division_index < len(division_positions):
                    frac_pos = float(division_positions[division_index])
                    division_index += 1
                    new_node = frac_pos * xs_k[-1]

                    xs_prev_with_new = np.sort(np.concatenate([old_x, [new_node]]))
                    insert_pos = np.searchsorted(old_x, new_node)
                    parent_cell = insert_pos - 1

                    if 0 <= parent_cell < previous_n_cells:
                        v_prev_with_split = np.insert(old_v, parent_cell + 1, old_v[parent_cell])
                    else:
                        xs_prev_with_new = xs_k.copy()
                        v_prev_with_split = v_k.copy()
                else:
                    try:
                        new_node, insert_pos = detect_new_node(old_x, xs_k)
                        xs_prev_with_new = np.sort(np.concatenate([old_x, [new_node]]))
                        parent_cell = insert_pos - 1

                        if 0 <= parent_cell < previous_n_cells:
                            v_prev_with_split = np.insert(old_v, parent_cell + 1, old_v[parent_cell])
                        else:
                            v_prev_with_split = v_k.copy()
                    except Exception:
                        xs_prev_with_new = xs_k.copy()
                        v_prev_with_split = v_k.copy()

                current_t = [current_t[-1], t_k]
                current_x = [xs_prev_with_new, xs_k]
                current_v = [v_prev_with_split, v_k]

            else:
                # Sloughing event
                prev_removed = old_x[:-1]
                prev_v = old_v[:-1] if len(old_v) > 1 else old_v

                current_t = [current_t[-1], t_k]
                current_x = [prev_removed, xs_k]
                current_v = [prev_v, v_k]

            previous_n_cells = current_n_cells

        else:
            current_t.append(t_k)
            current_x.append(xs_k)
            current_v.append(v_k)

    T = np.array(current_t)
    X = np.vstack(current_x)
    V = np.vstack(current_v)

    seg_times.append(T)
    seg_xedges.append(np.vstack([X, X[-1:]]))
    seg_vals.append(V)

    valid = [i for i, V in enumerate(seg_vals) if V.size > 0 and np.isfinite(V).all()]
    seg_times = [seg_times[i] for i in valid]
    seg_xedges = [seg_xedges[i] for i in valid]
    seg_vals = [seg_vals[i] for i in valid]

    return seg_times, seg_xedges, seg_vals

def scale_segment_x_coordinates(segments, length_scale):
    seg_times, seg_xedges, seg_vals = segments
    scaled_xedges = [X * length_scale for X in seg_xedges]
    return seg_times, scaled_xedges, seg_vals

# -----------------------------
# Division-event extraction
# -----------------------------
def extract_division_events_from_lengths(all_t, num_cells, lengths, division_positions):
    """
    Extract division event times and absolute positions from per-snapshot data.
    Returns:
        t_ev  : event times
        x_ev  : event positions
        L_ev  : gland length at each event time
    """
    t_ev, x_ev, L_ev = [], [], []

    previous_n_cells = int(num_cells[0]) if len(num_cells) else 0
    division_index = 0

    for k in range(1, len(all_t)):
        current_n_cells = int(num_cells[k])

        if current_n_cells == previous_n_cells + 1 and division_index < len(division_positions):
            t_k = float(all_t[k])
            L_k = float(lengths[k]) if k < len(lengths) else float(lengths[-1]) if len(lengths) else 1.0
            frac_pos = float(division_positions[division_index])
            x_k = frac_pos * L_k

            t_ev.append(t_k)
            x_ev.append(x_k)
            L_ev.append(L_k)

            division_index += 1

        previous_n_cells = current_n_cells

    return np.asarray(t_ev, float), np.asarray(x_ev, float), np.asarray(L_ev, float)

# -----------------------------
# Data preparation
# -----------------------------
def prepare_baseline_dataset():
    sims, meta = load_all_matching()

    if not sims:
        return {
            "segs_C": ([], [], []),
            "t0": np.array([]),
            "lengths0": np.array([]),
            "t_div": np.array([]),
            "x_div": np.array([]),
            "lengths_at_div": np.array([]),
            "times_all": [],
            "lengths_all": [],
            "tmax": 0.0,
            "xmax": 0.0,
        }

    # Choose the replicate whose mean non-dimensional length is closest to 1
    mean_lengths = []
    for sim in sims:
        lengths = np.asarray(sim.get("lengths", []), float)
        mean_lengths.append(lengths.mean() if lengths.size else np.nan)

    mean_lengths = np.asarray(mean_lengths, float)
    rep_index = int(np.nanargmin(np.abs(mean_lengths - 1.0)))
    sim0 = sims[rep_index]

    all_t0 = np.asarray(sim0["all_t"], float)
    all_x0 = sim0["all_xs"]
    all_C0 = sim0["all_C"]
    division_positions0 = sim0.get("division_positions", [])

    segs_C_raw = build_segments(all_t0, all_x0, all_C0, division_positions0)
    segs_C = scale_segment_x_coordinates(segs_C_raw, LENGTH_SCALE_UM)

    t_div_all = []
    x_div_all = []
    lengths_at_div_all = []
    times_all = []
    lengths_all = []

    for sim in sims:
        tt = np.asarray(sim.get("all_t", []), float)
        nn = np.asarray(sim.get("num_cells", []), int)
        ll = np.asarray(sim.get("lengths", []), float)

        times_all.append(tt)
        lengths_all.append(ll * LENGTH_SCALE_UM)

        td, xd, ld = extract_division_events_from_lengths(
            tt,
            nn,
            ll,
            sim.get("division_positions", [])
        )

        if td.size:
            t_div_all.append(td)
            x_div_all.append(xd * LENGTH_SCALE_UM)
            lengths_at_div_all.append(ld * LENGTH_SCALE_UM)

    lengths0 = np.asarray(sim0.get("lengths", []), float) * LENGTH_SCALE_UM

    tmax = max((t[-1] for t in times_all if len(t)), default=0.0)
    xmax = max((float(l.max()) for l in lengths_all if l.size), default=0.0)

    t_div = np.concatenate(t_div_all) if t_div_all else np.array([], float)
    x_div = np.concatenate(x_div_all) if x_div_all else np.array([], float)
    lengths_at_div = np.concatenate(lengths_at_div_all) if lengths_at_div_all else np.array([], float)

    return {
        "segs_C": segs_C,
        "t0": all_t0,
        "lengths0": lengths0,
        "t_div": t_div,
        "x_div": x_div,
        "lengths_at_div": lengths_at_div,
        "times_all": times_all,
        "lengths_all": lengths_all,
        "tmax": tmax,
        "xmax": xmax,
    }

# -----------------------------
# Small utilities
# -----------------------------
def _minmax(seg_vals):
    vmin, vmax = np.inf, -np.inf
    for V in seg_vals:
        if V.size:
            vmin = min(vmin, float(np.nanmin(V)))
            vmax = max(vmax, float(np.nanmax(V)))
    return vmin, vmax

def _first_key(d, *keys):
    for key in keys:
        if key in d:
            return d[key]
    return None

def _quartile_means_for_snapshot(xs_k, vals_k):
    """
    Mean value in the four quartiles of fractional position x/L.
    """
    if xs_k is None or len(xs_k) < 2:
        return [np.nan] * 4

    Lk = float(xs_k[-1])
    if Lk <= 0:
        return [np.nan] * 4

    cell_centres = 0.5 * (np.asarray(xs_k[:-1]) + np.asarray(xs_k[1:]))
    frac_pos = cell_centres / Lk
    vals = np.asarray(vals_k, float)

    quartiles = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]
    means = []

    for j, (lo, hi) in enumerate(quartiles):
        if j < 3:
            mask = (frac_pos >= lo) & (frac_pos < hi)
        else:
            mask = (frac_pos >= lo) & (frac_pos <= hi)

        means.append(np.nanmean(vals[mask]) if np.any(mask) else np.nan)

    return means

def _rolling_quartile_means(all_t, all_xs, series, window_days=2.5):
    """
    Rolling averages of quartile means over a centred time window.
    Returns a list of four arrays, one for each quartile.
    """
    if series is None or len(all_t) == 0:
        return [np.array([]) for _ in range(4)]

    all_t = np.asarray(all_t, float)

    qmeans = []
    for xs_k, vals_k in zip(all_xs, series):
        qmeans.append(_quartile_means_for_snapshot(xs_k, vals_k))
    qmeans = np.asarray(qmeans, float)

    half_window = 0.5 * float(window_days)
    pooled = [[] for _ in range(4)]

    for i, ti in enumerate(all_t):
        mask = (all_t >= ti - half_window) & (all_t <= ti + half_window)
        if not np.any(mask):
            continue

        window_mean = np.nanmean(qmeans[mask, :], axis=0)
        for j in range(4):
            if np.isfinite(window_mean[j]):
                pooled[j].append(window_mean[j])

    return [np.asarray(values, float) if len(values) else np.array([]) for values in pooled]

def _finalise_bins(bins_list):
    out = []
    for values in bins_list:
        if len(values):
            out.append(np.asarray(values, float))
        else:
            out.append(np.array([np.nan]))
    return out

# -----------------------------
# Plot helpers
# -----------------------------
def plot_division_histogram_over_time(ax, x_positions, xmax_plot, bins=30):
    """
    Time-aggregated horizontal histogram of division positions.
    """
    if x_positions.size:
        counts, edges = np.histogram(x_positions, bins=bins, range=(0.0, xmax_plot))
        centres = 0.5 * (edges[:-1] + edges[1:])
        height = edges[1] - edges[0]

        ax.barh(
            centres,
            counts,
            height=0.9 * height,
            align="center",
            color="C0",
            edgecolor="C0",
            alpha=0.6,
        )
    else:
        ax.text(0.5, 0.5, "no divisions", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("")
    ax.set_ylim(0.0, xmax_plot)
    ax.set_yticks([])
    ax.grid(alpha=0.2, axis="x")

def plot_divisions(ax, t_events, x_events, tmax, xmax_plot):
    if t_events.size:
        ax.hexbin(
            t_events,
            x_events,
            gridsize=60,
            extent=(0.0, tmax, 0.0, xmax_plot),
            cmap="magma",
            mincnt=1,
        )
    else:
        ax.text(0.5, 0.5, "no divisions", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlim(0.0, tmax)
    ax.set_ylim(0.0, xmax_plot)
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel('')
    ax.grid(alpha=0.2)

def plot_lengths(ax, times_all, lengths_all, t0, L0, tmax, xmax_plot):
    for tt, ll in zip(times_all, lengths_all):
        if len(tt) and len(ll):
            ax.plot(tt, ll, color="0.8", lw=1.0, alpha=0.7)

    if len(t0) and len(L0):
        ax.plot(t0, L0, color="C2", lw=2.2)

    ax.set_title("Tissue length")
    ax.set_xlim(0.0, tmax)
    ax.set_ylim(0.0, xmax_plot)
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel(LENGTH_LABEL)
    ax.grid(alpha=0.2)

def shared_division_title(ax_left, ax_right, text):
    """
    Place one centred title above a pair of axes.
    """
    posL = ax_left.get_position()
    posR = ax_right.get_position()

    x0_left = posL.x0
    x1_left = posL.x1
    x1_right = posR.x1

    midpoint_fig = 0.5 * (x0_left + x1_right)
    midpoint_axes = (midpoint_fig - x0_left) / (x1_left - x0_left)

    ax_left.set_title(text, x=midpoint_axes)

def horiz_boxplot(ax, bins, xlabel, title, nbins=None):
    box_color = "C0"
    bp = ax.boxplot(
        bins,
        vert=False,
        positions=[0.0, 0.125, 0.375, 0.625, 0.875, 1.0],
        widths=0.20,
        manage_ticks=False,
        showmeans=False,
        showfliers=False,
        patch_artist=True,
        whiskerprops=dict(alpha=0.7, color=box_color),
        capprops=dict(alpha=0.7, color=box_color),
        boxprops=dict(alpha=0.35, facecolor=box_color, edgecolor=box_color),
    )

    for median in bp.get("medians", []):
        median.set_color(box_color)
        median.set_linewidth(1.0)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    if "compression" in title.lower():
        # make sure 0 is the lower x-limit for compression plots
        ax.set_xlim(left=0.0)
    if nbins is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
    ax.set_yticklabels([r"$0$", "", "", "", "", r"$1$"])
    ax.grid(alpha=0.25)

# -----------------------------
# Prepare top-row data
# -----------------------------
Y = prepare_baseline_dataset()

vmin, vmax = _minmax(Y["segs_C"][2])
normC = Normalize(vmin=vmin, vmax=vmax)

tmax = Y["tmax"] if Y["tmax"] > 0 else 1.0
xmax_data = Y["xmax"] if Y["xmax"] > 0 else 1.0
xmax_plot = 1.1 * xmax_data

# -----------------------------
# Figure layout
# -----------------------------
H_top = 1.0 * row_height
H_gap = 0.22
H_bot = 0.95 * row_height
FIGSIZE = (doc_width, H_top + H_gap + H_bot)

fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[H_top, H_gap + H_bot])

# ---- TOP row ----
gs_top = GridSpecFromSubplotSpec(
    1, 5,
    subplot_spec=gs_outer[0],
    width_ratios=[1.0, 0.16, 1.0, 0.55, 1.0],
)

axA = fig.add_subplot(gs_top[0, 0])
axCB = fig.add_subplot(gs_top[0, 1])
axB = fig.add_subplot(gs_top[0, 2])
axBh = fig.add_subplot(gs_top[0, 3])
axC = fig.add_subplot(gs_top[0, 4])

# Bound WNT kymograph
last_pcm = None
for T, Xedges, V in zip(*Y["segs_C"]):
    n_edges = Xedges.shape[1]
    Tedges = np.vstack([T[:, None] * np.ones((1, n_edges)),
                        T[-1] * np.ones((1, n_edges))])

    last_pcm = axA.pcolormesh(
        Tedges,
        Xedges,
        V,
        cmap="inferno",
        shading="flat",
        norm=normC,
        rasterized=True,
        antialiased=False,
    )

axA.set_title(r"Bound WNT, $C_i$" + "\n" + r"(active $\beta$-catenin)")
axA.set_xlim(0.0, tmax)
axA.set_ylim(0.0, xmax_plot)
axA.set_xlabel(TIME_LABEL)
axA.set_ylabel(POSITION_LABEL)
axA.grid(alpha=0.2)

cbar = fig.colorbar(last_pcm, cax=axCB)
# cbar.set_label("Receptor occupancy")
axCB.yaxis.set_ticks_position("right")
axCB.yaxis.set_label_position("right")

# Division events + time-aggregated histogram
plot_divisions(axB, Y["t_div"], Y["x_div"], tmax, xmax_plot)
plot_division_histogram_over_time(axBh, Y["x_div"], xmax_plot)
shared_division_title(axB, axBh, "Division events")

# Tissue length
plot_lengths(axC, Y["times_all"], Y["lengths_all"], Y["t0"], Y["lengths0"], tmax, xmax_plot)

for ax in (axA, axB, axBh, axC, axCB):
    ax.set_zorder(0)

# ---- BOTTOM row ----
gs_bot = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1], height_ratios=[1e-6, H_bot])
gs_prof = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_bot[1, 0])
axs_prof = [fig.add_subplot(gs_prof[0, i]) for i in range(5)]

# -----------------------------
# Prepare pooled bottom-row summaries
# -----------------------------
sims, meta = load_all_matching()

if sims:
    sigma_bins_all = [[] for _ in range(4)]
    Y_bins_all = [[] for _ in range(4)]
    I_bins_all = [[] for _ in range(4)]
    W_bins_all = [[] for _ in range(4)]
    C_bins_all = [[] for _ in range(4)]

    window_days = 2.5

    for sim in sims:
        all_t = sim.get("all_t", [])
        all_xs = sim.get("all_xs", [])
        all_L0s = sim.get("all_L0s", [])

        # get receptor concentration, given by L0/L, where L = x_i-x_{i-1} is the cell length, and L0 is the reference length
        all_Rs = [np.asarray(L0_k, float) / np.diff(xs_k) for xs_k, L0_k in zip(all_xs, all_L0s)]

        all_W_raw = _first_key(sim, "all_W", "all_free_W", "all_Wnt")
        all_DL = _first_key(sim, "all_DL", "all_I")
        all_YAP = _first_key(sim, "all_YAP", "all_Y", "all_Y2")
        all_C = sim.get("all_C", None)

        # scale DL and C by receptor concentration to get occupancy
        if all_DL is not None:
            all_DL = [np.asarray(v, float) / R_k for v, R_k in zip(all_DL, all_Rs)]
        if all_C is not None:
            all_C = [np.asarray(v, float) / R_k for v, R_k in zip(all_C, all_Rs)]

        # Compression -> force in nN
        sigma_series = [
            (np.asarray(L0_k, float) - np.diff(xs_k)) * FORCE_SCALE_NN
            for xs_k, L0_k in zip(all_xs, all_L0s)
        ]

        # Free WNT -> concentration in nM
        W_series = [FREE_WNT_SCALE_NM * np.asarray(w, float) for w in all_W_raw] if all_W_raw is not None else None

        # These remain proportions
        I_series = [np.asarray(v, float) for v in all_DL] if all_DL is not None else None
        Y_series = [np.asarray(v, float) for v in all_YAP] if all_YAP is not None else None
        C_series = [np.asarray(v, float) for v in all_C] if all_C is not None else None

        bins_sigma = _rolling_quartile_means(all_t, all_xs, sigma_series, window_days=window_days)
        bins_Y = _rolling_quartile_means(all_t, all_xs, Y_series, window_days=window_days) if Y_series is not None else [np.array([])] * 4
        bins_I = _rolling_quartile_means(all_t, all_xs, I_series, window_days=window_days) if I_series is not None else [np.array([])] * 4
        bins_W = _rolling_quartile_means(all_t, all_xs, W_series, window_days=window_days) if W_series is not None else [np.array([])] * 4
        bins_C = _rolling_quartile_means(all_t, all_xs, C_series, window_days=window_days) if C_series is not None else [np.array([])] * 4

        for j in range(4):
            if bins_sigma[j].size:
                sigma_bins_all[j].extend(bins_sigma[j].tolist())
            if bins_Y[j].size:
                Y_bins_all[j].extend(bins_Y[j].tolist())
            if bins_I[j].size:
                I_bins_all[j].extend(bins_I[j].tolist())
            if bins_W[j].size:
                W_bins_all[j].extend(bins_W[j].tolist())
            if bins_C[j].size:
                C_bins_all[j].extend(bins_C[j].tolist())

    sigma_bins = [np.array([np.nan])] + _finalise_bins(sigma_bins_all) + [np.array([np.nan])]
    Y_bins = [np.array([np.nan])] + _finalise_bins(Y_bins_all) + [np.array([np.nan])]
    I_bins = [np.array([np.nan])] + _finalise_bins(I_bins_all) + [np.array([np.nan])]
    W_bins = [np.array([np.nan])] + _finalise_bins(W_bins_all) + [np.array([np.nan])]
    C_bins = [np.array([np.nan])] + _finalise_bins(C_bins_all) + [np.array([np.nan])]

    horiz_boxplot(axs_prof[0], sigma_bins, FORCE_LABEL, "Compression, $\\sigma$")
    horiz_boxplot(axs_prof[1], Y_bins, FRACTION_LABEL, "Nuclear YAP, $Y$")
    horiz_boxplot(axs_prof[2], I_bins, OCCUPANCY_LABEL, "Bound DKK1, $I_i$")
    horiz_boxplot(axs_prof[3], W_bins, CONCENTRATION_LABEL, "Free WNT, $W$", nbins=3)
    horiz_boxplot(axs_prof[4], C_bins, OCCUPANCY_LABEL, "Bound WNT, $C_i$" + "\n" + r"(active $\beta$-catenin)")

else:
    titles = ["Compression", "Nuclear YAP", "Bound DKK1", "Free WNT", "Bound WNT"]
    xlabels = [FORCE_LABEL, FRACTION_LABEL, OCCUPANCY_LABEL, CONCENTRATION_LABEL, OCCUPANCY_LABEL]

    for ax, title, xlabel in zip(axs_prof, titles, xlabels):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
        ax.set_yticklabels([r"$0$", "", "", "", "", r"$1$"])
        ax.grid(alpha=0.25)

axs_prof[0].set_ylabel(RELATIVE_POSITION_LABEL)
for ax in axs_prof[1:]:
    ax.set_yticklabels([])

# -----------------------------
# Save figure
# -----------------------------
save_dir = "./plots"
os.makedirs(save_dir, exist_ok=True)
outfile = os.path.join(save_dir, "fig6.pdf")
fig.savefig(outfile, dpi=300)
print("Saved:", outfile)