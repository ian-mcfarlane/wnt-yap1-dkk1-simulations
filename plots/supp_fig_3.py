#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Composite 10-panel figure — top-row wider + legend panel, middle/bottom rows unchanged.

Layout:
 top row:  plot (Y vs t)  |  plot (Y vs sigma)  | legend panel
 middle:  plot | plot | kymo (wider) | histo (narrow)
 bottom:  plot | plot | kymo (wider) | histo (narrow)

Middle and bottom rows use width ratios: [1.0, 1.0, 1.7, 0.55]
Top row is created with a separate sub-GridSpec so its columns can be wider.
"""
from dataclasses import dataclass, replace, astuple
import os, glob, pickle, re
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from numpy.polynomial import Polynomial as Poly
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks
import yaml

# ----------------------------
# User settings (edit as needed)
# ----------------------------
BASE_DIR_DOWN = "./outputs_params_refined/mutations/alphaM_low"
BASE_DIR_UP   = "./outputs_params_refined/mutations/alphaM_high"

TARGET = dict(beta=1.5, kappa=1350, a_step=0.0, b_step=0.7, phiD=61.7, Wdiv=0.0098)

KAPPA = TARGET["kappa"]
W_DIV = TARGET["Wdiv"]
A_STEP = TARGET["a_step"]
B_STEP = TARGET["b_step"]
PHI = TARGET["phiD"]

OUTFILE = "./plots/supp_fig_3.pdf"
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

# Plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

FIGSIZE = (8.27, 11.69) # A4 size in inches

LENGTH_SCALE_UM = 600.0
FORCE_SCALE_NN = 49.06 * LENGTH_SCALE_UM

TIME_LABEL = r"Time, $t$ (days)"
POSITION_LABEL = r"Position, $x$ ($\mu\mathrm{m}$)"
Y_TIMELABEL = r"$Y(t)$ (fraction)"
Y_LABEL = r"$Y$ (fraction)"
FORCE_LABEL = r"Compression, $\sigma(t)$ (nN)"

def sigma_to_force(sigma):
    arr = FORCE_SCALE_NN * np.asarray(sigma, float)
    return float(arr) if np.ndim(arr) == 0 else arr

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


def parse_from_name(fn: str):
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

def find_files_for_dir(base_dir: str) -> List[str]:
    patt = os.path.join(
        base_dir,
        f"*kappa{KAPPA:.2e}_phiD{PHI:.1f}_Wdiv{W_DIV:.4f}_a{A_STEP:.2f}_b{B_STEP:.2f}_rep*.pkl"
    )
    print("Looking for files with pattern:", patt)
    return sorted(glob.glob(patt))

def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# ----------------------------
# Kymograph helpers
# ----------------------------
def detect_new_node(old_x, xs_k, ratio_threshold=4.0):
    tree = cKDTree(old_x[:, None])
    dists, _ = tree.query(xs_k[:, None], k=1)
    new_idx = int(np.argmax(dists))
    return xs_k[new_idx], new_idx

def build_segments(all_t, all_xs, all_values, division_positions=None):
    if division_positions is None:
        division_positions = []
    seg_times, seg_xedges, seg_vals = [], [], []
    cur_t, cur_x, cur_v = [all_t[0]], [all_xs[0]], [all_values[0]]
    prev_N = len(all_xs[0]) - 1
    div_idx = 0

    for t_k, xs_k, v_k in zip(all_t[1:], all_xs[1:], all_values[1:]):
        N_k = len(xs_k) - 1
        if N_k != prev_N:
            T = np.array(cur_t); X = np.vstack(cur_x); V = np.vstack(cur_v)
            seg_times.append(T); seg_xedges.append(np.vstack([X, X[-1:]])); seg_vals.append(V)
            old_x = cur_x[-1]; old_v = cur_v[-1]
            if N_k > prev_N:
                if div_idx < len(division_positions):
                    p = float(division_positions[div_idx]); div_idx += 1
                    new_node = p * xs_k[-1]
                    xs_prev_with_new = np.sort(np.concatenate([old_x, [new_node]]))
                    insert_pos = np.searchsorted(old_x, new_node); parent_cell = insert_pos - 1
                    if 0 <= parent_cell < prev_N:
                        v_prev_with_split = np.insert(old_v, parent_cell + 1, old_v[parent_cell])
                    else:
                        xs_prev_with_new = xs_k.copy(); v_prev_with_split = v_k.copy()
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
                        xs_prev_with_new = xs_k.copy(); v_prev_with_split = v_k.copy()
                cur_t = [cur_t[-1], t_k]; cur_x = [xs_prev_with_new, xs_k]; cur_v = [v_prev_with_split, v_k]
            else:
                prev_removed = old_x[:-1]; prev_v = old_v[:-1] if len(old_v) > 1 else old_v
                cur_t = [cur_t[-1], t_k]; cur_x = [prev_removed, xs_k]; cur_v = [prev_v, v_k]
            prev_N = N_k
        else:
            cur_t.append(t_k); cur_x.append(xs_k); cur_v.append(v_k)

    T = np.array(cur_t); X = np.vstack(cur_x); V = np.vstack(cur_v)
    seg_times.append(T); seg_xedges.append(np.vstack([X, X[-1:]])); seg_vals.append(V)
    ok = [i for i, V in enumerate(seg_vals) if V.size > 0 and np.isfinite(V).all()]
    seg_times = [seg_times[i] for i in ok]; seg_xedges = [seg_xedges[i] for i in ok]; seg_vals = [seg_vals[i] for i in ok]
    return seg_times, seg_xedges, seg_vals

def extract_division_events_from_lengths(all_t, num_cells, lengths, division_positions):
    t_ev, x_ev = [], []
    prev_N = int(num_cells[0]) if len(num_cells) else 0
    idx = 0
    for k in range(1, len(all_t)):
        N_k = int(num_cells[k])
        if N_k == prev_N + 1 and idx < len(division_positions):
            t_ev.append(all_t[k]); x_ev.append(float(division_positions[idx]) * float(lengths[k])); idx += 1
        prev_N = N_k
    return np.asarray(t_ev, float), np.asarray(x_ev, float)

# ----------------------------
# Division plotting helpers
# ----------------------------
def plot_division_histogram_over_time(ax, X, xmax, bins=30, label_xaxis=True):
    if X.size:
        counts, edges = np.histogram(X, bins=bins, range=(0.0, xmax))
        centres = 0.5 * (edges[:-1] + edges[1:])
        height  = edges[1] - edges[0]
        ax.barh(centres, counts, height=0.9*height, align='center', color='C0', edgecolor='C0', alpha=0.7)
    else:
        ax.text(0.5, 0.5, "no divisions", ha="center", va="center", transform=ax.transAxes)
    if label_xaxis:
        ax.set_xlabel("Frequency")
    ax.set_ylim(0.0, xmax)
    ax.set_yticks([])
    ax.grid(alpha=0.2, axis="x")

# ----------------------------
# Build per-row multi-cell data
# ----------------------------
def prep_row_for_dir(base_dir: str, phi: float = PHI):
    files = find_files_for_dir(base_dir)
    if not files:
        return None
    rep0 = None
    for f in files:
        md = parse_from_name(f)
        if md and md["rep"] == 0:
            rep0 = f; break
    if rep0 is None:
        for f in files:
            try:
                d = load_pkl(f)
            except Exception:
                continue
            if all(k in d for k in ("all_t", "all_xs", "all_C")):
                rep0 = f; break
    if rep0 is None:
        return None

    d0 = load_pkl(rep0)
    all_t0 = np.asarray(d0["all_t"], float)
    all_x0 = d0["all_xs"]
    all_x0 = [x0 * LENGTH_SCALE_UM for x0 in all_x0] if all_x0 else all_x0
    all_C0 = d0["all_C"]
    divpos0 = d0.get("division_positions", [])
    segs_C  = build_segments(all_t0, all_x0, all_C0, divpos0)

    t_div_all, x_div_all = [], []
    times_all, lengths_all = [], []
    for fp in files:
        try:
            dd = load_pkl(fp)
        except Exception:
            continue
        tt = np.asarray(dd.get("all_t", []), float)
        nn = np.asarray(dd.get("num_cells", []), int)
        ll = np.asarray(dd.get("lengths", []), float)
        if tt.size and ll.size:
            ll = ll * LENGTH_SCALE_UM
            times_all.append(tt); lengths_all.append(ll)
        td, xd = extract_division_events_from_lengths(tt, nn, ll, dd.get("division_positions", []))
        if td.size:
            t_div_all.append(td); x_div_all.append(xd)

    t_div = np.concatenate(t_div_all) if t_div_all else np.array([])
    x_div = np.concatenate(x_div_all) if x_div_all else np.array([])

    return dict(
        dir=base_dir, phi=phi, segs_C=segs_C, t0=all_t0,
        lengths0=np.asarray(d0.get("lengths", []), float),
        t_div=t_div, x_div=x_div, times_all=times_all, lengths_all=lengths_all
    )

# ============================================================================
# Single-cell helpers and model functions
# ============================================================================

# ----------------------------
# Single-cell configuration
# ----------------------------

# Fractional perturbation applied to alpha_M, up and down.
ALPHA_M_VARY = 0.2

# Compression grid on which the steady-state branches are computed.
SIG_GRID = np.linspace(0.0, 6.0e-4, 2001)

# Loading protocol: two smooth rises up to sigma_high, then mirrored back down.
# The plateaus are long compared with tau_M so the cell reaches its attractor
# before the next step; shortening them suppresses the switch entirely.
PROTOCOL_KW = dict(
    centre=200.0,
    rise1_center=80.0, rise1_width=30.0,
    rise2_center=160.0, rise2_width=30.0,
)
N_TEVAL = 3001

# Padding added above/below the extreme branch values when setting y-limits.
Y_PAD = 0.02


# ----------------------------
# Non-dimensional parameter groups
# ----------------------------

def build_nd_groups(p: dict, kappa_non_dim) -> dict:
    T_DAY = 24.0 * 3600.0
    L_dim = p['total_length']
    R0 = p.get('R_total', p.get('R0', 1.0))

    eta_dim = 3.15e3
    kappa_dim = kappa_non_dim / T_DAY * eta_dim

    nd = {}
    nd['DW'] = p['D_W'] * T_DAY / L_dim**2
    nd['DD'] = p['D_D'] * T_DAY / L_dim**2
    nd['beta_on1']  = p['k1_on'] * R0 * T_DAY
    nd['beta_off1'] = p['k1_off'] * T_DAY
    nd['beta_on2']  = p['k2_on'] * R0 * T_DAY
    nd['beta_off2'] = p['k2_off'] * T_DAY
    nd['beta_off3'] = p['k_out0'] * T_DAY
    nd['deltaW']    = p['delta_W'] * T_DAY
    nd['deltaD']    = p['delta_D'] * T_DAY
    nd['phiD']      = p['r_dy'] * T_DAY / R0
    nd['m_dy']      = p['m_dy']
    nd['R0']        = R0

    nd['xi_D']    = p['xi_D']
    nd['K_sigma'] = p['K_sigma_dim'] / (kappa_dim * L_dim)
    nd['A_sigma'] = p['A_sigma']
    nd['alpha_M'] = p['alpha_M']
    nd['Y_tot']   = p['Y_tot']
    nd['xi_M']    = p['xi_M']
    nd['n_Y']     = p['n_Y']

    return nd


# read parameters from YAML file
with open("parameters/base_params_yap_bistab_refined.yaml", "r") as f:
    base_params = yaml.safe_load(f)
base_params["total_length"] = 600.0

nd_params = build_nd_groups(base_params, kappa_non_dim=1350)


@dataclass(frozen=True)
class ParamsIE:
    k0: float = nd_params['beta_off3']
    K_sigma: float = nd_params['K_sigma']
    A_sigma: float = nd_params['A_sigma']
    alpha_M_in: float = nd_params['alpha_M']
    hill_K: float = nd_params['xi_M']
    hill_n: float = nd_params['n_Y']
    tau_M: float = 7.0        # ECM memory relaxation time, in days


P = ParamsIE()


def alpha_M_cases(P_base=None, vary=ALPHA_M_VARY):
    """
    Baseline parameter set together with alpha_M perturbed down and up by
    `vary` (a fraction). Returns (P_base, P_down, P_up).
    """
    if P_base is None:
        P_base = ParamsIE()

    P_down = replace(P_base, alpha_M_in=P_base.alpha_M_in * (1.0 - vary))
    P_up   = replace(P_base, alpha_M_in=P_base.alpha_M_in * (1.0 + vary))

    return P_base, P_down, P_up


# ----------------------------
# Transport, Hill gate, derivatives
# ----------------------------

def S_sigma(s, P): return s/(P.K_sigma + s + 1e-16)
def k_in(sigma, M, P): return P.k0 * (1.0 + P.A_sigma*S_sigma(sigma, P) + P.alpha_M_in*M)


def Ystar_IE(sigma, M, P):
    A = k_in(sigma, M, P); B = P.k0
    return A/(A+B+1e-16)


def dYdM_IE(sigma, M, P):
    A = k_in(sigma, M, P); B = P.k0; Ap = P.k0 * P.alpha_M_in
    return (Ap*B)/((A+B+1e-16)**2)


def hill_Y(Y, P):
    y = np.clip(np.asarray(Y, float), 1e-15, 1-1e-15)
    t = (y / P.hill_K)**P.hill_n
    return t/(1.0 + t)


def dhill_dy(Y, P):
    y = np.clip(np.asarray(Y, float), 1e-15, 1-1e-15)
    t = (y / P.hill_K)**P.hill_n
    return (P.hill_n / P.hill_K) * (y / P.hill_K)**(P.hill_n - 1) / (1.0 + t)**2


def dF_dM(sigma, M, P):
    return dhill_dy(Ystar_IE(sigma, M, P), P) * dYdM_IE(sigma, M, P) - 1.0


# ----------------------------
# Fixed points via the Y-polynomial
# ----------------------------

def _poly_coeffs_Y(sigma, P):
    S = S_sigma(sigma, P); Ahat = P.A_sigma * S
    n = int(round(P.hill_n)); Kn = P.hill_K ** n
    c = np.zeros(n+2)
    c[0] = -(1.0 + Ahat) * Kn
    c[1] = (2.0 + Ahat) * Kn
    c[n]   = -(1.0 + Ahat + P.alpha_M_in)
    c[n+1] =  (2.0 + Ahat + P.alpha_M_in)
    return c


def roots_at_sigma_poly(sigma, P: ParamsIE, y_tol=1e-10):
    coeffs = _poly_coeffs_Y(sigma, P)
    Y_roots = Poly(coeffs).roots()
    Y = np.real(Y_roots[np.abs(np.imag(Y_roots)) < y_tol])
    Y = Y[(Y > 0.0) & (Y < 1.0)]
    if Y.size == 0:
        return np.array([]), np.array([], dtype=bool)
    n = int(round(P.hill_n)); Kn = P.hill_K ** n
    M = (Y**n) / (Kn + Y**n)
    stab = np.array([dF_dM(sigma, m, P) < 0.0 for m in M], dtype=bool)
    order = np.argsort(M)
    return M[order], stab[order]


def track_branches_poly(sigmas, P):
    branches, prev = [], []
    for s in sigmas:
        Ms, st = roots_at_sigma_poly(s, P)
        unmatched_prev, unmatched_curr = set(range(len(prev))), set(range(len(Ms)))
        cand = [(abs(Ms[j]-prev[i][1]), i, j) for i in range(len(prev)) for j in range(len(Ms))]
        cand.sort(key=lambda x: x[0])
        for _, i, j in cand:
            if i in unmatched_prev and j in unmatched_curr and abs(Ms[j]-prev[i][1]) < 0.2:
                bi, _ = prev[i]
                branches[bi]['sigma'].append(s); branches[bi]['M'].append(Ms[j]); branches[bi]['stab'].append(st[j])
                unmatched_prev.remove(i); unmatched_curr.remove(j)
        for i in unmatched_prev:
            bi, _ = prev[i]
            branches[bi]['sigma'].append(np.nan); branches[bi]['M'].append(np.nan); branches[bi]['stab'].append(True)
        for j in unmatched_curr:
            branches.append({'sigma': [s], 'M': [Ms[j]], 'stab': [st[j]]})
        prev = [(bi, branches[bi]['M'][-1]) for bi in range(len(branches))
                if not np.isnan(branches[bi]['M'][-1])]
    return branches


# Branch tracking over a 2001-point grid is the slowest step here and is needed
# three times per case (y-limits, the sigma-Y panel, diagnostics), so cache it.
_BRANCH_CACHE = {}


def branches_for(sig_grid, P):
    key = (len(sig_grid), float(sig_grid[0]), float(sig_grid[-1]), astuple(P))
    if key not in _BRANCH_CACHE:
        _BRANCH_CACHE[key] = track_branches_poly(sig_grid, P)
    return _BRANCH_CACHE[key]


def branch_curves(sig_grid, P):
    """Yield (sigma, Y, stable) arrays for each tracked branch, NaNs dropped."""
    for br in branches_for(sig_grid, P):
        Sg = np.asarray(br['sigma'], float)
        Mg = np.asarray(br['M'], float)
        st = np.asarray(br['stab'], dtype=bool)

        valid = (~np.isnan(Sg)) & (~np.isnan(Mg))
        if not np.any(valid):
            continue

        yield Sg[valid], Ystar_IE(Sg[valid], Mg[valid], P), st[valid]


# ----------------------------
# Protocol builder
# ----------------------------

def Lambda_w(t, t0, w):
    scale = 8.0 / float(w) if w > 0 else 1e9
    return 1.0 / (1.0 + np.exp(-scale * (t - t0)))


def sigma_symmetric_two_rises(levels, centre=100.0,
                              rise1_center=40.0, rise1_width=8.0,
                              rise2_center=80.0, rise2_width=8.0):
    """
    sigma(t) rises L0 -> L1 -> L2 over [0, centre] and mirrors back down over
    [centre, 2*centre], so the protocol is symmetric about t = centre.
    """
    L0, L1, L2 = float(levels[0]), float(levels[1]), float(levels[2])
    C = float(centre)

    def _sigma_single(tt):
        if tt >= C: tt = 2.0 * C - tt
        if tt <= 0.0: return L0
        s1 = Lambda_w(tt, rise1_center, rise1_width)
        s2 = Lambda_w(tt, rise2_center, rise2_width)
        v_pre = (1.0 - s1) * L0 + s1 * L1
        return (1.0 - s2) * v_pre + s2 * L2

    def sigma_t(t_in):
        t = np.asarray(t_in, dtype=float)
        if t.shape == ():
            return float(_sigma_single(float(t)))
        out = np.empty_like(t, dtype=float)
        for i, val in enumerate(t.flat):
            out.flat[i] = _sigma_single(float(val))
        return out

    return sigma_t, 2.0 * C


def build_protocol(folds_base, sigma_high=5.0e-4, **protocol_kw):
    """
    Protocol levels placed relative to the baseline fold points: below the
    lower fold, inside the bistable window, then well above the upper fold.
    Returns (sigma_func, T_total, levels).
    """
    kw = dict(PROTOCOL_KW)
    kw.update(protocol_kw)

    Lc, Rc = folds_base
    left_most  = float(Lc) if not np.isnan(Lc) else 6e-5
    right_most = float(Rc) if not np.isnan(Rc) else 2.2e-4

    sigma_low  = 0.5 * left_most
    bist_level = 0.5 * (left_most + right_most)
    levels = [sigma_low, bist_level, sigma_high, bist_level, sigma_low]

    sigma_func, T_total = sigma_symmetric_two_rises(levels, **kw)
    return sigma_func, T_total, levels


# ----------------------------
# Single-cell RHS and time integration
# ----------------------------

def rhs_IE(t, y, sigma_func, P):
    """
    Relaxation of the ECM memory variable M towards the Hill gate of Y,
    on the timescale tau_M.
    """
    M = y[0]
    sigma = sigma_func(t) if callable(sigma_func) else float(sigma_func)
    Y = Ystar_IE(sigma, M, P)
    return [(hill_Y(Y, P) - M) / P.tau_M]


def initial_M(sigma0, P):
    """
    Steady state of M at compression sigma0: the lowest stable fixed point,
    so the cell starts on the lower branch rather than off it.
    """
    Ms, stab = roots_at_sigma_poly(sigma0, P)
    if Ms.size == 0:
        return 0.0
    if np.any(stab):
        return float(Ms[stab][0])
    return float(Ms[0])


def simulate_case(sigma_func, t_eval, P, M0=None, method="RK45"):
    """
    Integrate the single-cell model over t_eval and return the trajectory
    together with the compression and YAP traces.
    """
    if M0 is None:
        M0 = initial_M(float(sigma_func(t_eval[0])), P)

    sol = solve_ivp(rhs_IE, (t_eval[0], t_eval[-1]), [M0], t_eval=t_eval,
                    args=(sigma_func, P), rtol=1e-8, atol=1e-10, method=method)

    t = sol.t
    M = np.clip(sol.y[0], 0.0, 1.0)
    S = sigma_func(t)
    Y = Ystar_IE(S, M, P)

    return dict(t=t, M=M, S=S, Y=Y, P=P, M0=M0)


# ----------------------------
# Shared y-limits
# ----------------------------

def common_y_limits(sig_grid, P_list, sims=(), pad=Y_PAD, floor=None):
    """
    Y-range spanning every branch and every trajectory supplied, so the same
    limits can be used on all six single-cell panels without clipping the
    upper stable branch.
    """
    lo, hi = np.inf, -np.inf

    for P_case in P_list:
        for _, Yg, _ in branch_curves(sig_grid, P_case):
            lo = min(lo, float(np.nanmin(Yg)))
            hi = max(hi, float(np.nanmax(Yg)))

    for sim in sims:
        Y = np.asarray(sim["Y"], float)
        if Y.size:
            lo = min(lo, float(np.nanmin(Y)))
            hi = max(hi, float(np.nanmax(Y)))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.45, 1.0

    lo, hi = lo - pad, min(1.0, hi + pad)
    if floor is not None:
        lo = max(lo, floor)

    return lo, hi


# ----------------------------
# Arrow placement helpers (unchanged)
# ----------------------------

def two_pass_arrow_fractions_from_timecourse(t, y):
    """
    Return two fractions: one from the strongest upward transition in Y(t),
    and one from the strongest downward transition in Y(t).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]

    if t.size < 5:
        return (0.25, 0.75)

    dydt = np.gradient(y, t)

    # Light smoothing.
    win = max(5, int(0.01 * t.size) | 1)
    kernel = np.ones(win) / win
    dydt_s = np.convolve(dydt, kernel, mode="same")

    i_up = int(np.argmax(dydt_s))
    i_dn = int(np.argmin(dydt_s))

    return (i_up / float(t.size - 1), i_dn / float(t.size - 1))


def auto_arrow_fractions_from_timecourse(
    t,
    y,
    n_arrows=3,
    min_separation_frac=0.10,
    prominence_quantile=0.80,
):
    """
    Choose arrow locations from the steepest parts of Y(t).

    Returns fractions in [0, 1] suitable for use on the corresponding
    trajectory array of the same length.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]

    if t.size < 5:
        return (0.20, 0.50, 0.80)[:n_arrows]

    dydt = np.gradient(y, t)

    # Light smoothing to suppress numerical noise while preserving main transitions.
    win = max(5, int(0.01 * t.size) | 1)  # odd window length
    kernel = np.ones(win) / win
    dydt_s = np.convolve(dydt, kernel, mode="same")
    score = np.abs(dydt_s)

    # Prefer prominent peaks in |dY/dt|.
    prom = np.quantile(score, prominence_quantile)
    peaks, props = find_peaks(score, prominence=prom)

    if peaks.size == 0:
        peaks = np.array([int(np.argmax(score))])

    # Rank peaks by steepness.
    order = np.argsort(score[peaks])[::-1]
    peaks = peaks[order]

    # Enforce spacing between selected arrows.
    min_sep = max(1, int(min_separation_frac * t.size))
    selected = []
    for p in peaks:
        if all(abs(p - q) >= min_sep for q in selected):
            selected.append(int(p))
        if len(selected) >= n_arrows:
            break

    # Fallback if not enough peaks were found.
    if len(selected) < n_arrows:
        extra = np.linspace(0.15, 0.85, n_arrows)
        fallback = [int(round(frac * (t.size - 1))) for frac in extra]
        for p in fallback:
            if all(abs(p - q) >= min_sep for q in selected):
                selected.append(p)
            if len(selected) >= n_arrows:
                break

    selected = np.array(sorted(set(selected)), dtype=int)
    if selected.size == 0:
        return (0.20, 0.50, 0.80)[:n_arrows]

    # Convert selected indices to fractions of the trajectory array.
    fracs = selected / float(t.size - 1)

    # Keep only up to n_arrows.
    fracs = fracs[:n_arrows]

    # If we ended up with fewer than requested, pad with evenly spaced values.
    if fracs.size < n_arrows:
        pad = np.linspace(0.2, 0.8, n_arrows)
        fracs = np.unique(np.r_[fracs, pad])[:n_arrows]

    return tuple(fracs)


# def add_external_time_arrows(
#     ax, x, y,
#     fractions=(0.14, 0.38, 0.68),
#     colour='forestgreen',
#     pad_pts=10,
#     arrow_len_pts=25,
#     lw=1.0,
#     mutation_scale=14,
#     pair_mode=False,
# ):
#     """
#     Draw short arrows parallel to the trajectory, offset in screen space.

#     If pair_mode=True and exactly two fractions are supplied, place the two
#     arrows at their own trajectory locations but offset them using a shared
#     normal so one sits above and one below a retraced branch.
#     """
#     x = np.asarray(x, float)
#     y = np.asarray(y, float)

#     finite = np.isfinite(x) & np.isfinite(y)
#     x = x[finite]
#     y = y[finite]
#     if x.size < 3:
#         return

#     pts = ax.transData.transform(np.column_stack([x, y]))
#     dx = np.gradient(pts[:, 0])
#     dy = np.gradient(pts[:, 1])
#     inv = ax.transData.inverted()

#     def _unit(vx, vy):
#         n = np.hypot(vx, vy)
#         if n < 1e-9:
#             return None
#         return np.array([vx / n, vy / n], dtype=float)

#     def _draw_arrow_from_display_point(p, tvec):
#         p0 = p - 0.5 * arrow_len_pts * tvec
#         p1 = p + 0.5 * arrow_len_pts * tvec
#         x0, y0 = inv.transform(p0)
#         x1, y1 = inv.transform(p1)

#         arrow = FancyArrowPatch(
#             (x0, y0), (x1, y1),
#             transform=ax.transData,
#             arrowstyle='-|>',
#             mutation_scale=mutation_scale,
#             lw=lw,
#             color=colour,
#             clip_on=False,
#             zorder=8,
#         )
#         ax.add_patch(arrow)

#     idxs = [int(np.clip(round(f * (x.size - 1)), 1, x.size - 2)) for f in fractions]

#     # Special handling for retracing trajectories.
#     if pair_mode and len(idxs) == 2:
#         i1, i2 = idxs

#         t1 = _unit(dx[i1], dy[i1])
#         t2 = _unit(dx[i2], dy[i2])
#         if t1 is None or t2 is None:
#             return

#         # Local normals.
#         n1 = np.array([-t1[1], t1[0]], dtype=float)
#         n2 = np.array([-t2[1], t2[0]], dtype=float)

#         # Align normals so they represent the same geometric side.
#         if np.dot(n1, n2) < 0:
#             n2 = -n2

#         nref = n1 + n2
#         nref_norm = np.hypot(nref[0], nref[1])
#         if nref_norm < 1e-9:
#             nref = n1
#         else:
#             nref /= nref_norm

#         # Slightly larger offset for clarity in the retracing case.
#         pad_pair = 1.25 * pad_pts

#         p1 = pts[i1] - pad_pair * nref
#         p2 = pts[i2] + pad_pair * nref

#         _draw_arrow_from_display_point(p1, t1)
#         _draw_arrow_from_display_point(p2, t2)
#         return

#     # Default behaviour for the other panels.
#     cx = np.mean(pts[:, 0])
#     cy = np.mean(pts[:, 1])

#     for i in idxs:
#         tvec = _unit(dx[i], dy[i])
#         if tvec is None:
#             continue

#         nx, ny = -tvec[1], tvec[0]
#         side = np.sign((pts[i, 0] - cx) * nx + (pts[i, 1] - cy) * ny)
#         if side == 0:
#             side = 1.0

#         p = pts[i] + pad_pts * np.array([side * nx, side * ny])
#         _draw_arrow_from_display_point(p, tvec)

def add_external_time_arrows(
    ax, x, y,
    fractions=(0.14, 0.38, 0.68),
    colour='forestgreen',
    pad_pts=10,
    arrow_len_pts=25,
    lw=1.0,
    mutation_scale=14,
    pair_mode=False,      # kept for backwards compatibility; no longer needed
    side='right',
):
    """
    Draw short arrows parallel to the trajectory, offset in screen space.

    Each arrow is offset to a fixed side of the local direction of travel
    ('right' or 'left'). Legs travelling in opposite directions therefore
    get arrows on opposite sides: e.g. with side='right', rightward-moving
    segments get arrows below the curve and leftward-moving ones above.
    This handles loops, figure-eights and retraced branches consistently.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        return

    pts = ax.transData.transform(np.column_stack([x, y]))
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    inv = ax.transData.inverted()

    sgn = 1.0 if side == 'right' else -1.0

    idxs = [int(np.clip(round(f * (x.size - 1)), 1, x.size - 2)) for f in fractions]

    for i in idxs:
        n = np.hypot(dx[i], dy[i])
        if n < 1e-9:
            continue
        tvec = np.array([dx[i], dy[i]]) / n
        # Right-hand normal of the direction of travel (display coords, y up).
        nvec = sgn * np.array([tvec[1], -tvec[0]])

        p = pts[i] + pad_pts * nvec
        p0 = p - 0.5 * arrow_len_pts * tvec
        p1 = p + 0.5 * arrow_len_pts * tvec
        (x0, y0), (x1, y1) = inv.transform(p0), inv.transform(p1)

        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            transform=ax.transData,
            arrowstyle='-|>',
            mutation_scale=mutation_scale,
            lw=lw,
            color=colour,
            clip_on=False,
            zorder=8,
        ))


# ----------------------------
# Single-cell panels
# ----------------------------

def plot_singlecell_timecourse(ax, sim, folds=(np.nan, np.nan),
                               show_ylabel=True, Smin_Smax=None,
                               ylim=(0.45, 1.0), label_folds=False):
    """
    Y(t) with the imposed compression shown as a shaded band rescaled onto the
    Y axis. Dashed horizontal lines mark the fold compressions on that band.
    `sim` is the dict returned by simulate_case.
    """
    t, S, Y = sim["t"], sim["S"], sim["Y"]
    ylo, yhi = ylim

    if Smin_Smax is not None:
        Smin, Smax = float(Smin_Smax[0]), float(Smin_Smax[1])
    else:
        Smin, Smax = float(np.min(S)), float(np.max(S))
    if Smax - Smin < 1e-16:
        Smin, Smax = 0.0, 1.0

    def to_axis(s):
        return ylo + (yhi - ylo) * (s - Smin) / (Smax - Smin)

    ax.fill_between(t, ylo, to_axis(S), color='C4', alpha=0.18)
    ax.plot(t, Y, color='C0', lw=2.0, label=r'$Y$')

    fold_labels = (r"$\sigma^{+}_{\rm lo}$", r"$\sigma^{+}_{\rm hi}$")
    for val, lab in zip(folds, fold_labels):
        if np.isnan(val) or not (Smin < val < Smax):
            continue
        y_line = to_axis(val)
        ax.hlines(y_line, t[0], t[-1], colors='0.5', linestyles='--', linewidth=0.8)
        if label_folds:
            ax.text(0.02 * t[-1], y_line + 0.005, lab, fontsize=8, va='bottom')

    if show_ylabel:
        ax.set_ylabel(Y_TIMELABEL)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(ylo, yhi)
    ax.grid(alpha=0.25)

    return t, S, Y


def plot_branches_and_trajectory(ax, sig_grid, sim, P: ParamsIE,
                                 folds=(np.nan, np.nan),
                                 arrow_fractions=None,
                                 pair_mode=False,
                                 ylim=(0.45, 1.0),
                                 label_folds=False):
    """
    Steady-state branches for this parameter set, with the trajectory from
    `sim` overlaid. The x-axis is converted to force via sigma_to_force.
    """
    S_traj = sigma_to_force(sim["S"])
    Y_traj = np.asarray(sim["Y"], float)
    ylo, yhi = ylim

    for Sg, Yg, st in branch_curves(sig_grid, P):
        Sg_force = sigma_to_force(Sg)
        ax.plot(Sg_force[st],  Yg[st],  color='k', lw=1.6)
        ax.plot(Sg_force[~st], Yg[~st], color='k', lw=1.6, ls='--')

    ax.plot(S_traj, Y_traj, color='C0', lw=2.2, label='Trajectory')
    ax.plot(S_traj[0],  Y_traj[0],  's', color='k', ms=6, label='Start')
    ax.plot(S_traj[-1], Y_traj[-1], 'o', color='purple', ms=4, label='End', zorder=5)

    smin, smax = float(sig_grid[0]), float(sig_grid[-1])
    fold_labels = (r"$\sigma^{+}_{\rm lo}$", r"$\sigma^{+}_{\rm hi}$")
    for val, lab in zip(folds, fold_labels):
        if np.isnan(val) or not (smin < val < smax):
            continue
        ax.axvline(sigma_to_force(val), color='0.5', linestyle='--', lw=0.9)
        if label_folds:
            ax.text(sigma_to_force(val), ylo + 0.02 * (yhi - ylo), lab,
                    rotation=90, va='bottom', ha='right', fontsize=8)

    ax.set_xlim(0.0, sigma_to_force(sig_grid[-1]))
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel(Y_LABEL)
    ax.grid(alpha=0.25)

    if arrow_fractions is None:
        arrow_fractions = (0.12, 0.34, 0.58)

    add_external_time_arrows(ax, S_traj, Y_traj,
                             fractions=arrow_fractions,
                             pair_mode=pair_mode)


# ----------------------------
# Utility: find bistable window (folds) on grid
# ----------------------------

def find_bistable_window(sig_grid, P):
    def n_stable(s):
        Ms, st = roots_at_sigma_poly(s, P)
        return int(np.sum(st)) if Ms.size else 0

    counts = np.array([n_stable(s) for s in sig_grid])
    bi = (counts == 2)
    if not np.any(bi):
        return np.nan, np.nan

    idx = np.flatnonzero(bi)
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[cuts + 1]]
    stops  = np.r_[idx[cuts], idx[-1]]

    spans = [(sig_grid[s], sig_grid[t]) for s, t in zip(starts, stops)]
    L, R = max(spans, key=lambda ab: ab[1] - ab[0])

    li = np.searchsorted(sig_grid, L)
    if li > 0 and counts[li-1] != 2:
        L = 0.5*(sig_grid[li-1] + sig_grid[li])
    else:
        L = sig_grid[0]

    ri = np.searchsorted(sig_grid, R)
    if ri < len(sig_grid)-1 and counts[min(ri+1, len(sig_grid)-1)] != 2:
        R = 0.5*(sig_grid[ri] + sig_grid[min(ri+1, len(sig_grid)-1)])
    else:
        R = sig_grid[-1]

    return L, R

# ----------------------------
# Main composition routine
# ----------------------------
def main():
    # prepare multi-cell rows
    row_down = prep_row_for_dir(BASE_DIR_DOWN)
    row_up   = prep_row_for_dir(BASE_DIR_UP)

    # parameter sets baseline / down / up
    P_base, P_down, P_up = alpha_M_cases(vary=ALPHA_M_VARY)

    print(f"alpha_M: {P_down.alpha_M_in:.4g} (down) | "
          f"{P_base.alpha_M_in:.4g} (base) | {P_up.alpha_M_in:.4g} (up)")

    sig_grid = SIG_GRID
    global Smin_global, Smax_global, folds_base
    Smin_global, Smax_global = sig_grid[0], sig_grid[-1]

    folds_base = find_bistable_window(sig_grid, P_base)
    folds_down = find_bistable_window(sig_grid, P_down)
    folds_up   = find_bistable_window(sig_grid, P_up)

    for tag, fd in (("base", folds_base), ("down", folds_down), ("up", folds_up)):
        if np.isnan(fd[0]):
            print(f"  folds [{tag}]: none on this grid (monostable)")
        else:
            print(f"  folds [{tag}]: [{fd[0]:.3e}, {fd[1]:.3e}]")

    # define protocol levels based on baseline folds (fallbacks kept)
    sigma_protocol, T_total, levels = build_protocol(folds_base, sigma_high=5.0e-4)
    t_eval = np.linspace(0.0, T_total, N_TEVAL)

    # compute single-cell trajectories for each Pcase
    sim_base = simulate_case(sigma_protocol, t_eval, P_base)
    sim_down = simulate_case(sigma_protocol, t_eval, P_down)
    sim_up   = simulate_case(sigma_protocol, t_eval, P_up)

    # Shared y-limits across all six single-cell panels.
    YLIM = common_y_limits(sig_grid,
                           [P_base, P_down, P_up],
                           sims=[sim_base, sim_down, sim_up])
    print(f"  single-cell y-limits: [{YLIM[0]:.3f}, {YLIM[1]:.3f}]")

    # determine a common colourscale for the two kymographs (E and I) if both exist
    vmins, vmaxs = [], []
    for r in (row_down, row_up):
        if r is None: continue
        seg_vals = r['segs_C'][2]
        for V in seg_vals:
            if V.size:
                vmins.append(float(np.nanmin(V))); vmaxs.append(float(np.nanmax(V)))
    if vmins and vmaxs:
        global_vmin = min(vmins); global_vmax = max(vmaxs)
    else:
        global_vmin, global_vmax = 0.0, 1.0
    global_norm = Normalize(vmin=global_vmin, vmax=global_vmax)

    # Create top-level GridSpec: three rows; widths are not used directly because
    # top row gets its own sub-GridSpec and middle/bottom rows get theirs.
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    outer = GridSpec(nrows=3, ncols=1, height_ratios=[1.0, 1.0, 1.0], figure=fig, hspace=0.55)

    # ---- Top row: separate sub-GridSpec (three columns: plot, plot, legend)
    # Make top two plot columns wider than the corresponding columns below.
    # Choose top width ratios so the top plots are noticeably wider.
    top_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                     width_ratios=[1.6, 1.6, 0.6], wspace=0.5)
    axA = fig.add_subplot(top_gs[0, 0])        # Y vs t
    axB = fig.add_subplot(top_gs[0, 1])        # sigma-Y
    axLegend = fig.add_subplot(top_gs[0, 2])   # legend panel (right)
    axLegend.axis('off')

    # ---- Middle row: 1 x 4 sub-GridSpec with required ratios [1.0,1.0,1.7,0.55]
    mid_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                     width_ratios=[1.0, 1.0, 1.7, 0.55], wspace=0.60)
    axC = fig.add_subplot(mid_gs[0, 0])
    axD = fig.add_subplot(mid_gs[0, 1])
    axE = fig.add_subplot(mid_gs[0, 2])
    axF = fig.add_subplot(mid_gs[0, 3])

    # ---- Bottom row: 1 x 4 sub-GridSpec with same ratios as middle
    bot_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2],
                                     width_ratios=[1.0, 1.0, 1.7, 0.55], wspace=0.60)
    axG = fig.add_subplot(bot_gs[0, 0])
    axH = fig.add_subplot(bot_gs[0, 1])
    axI = fig.add_subplot(bot_gs[0, 2], sharey=axE)
    axJ = fig.add_subplot(bot_gs[0, 3])

    # Place left-side row labels (single label per logical row). Use figure coords computed from axes.
    left_margin_x = 0.03
    # left_labels = [("Baseline", axA), (r"$\alpha_{\mathrm{M}} \downarrow$: Weak ECM feedback", axC),
    #                (r"$\alpha_{\mathrm{M}} \uparrow$: Strong ECM feedback", axG)]
    left_labels = [("Baseline", axA), ("Weak ECM feedback", axC),
                   ("Strong ECM feedback", axG)]
    
    for txt, ref_ax in left_labels:
        pos = ref_ax.get_position()
        y_center = 0.5 * (pos.y0 + pos.y1)
        fig.text(left_margin_x, y_center, txt, ha='center', va='center', fontsize=11, rotation=90)

    # --- Top-left: timecourse (baseline) ---
    tA, SA, YA = plot_singlecell_timecourse(axA, sim_base, folds=folds_base,
                                            show_ylabel=True,
                                            Smin_Smax=(Smin_global, Smax_global),
                                            ylim=YLIM, label_folds=True)
    axA.set_title("YAP timecourse (single cell)")
    axA.set_xlabel(TIME_LABEL)

    arrow_fracs_A = auto_arrow_fractions_from_timecourse(tA, YA, n_arrows=3)

    # --- Top-middle: sigma-Y trajectory (baseline) ---
    plot_branches_and_trajectory(axB, sig_grid, sim_base, P_base,
                                 folds=folds_base,
                                 arrow_fractions=arrow_fracs_A,
                                 ylim=YLIM, label_folds=True)

    axB.set_title(r"$\sigma$-$Y$ trajectory")
    axB.set_xticks(sigma_to_force(np.array([0.0, 2.0e-4, 4.0e-4, 6.0e-4])))
    axB.set_xlabel(FORCE_LABEL)

    # Build combined legend in the legend panel (axLegend). Use default legend font size so it looks standard.
    proxy_Y = Line2D([0], [0], color='C0', lw=2.0, label=r'$Y(t)$, $Y(\sigma)$')
    proxy_sigma = Patch(facecolor='C4', alpha=0.18, label=r'$\sigma(t)$')
    proxy_fold = Line2D([0], [0], color='0.5', lw=0.9, linestyle='--', label='fold points')
    proxy_start = Line2D([0], [0], color='k', marker='s', linestyle='none', ms=6, label='Start')
    proxy_end = Line2D([0], [0], color='purple', marker='o', linestyle='none', ms=4, label='End')
    proxy_stable = Line2D([0], [0], color='k', lw=1.6, label='stable')
    proxy_unstable = Line2D([0], [0], color='k', lw=1.6, linestyle='--', label='unstable')

    legend_handles = [proxy_Y, proxy_sigma, proxy_fold,
                      proxy_start, proxy_end, proxy_stable, proxy_unstable]
    legend_labels = [h.get_label() for h in legend_handles]
    axLegend.legend(handles=legend_handles, labels=legend_labels, loc='center', frameon=True,
                    ncol=1, fontsize=plt.rcParams['legend.fontsize'])
    axLegend.axis('off')

    # --- Middle row (alphaM down) ---
    tC, SC, YC = plot_singlecell_timecourse(axC, sim_down, folds=folds_down,
                                            show_ylabel=True,
                                            Smin_Smax=(Smin_global, Smax_global),
                                            ylim=YLIM)
    
    # arrow_fracs_C = two_pass_arrow_fractions_from_timecourse(tC, YC)
    arrow_fracs_C = auto_arrow_fractions_from_timecourse(tC, YC, n_arrows=4)
    axC.set_title("YAP\ntimecourse")
    axC.set_xlabel(TIME_LABEL)

    plot_branches_and_trajectory(axD, sig_grid, sim_down, P_down,
                                 folds=folds_down,
                                 arrow_fractions=arrow_fracs_C,
                                 pair_mode=False,
                                 ylim=YLIM)

    axD.set_title(r"$\sigma$-$Y$" + "\ntrajectory")
    axD.set_xlabel(FORCE_LABEL)

    # Kymograph for middle row (E)
    last_pcm_E = None
    axE_ylim_max = 1.0
    if row_down is not None:
        for Tseg, Xedges, Vseg in zip(*row_down['segs_C']):
            Ni1 = Xedges.shape[1]
            Tedges = np.vstack([Tseg[:, None]*np.ones((1, Ni1)), Tseg[-1]*np.ones((1, Ni1))])
            last_pcm_E = axE.pcolormesh(Tedges, Xedges, Vseg, cmap="inferno",
                                        shading="flat", norm=global_norm,
                                        rasterized=True, antialiased=False)
        axE.set_title("Bound WNT\n"+r"(active $\beta$-catenin)")
        axE.set_ylabel(POSITION_LABEL)
        axE.set_xlabel(TIME_LABEL)
        axE.set_xlim(0.0, float(row_down['t0'][-1]) if row_down['t0'].size else t_eval[-1])
        axE.set_ylim(0.0, np.max([L.max() for L in row_down['lengths_all']]) * 1.01 if row_down['lengths_all'] else 1.0)
        axE_ylim_max = axE.get_ylim()[1]
        yticks = np.arange(0.0, axE_ylim_max + 0.1 * LENGTH_SCALE_UM, 0.2 * LENGTH_SCALE_UM)
        axE.set_yticks(yticks)
        axE.grid(alpha=0.2)
    else:
        axE.text(0.5, 0.5, "No multi-cell data (down).", ha="center", va="center", transform=axE.transAxes)
        
    # Narrow 1D histogram in F
    axF.set_title("Division\nfrequency")
    if row_down is not None:
        X_div = row_down['x_div'] #* LENGTH_SCALE_UM
        xmax = axE.get_ylim()[1] if axE.get_ylim() else 1.0
        plot_division_histogram_over_time(axF, X_div, xmax=xmax, bins=30, label_xaxis=False)
    else:
        axF.text(0.5, 0.5, "No multi-cell data (down).", ha="center", va="center", transform=axF.transAxes)
    axF.set_xlabel("Frequency")

    # --- Bottom row (alphaM up) ---
    tG, SG, YG = plot_singlecell_timecourse(axG, sim_up, folds=folds_up,
                                            show_ylabel=True,
                                            Smin_Smax=(Smin_global, Smax_global),
                                            ylim=YLIM)
    
    arrow_fracs_G = auto_arrow_fractions_from_timecourse(tG, YG, n_arrows=3)

    axG.set_title("YAP\ntimecourse")
    axH.set_title(r"$\sigma$-$Y$" + "\ntrajectory")

    plot_branches_and_trajectory(axH, sig_grid, sim_up, P_up,
                                 folds=folds_up,
                                 arrow_fractions=arrow_fracs_G,
                                 ylim=YLIM)

    last_pcm_I = None
    axI.set_title("Bound WNT\n" + r"(active $\beta$-catenin)")
    if row_up is not None:
        for Tseg, Xedges, Vseg in zip(*row_up['segs_C']):
            Xedges_um = Xedges
            Ni1 = Xedges_um.shape[1]
            Tedges = np.vstack([Tseg[:, None] * np.ones((1, Ni1)), Tseg[-1] * np.ones((1, Ni1))])
            last_pcm_I = axI.pcolormesh(
                Tedges, Xedges_um, Vseg, cmap="inferno",
                shading="flat", norm=global_norm,
                rasterized=True, antialiased=False
            )
        axI.set_xlim(0.0, float(row_up['t0'][-1]) if row_up['t0'].size else t_eval[-1])
        axI.set_ylabel(POSITION_LABEL)
        axI.set_xlabel(TIME_LABEL)
        axI.set_ylim(
            0.0,
            np.max([L.max() for L in row_up['lengths_all']] + [axE_ylim_max]) * 1.05
            if row_up['lengths_all'] else 1.0
        )
        axI.grid(alpha=0.2)
    else:
        axI.text(0.5, 0.5, "No multi-cell data (up).", ha="center", va="center", transform=axI.transAxes)
    
    # 1D histogram J (narrow)
    axJ.set_title("Division\nfrequency")
    if row_up is not None:
        X_div_up = row_up['x_div'] #* LENGTH_SCALE_UM
        xmax_up = axI.get_ylim()[1] if axI.get_ylim() else 1.0
        plot_division_histogram_over_time(axJ, X_div_up, xmax=xmax_up, bins=30, label_xaxis=False)
    else:
        axJ.text(0.5, 0.5, "No multi-cell data (up).", ha="center", va="center", transform=axJ.transAxes)
    
    # Row letters A, B, C on the left (one per logical row)
    left_margin_ABC = 0.005
    label_axes = [("A", axA), ("B", axC), ("C", axG)]
    for txt, ref_ax in label_axes:
        pos = ref_ax.get_position()
        y_top = pos.y1 + 0.04
        fig.text(left_margin_ABC, y_top, txt, ha='center', va='center', fontsize=13, fontweight="bold")

    # Tidy titles a touch to avoid overlap
    for ax in [axA, axB, axC, axD, axE, axF, axG, axH, axI, axJ]:
        t = ax.title
        if t:
            ax.title.set_y(1.02)

    axG.set_xlabel(TIME_LABEL)
    axH.set_xlabel(FORCE_LABEL)
    axI.set_xlabel(TIME_LABEL)
    axJ.set_xlabel("Frequency")

    # small downward nudge for middle and bottom rows to give room for left labels
    extra_gap = 0.015
    for ax in (axC, axD, axE, axF, axG, axH, axI, axJ):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - extra_gap, pos.width, pos.height])

    # create aligned colourbars after axes positions finalised
    if last_pcm_E is not None:
        divE = make_axes_locatable(axE)
        caxE = divE.append_axes("right", size="4%", pad=0.02)
        cbarE = fig.colorbar(last_pcm_E, cax=caxE)
        cbarE.ax.yaxis.set_ticks_position('right')
    if last_pcm_I is not None:
        divI = make_axes_locatable(axI)
        caxI = divI.append_axes("right", size="4%", pad=0.02)
        cbarI = fig.colorbar(last_pcm_I, cax=caxI)
        cbarI.ax.yaxis.set_ticks_position('right')

    # Save figure
    plt.savefig(OUTFILE, dpi=300, bbox_inches='tight')
    print("Saved:", OUTFILE)
    plt.close(fig)

if __name__ == "__main__":
    main()