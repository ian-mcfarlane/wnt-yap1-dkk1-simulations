"""
Simulation module for mechano-chemical gastric gland model.

This module contains `run_sim(...)` which runs and saves a single hybrid model
simulation. Helper functions are organised as follows:
- utilities and RNG
- nondimensionalisation
- mechano-chemical helper functions (YAP, Hill, compression)
- PDE/source helpers and steady-state constructors
- core njit RHS for cells
- `run_sim`

"""
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


from pathlib import Path
from datetime import datetime
import glob
import hashlib
import pickle
import math
import numpy as np
from scipy.integrate import solve_ivp, quad
from numba import njit
from multicellular_sims.voronoi_method import voronoi_backward_euler_half
import yaml

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def make_rng_from_params(*params, context: str = "") -> np.random.Generator:
    def _norm(p):
        if isinstance(p, (float, np.floating)):
            return f"{float(p):.8f}"
        return str(p)
    key = "|".join(_norm(p) for p in params) + f"|{context}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    entropy = np.frombuffer(digest, dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ------------------------------------------------------------------------------
# Nondimensional groups
# ------------------------------------------------------------------------------
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

    nd['xi_D']       = p['xi_D']
    nd['K_sigma'] = (p['K_sigma'] if 'K_sigma' in p
                     else p['K_sigma_dim'] / (kappa_dim * L_dim))
    nd['A_sigma'] = p['A_sigma']
    nd['alpha_M'] = p['alpha_M']
    nd['Y_tot']   = p['Y_tot']
    nd['xi_M']    = p['xi_M']
    nd['n_Y']     = p['n_Y']
    nd['tau_M']   = p['tau_M']

    nd['P_0']      = p.get('p_0_dim', 1.5 / T_DAY) * T_DAY
    nd['W_div_switch']  = p.get('K_div_dim', 0.025 * R0) / R0
    nd['p_div']         = int(p.get('p_div', 4))
    nd['q_div']         = int(p.get('q_div', 8))
    nd['ell_div']  = p.get('ell_div_dim', 4.9) / L_dim
    nd['r_div_half']    = p.get('r_div_half', 1.0)

    return nd

# ------------------------------------------------------------------------------
# Mechano-chemical helper functions
# ------------------------------------------------------------------------------
def hill_Y(y, K, n):
    y = np.asarray(y, float)
    t = (y / K) ** n
    return t / (1.0 + t)


def hill_sigma(sigma, K_sigma):
    sigma = np.asarray(sigma, float)
    return sigma / (K_sigma + sigma)


def k_in_IE(sigma, M, par):
    return par['beta_off3'] * (1.0 + par['A_sigma'] * hill_sigma(sigma, par['K_sigma']) + par['alpha_M'] * M)


def k_out_IE(par):
    return par['beta_off3']


@njit(cache=True)
def _yap_qssa_vec_njit(sigma, M, k0, K_sigma, A_sigma, alpha_M, Y_tot):
    n = M.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        S = sigma[i] / (K_sigma + sigma[i])
        A = k0 * (1.0 + A_sigma * S + alpha_M * M[i])
        B = k0
        out[i] = Y_tot * A / (A + B)
    return out


def yap_qssa_vec(sigma, M, par):
    return _yap_qssa_vec_njit(
        np.asarray(sigma, dtype=np.float64),
        np.asarray(M, dtype=np.float64),
        par['beta_off3'],
        par['K_sigma'],
        par['A_sigma'],
        par['alpha_M'],
        par['Y_tot']
    )


def yap_rhs_cellwise(sigma, Y, M, par):
    Minf = hill_Y(Y, par['xi_M'], par['n_Y'])
    dM = (Minf - M) / par['tau_M']
    dY = np.zeros_like(Y)
    return dY, dM


@njit(cache=True)
def _compute_cell_compression_njit(xs, L_target):
    N = L_target.size
    sigma = np.empty(N, dtype=np.float64)
    for i in range(N):
        L = xs[i+1] - xs[i]
        d = L_target[i] - L
        sigma[i] = d if d > 0.0 else 0.0
    return sigma


def compute_cell_compression(xs, L_target):
    return _compute_cell_compression_njit(np.asarray(xs, dtype=np.float64), np.asarray(L_target, dtype=np.float64))


def _mask_to_float(mask):
    return np.where(mask, 1.0, 0.0)


def shape_scale_from_quantiles(q10, q90, p_low=0.1, p_high=0.9):
    a = -math.log(1.0 - p_low)
    b = -math.log(1.0 - p_high)
    r = max(q90 / max(q10, 1e-12), 1.0 + 1e-9)
    k = math.log(b / a) / math.log(r)
    lam = q10 / (a ** (1.0 / k))
    return float(k), float(lam)

def _trend_drift(ts, vals, window):
    """
    Least-squares slope over a trailing window, as a fraction of the window
    mean and of the within-window s.d. A half-split was tried first and
    rejects genuine oscillation, since a period comparable to the window puts
    the two halves out of phase; a regression slope averages whole periods
    away provided the window spans several of them.
    Returns (rel, in_sd, n_pts); (nan, nan, 0) while the window is not full.
    """
    ts = np.asarray(ts, float); vals = np.asarray(vals, float)
    if ts.size < 12 or (ts[-1] - ts[0]) < window:
        return np.nan, np.nan, 0
    m = ts >= ts[-1] - window
    tw, vw = ts[m], vals[m]
    if tw.size < 12:
        return np.nan, np.nan, 0
    b = float(np.polyfit(tw, vw, 1)[0])
    span = tw[-1] - tw[0]
    mu = max(abs(float(vw.mean())), 1e-12)
    sd = max(float(np.std(vw)), 1e-12)
    return abs(b) * span / mu, abs(b) * span / sd, int(tw.size)

# ------------------------------------------------------------------------------
# PDE / source and Green's-function helpers
# ------------------------------------------------------------------------------
@njit(cache=True)
def _remesh_core(x_old, c_old, x_new):
    n_new = x_new.size - 1
    mass = np.zeros(n_new)
    i = 0; j = 0
    while i < c_old.size and j < n_new:
        l = max(x_old[i], x_new[j]); r = min(x_old[i+1], x_new[j+1])
        if r > l: mass[j] += c_old[i] * (r - l)
        if x_old[i+1] <= x_new[j+1]: i += 1
        else: j += 1
    if x_new[-1] > x_old[-1]:
        for j in range(n_new):
            l = max(x_new[j], x_old[-1]); r = x_new[j+1]
            if r > l: mass[j] += c_old[-1] * (r - l)
    return mass / np.diff(x_new)

def remesh_conservative(x_old, c_old, x_new):
    x_old = np.asarray(x_old, dtype=float)
    c_old = np.asarray(c_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    return _remesh_core(x_old, c_old, x_new)

def _greens_norm(k, D, L):
    alpha = np.sqrt(k / D)
    return D * alpha * np.sinh(alpha * L)


def _G_kernel(alpha, L, x, xi):
    if xi <= x:
        return np.cosh(alpha * (L - x)) * np.cosh(alpha * xi)
    else:
        return np.cosh(alpha * (L - xi)) * np.cosh(alpha * x)


def S_step(x, t, L, S0, a, b):
    mask = (x >= a * L) & (x <= b * L)
    return S0 * _mask_to_float(mask)


def S_linear_to_0p7(x, t, L, S0, b=0.7):
    xb = b * L
    s = np.clip(1.0 - x / xb, 0.0, 1.0)
    s = np.where(x <= xb, s, 0.0)
    return S0 * s


def S_linear_to_top(x, t, L, S0):
    return S0 * np.clip(1.0 - x / np.maximum(L, 1e-12), 0.0, 1.0)


def S_gaussian0(x, t, L, S0, sigma):
    sig = max(float(sigma), 1e-8) * L
    return S0 * np.exp(-0.5 * (x / sig) ** 2)


def S_step_with_osc(x, t, L, S0, a, b, **kwargs):
    base = S_step(x, t, L, S0, a=a, b=b)
    amp = None
    for k in ('amp', 'amplitude', 'A', 'S0'):
        if k in kwargs:
            amp = float(kwargs[k]); break
    if amp is None:
        amp = 0.0
    freq  = float(kwargs.get('freq', 1.0))
    phase = float(kwargs.get('phase', 0.0))
    mask  = (x >= a * L) & (x <= b * L)
    osc   = (S0 * amp) * np.sin(2 * np.pi * freq * t + phase) * _mask_to_float(mask)
    return np.clip(base + osc, 0.0, None)


def S_step_with_osc2D(
    x, t, L, S0, a, b,
    amp_t=0.0, freq=1.0, phase_t=0.0,
    amp_x=0.0, n_spatial=1, phase_x=0.0, freq_x=0.0
):
    xa = np.asarray(x, dtype=float)
    mask = (xa >= a * L) & (xa <= b * L)
    width = max((b - a) * L, 1e-12)
    xi = np.where(mask, (xa - a * L) / width, 0.0)
    osc_t = amp_t * np.sin(2 * np.pi * freq * t + phase_t)
    osc_x = amp_x * np.sin(2 * np.pi * (n_spatial * xi + freq_x * t) + phase_x)
    val = S0 * (1.0 + osc_t + osc_x) * _mask_to_float(mask)
    return np.clip(val, 0.0, None)


def source_integral(L, S_func, *, t=0.0, amp_key='S0', **S_kwargs):
    kw = dict(S_kwargs); kw[amp_key] = 1.0
    integrand = lambda xi: float(S_func(xi, t, L, **kw))
    val, _ = quad(integrand, 0.0, L)
    return val


def compute_source_amplitude_match_flux(target_flux, L, S_func, *, t=0.0, amp_key='S0', **S_kwargs):
    denom = source_integral(L, S_func, t=t, amp_key=amp_key, **S_kwargs)
    if abs(denom) < 1e-15:
        raise ValueError("Flux calibration failed: zero integral for unit amplitude.")
    return target_flux / denom


def wnt_steady_state_from_source(x_vals, k, D, L, S_func, *, t=0.0, **S_kwargs):
    alpha = np.sqrt(k / D)
    norm  = _greens_norm(k, D, L)
    if x_vals is None:
        x_vals = np.linspace(0, L, 200)
    x_vals = np.asarray(x_vals, float)
    W = np.empty_like(x_vals, dtype=float)
    for i, x in enumerate(x_vals):
        kernel = lambda xi: (_G_kernel(alpha, L, x, xi) / norm) * float(S_func(xi, t, L, **S_kwargs))
        I, _ = quad(kernel, 0.0, L)
        W[i] = I
    return x_vals, W


def compute_source_amplitude(W_value_target, x_ref, k, D, L, S_func, *, t=0.0, amp_key='S0', **S_kwargs):
    kwargs = dict(S_kwargs); kwargs[amp_key] = 1.0
    _, Wtest = wnt_steady_state_from_source([x_ref], k, D, L, S_func, t=t, **kwargs)
    I = float(Wtest[0])
    if abs(I) < 1e-15:
        raise ValueError("Zero Green's convolution at x_ref.")
    return W_value_target / I


def compute_source_amplitude_match_max(W_max_target, k, D, L, S_func, *, t=0.0, amp_key='S0', **S_kwargs):
    kwargs = dict(S_kwargs); kwargs[amp_key] = 1.0
    xs = np.linspace(0.0, L, 400)
    _, W = wnt_steady_state_from_source(xs, k, D, L, S_func, t=t, **kwargs)
    m = float(np.max(W))
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError("Calibration failed.")
    return W_max_target / m


# ------------------------------------------------------------------------------
# Core RHS (numba-compiled inner kernel)
# ------------------------------------------------------------------------------
@njit(cache=True)
def _rhs_cells_core(y, N,
                    grow_coef, L_max,
                    beta_on1, beta_off1, beta_on2, beta_off2,
                    xi_M, n_Y, tau_M, kappa_spring):
    """
    RHS with state ordering:
      [xs (N+1),
       L_target (N),
       W (N), C (N), D (N), DL (N), Y (N), M (N)]
    Returns derivative vector of length 8*N + 1.
    """
    off = 0
    xs = y[off:off + N + 1]; off += N + 1
    L_target = y[off:off + N]; off += N
    W_cell = y[off:off + N]; off += N
    C = y[off:off + N]; off += N
    D_cell = y[off:off + N]; off += N
    DL = y[off:off + N]; off += N
    Y = y[off:off + N]; off += N
    M = y[off:off + N]

    out = np.empty(8 * N + 1, dtype=np.float64)

    # velocity-like array for edge motion
    v = out[0:N + 1]
    for i in range(N + 1):
        v[i] = 0.0

    L = np.empty(N, dtype=np.float64)
    for i in range(N):
        L[i] = xs[i + 1] - xs[i]

    # if kappa_spring is scalar, use it for all springs
    for i in range(1, N):
        # v[i] = kappa[i] * (L[i] - L_target[i]) - kappa[i-1] * (L[i-1] - L_target[i-1])
        v[i] = kappa_spring * ((L[i] - L_target[i]) - (L[i - 1] - L_target[i - 1]))

    v[N] = -kappa_spring * (L[N - 1] - L_target[N - 1])

    pos = N + 1
    dL0 = out[pos:pos + N]; pos += N
    Ltc = np.empty(N, dtype=np.float64)
    for i in range(N):
        li = L_target[i]
        if li >= L_max:
            dL0[i] = 0.0
            Ltc[i] = L_max
        else:
            dL0[i] = grow_coef
            Ltc[i] = li

    # reaction / other derivatives
    dW = out[pos:pos + N]; pos += N
    dC = out[pos:pos + N]; pos += N
    dD = out[pos:pos + N]; pos += N
    dDL = out[pos:pos + N]; pos += N
    dY = out[pos:pos + N]; pos += N
    dM = out[pos:pos + N]

    # reaction terms per cell
    for i in range(N):
        Li = L[i] if L[i] > 1e-12 else 1e-12
        Ltot = Ltc[i] / Li
        l_free = Ltot - C[i] - DL[i]

        t1 = beta_on1 * W_cell[i] * l_free
        t2 = beta_off1 * C[i]
        dW[i] = -t1 + t2
        dC[i] =  t1 - t2

        u1 = beta_on2 * D_cell[i] * l_free
        u2 = beta_off2 * DL[i]
        dD[i]  = -u1 + u2
        dDL[i] =  u1 - u2

        dY[i] = 0.0
        yv = Y[i]
        if yv < 1e-12:
            yv = 1e-12
        elif yv > 1.0 - 1e-12:
            yv = 1.0 - 1e-12
        t = (yv / xi_M) ** n_Y
        Minf = t / (1.0 + t)
        dM[i] = (Minf - M[i]) / (tau_M if tau_M > 1e-12 else 1.0)

    # advection terms from edge motion (conservative remap effect)
    for i in range(N):
        dLi = v[i + 1] - v[i]
        Li = L[i] if L[i] > 1e-12 else 1e-12
        fac = dLi / Li
        dC[i]  -= fac * C[i]
        dDL[i] -= fac * DL[i]

    return out


def rhs_cells_fast(t, y, N, grow_coef, L_max, kappa_spring, nd):
    """
    Wrapper that supplies scalar kappa_spring and parameters into compiled core.
    """
    return _rhs_cells_core(
        np.asarray(y, dtype=np.float64), N,
        grow_coef, L_max,
        nd['beta_on1'], nd['beta_off1'], nd['beta_on2'], nd['beta_off2'],
        nd['xi_M'], nd['n_Y'], nd['tau_M'], kappa_spring
    )

def _apply_post_burnin(nd, post_burnin_scale, rapid_slough, tend,
                       p_low=0.1, p_high=0.9):
    """
    Everything that must happen when burn-in ends -- whether because the drift
    test settled, because the state came from a cached baseline, or because
    init_length != 1 skipped burn-in altogether. Factored out so the three
    paths cannot diverge.

    Returns (sl_shape, sl_scale, tend); the sl_* are None when rapid_slough is
    off, meaning "leave the baseline sloughing alone".
    """
    if post_burnin_scale:
        for k, f in post_burnin_scale.items():
            nd[k] = nd[k] * float(f)
        tend = 65 * 7.0
    sl_shape = sl_scale = None
    if rapid_slough:
        tau_min, tau_max = 1.5 / 4.6, 2.5 / 4.6
        sl_shape = ((math.log(-math.log(1 - p_high))
                     - math.log(-math.log(1 - p_low)))
                    / math.log(tau_max / tau_min))
        sl_scale = tau_min / (-math.log(1 - p_low)) ** (1 / sl_shape)
        tend = 30 * 7.0
    return sl_shape, sl_scale, tend

# ------------------------------------------------------------------------------
# Main simulation entrypoint (kappa is the renamed 'eps')
# ------------------------------------------------------------------------------
def run_sim(
    kappa, phi_D, a_step, b_step, rep, outdir, *,
    rapid_slough=False,
    source_fn=None, source_kwargs=None, amp_key='S0',
    minimal_save=False, maximal_save=False,
    D_W=None, D_D=None, init_length=1,
    # optional division-parameter overrides (safe defaults preserved)
    beta_div=None, L_div_factor=None,
    # optional t_end and burn_in control (for testing)
    t_end_input=None, burn_in_time_input=None,
    # optional different parameter file
    parameter_file=None,
    nd_overrides=None,        # dict applied to nd right after build_nd_groups
    post_burnin_scale=None,   # dict of multiplicative factors applied at the burn-in transition
    div_params=None,          # beta_div, p_div, q_div, r_div_half, L_div_factor
    baseline_dir=None,        # explicit directory to initialise from (replaces the string-surgery hack)
    tag=None,                 # appended to sim_id and hashed into the RNG
    max_length=3.0,
    sigma_ref=5.0e-4,         # compression at which the YAP range is matched
    # adaptive burn-in parameters
    adaptive_burn_in=True, probe_every=0.7, burn_window=None,
    drift_tol_rel=0.02, drift_tol_sd=0.6,
    max_burn=700.0,
    # adaptive time-stepping parameters
    dt_min=1e-4, hazard_tol=0.05, max_steps=None
):
    """
    Run a single stochastic simulation and save results to `outdir`.

    Parameter `kappa` (previously `eps`) is a scalar spring constant applied
    uniformly across all cell springs.
    """
    # load base params and override diffusivities if requested
    param_file = parameter_file or './parameters/base_params_yap_bistab_refined.yaml'
    with open(param_file, 'r') as f:
        base_params = yaml.safe_load(f)
    base_params['total_length'] = base_params.get('total_length', 600.0)
    p_local = dict(base_params)
    if D_W is not None:
        p_local['D_W'] = float(D_W)
    if D_D is not None:
        p_local['D_D'] = float(D_D)

    nd = build_nd_groups(p_local, kappa)
    if phi_D is not None:
        nd['phiD'] = float(phi_D)
    if nd_overrides:
        nd.update({k: float(v) for k, v in nd_overrides.items()})

    phi_D = float(nd['phiD']) 

    kappa_spring = float(kappa)

    # print all nd parameters if rep = 0
    if rep == 0:
        print("Non-dimensional parameters:")
        for k, v in sorted(nd.items()):
            print(f"  {k}: {v:.4g}")


    # mechanical defaults (dimensionless)
    L_target0 = 2.5 / base_params['total_length']
    L_max     = 2 * L_target0
    L_div_default = 0.95 * 2 * L_target0

    # allow overrides passed in by caller:
    if L_div_factor is None:
        L_div = L_div_default
    else:
        # treat L_div_factor as a multiplier of the previous default
        L_div = float(L_div_factor) * L_div_default

    # division parameters (allow overrides)
    div = dict(beta_div=nd['P_0'], p_div=nd['p_div'], q_div=nd['q_div'], r_div_half=1.0, L_div=nd['ell_div'])
    if div_params: div.update(div_params)
    beta_div = float(div["beta_div"])
    p_div = int(div["p_div"])
    q_div = int(div["q_div"])
    r_half = float(div["r_div_half"])
    L_div = float(div["L_div"])
    W_div_switch = float(nd['W_div_switch'])

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if minimal_save and maximal_save:
        raise ValueError("Set only one of minimal_save or maximal_save (not both).")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    tag_sfx = f"_{tag}" if tag else ""
    sim_id = (f"{now}_kappa{kappa_spring:.3g}_phiD{phi_D:.1f}_Wdiv{W_div_switch:.4f}"
              f"_a{a_step:.2f}_b{b_step:.2f}_rep{rep}{tag_sfx}")
    if init_length != 1:
        sim_id += f"_initL{init_length:.2f}"
    fname = outdir / (sim_id + ".pkl")

    pattern = str(outdir / (f"*_kappa{kappa_spring:.3g}_phiD{phi_D:.1f}"
                            f"_Wdiv{W_div_switch:.4f}_a{a_step:.2f}_b{b_step:.2f}"
                            f"_rep{rep}{tag_sfx}*.pkl"))
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    rng = make_rng_from_params(
        kappa_spring, phi_D, W_div_switch, a_step, b_step, rep,
        rapid_slough,
        source_fn, source_kwargs, amp_key,
        D_W, D_D, init_length,
        tag,
        context="outputs"
    )

    # slough Weibull initialization
    p_low, p_high = 0.1, 0.9
    tau_min, tau_max = 3 / 4.6, 5 / 4.6
    q10_cur, q90_cur = float(tau_min), float(tau_max)
    sl_shape, sl_scale = shape_scale_from_quantiles(q10_cur, q90_cur, p_low, p_high)
    tau_tip = 0.0

    # source selection and calibration
    if source_fn is None:
        source_fn = S_step
    if source_kwargs is None:
        source_kwargs = dict(a=a_step, b=b_step)

    norm_mode = source_kwargs.get('_norm', 'max')
    _skw = {k: v for k, v in source_kwargs.items() if k != '_norm'}
    W_target_max = 0.476 / nd['R0']
    a_cal = _skw.get('a', a_step); b_cal = _skw.get('b', b_step)
    x_ref = 0.5 * (a_cal + b_cal)

    S0_step_ref = compute_source_amplitude(
        W_target_max, x_ref, nd['deltaW'], nd['DW'], L=1.0,
        S_func=S_step, t=0.0, amp_key='S0', a=a_cal, b=b_cal
    )
    if rep == 0:
        print(f"Reference source amplitude for step: S0_step_ref = {S0_step_ref:.4g}",
              "\tor, per week:", S0_step_ref * 7.0)

    if norm_mode == 'flux':
        target_flux = S0_step_ref * (b_cal - a_cal)
        if source_fn in (S_step_with_osc, S_step_with_osc2D):
            S0 = S0_step_ref
        else:
            S0 = compute_source_amplitude_match_flux(
                target_flux, L=1.0, S_func=source_fn, t=0.0, amp_key=amp_key, **_skw
            )
    else:
        if source_fn in (S_step_with_osc, S_step_with_osc2D):
            S0 = S0_step_ref
        elif source_fn is S_step:
            S0 = S0_step_ref
        else:
            S0 = compute_source_amplitude_match_max(
                W_target_max, nd['deltaW'], nd['DW'], L=1.0,
                S_func=source_fn, t=0.0, amp_key=amp_key, **_skw
            )

    source_kwargs = dict(_skw); source_kwargs[amp_key] = S0

    t_star = 18.6 / 24
    grow_coef = ((L_max - L_target0 * (1 - 2 * np.exp(-kappa_spring * t_star)))
                 / (t_star - (1 / kappa_spring) * (1 - np.exp(-kappa_spring * t_star))))

    # simulation times
    tend, dt_max = 49.0, 5e-3
    dt = dt_max
    H_prev = 0.0             # hazard from the previous step; sets this step's dt
    n_dt_changes = 0
    n_dt_floor = 0
    dt_used_min = dt_max
    hdt_max = 0.0
    step = 0
    if max_steps is None:
        max_steps = int(3 * (max_burn + 500.0) / dt_max)

    if t_end_input is not None:
        tend = float(t_end_input)
    t_save_step = 0.1
    t_next_save = t_save_step
    burning_in_bool = True

    stationary = None
    burn_in_used = np.nan
    next_probe = 0.0
    pt, pL, pM = [], [], []

    # attempt to re-use recent baseline run for burn-in if available
    matches = []
    if baseline_dir:
        patt = str(Path(baseline_dir) / f"*_kappa{float(kappa_spring):.3g}_*_rep0*.pkl")
        matches = sorted(glob.glob(patt), key=lambda p: Path(p).stat().st_mtime, reverse=True)

    # initialise state either from a matched baseline or cold start
    if matches:
        with open(matches[0], "rb") as f:
            prev = pickle.load(f)
        if not prev.get("num_cells") or prev["num_cells"][-1] == 0:
            matches = []
        else:
            xs  = np.asarray(prev["all_xs"][-1])
            L0s = np.asarray(prev["all_L0s"][-1])
            N = xs.size - 1
            W   = np.asarray(prev["all_W"][-1])
            C   = np.asarray(prev["all_C"][-1])
            D   = np.asarray(prev["all_D"][-1])
            DL  = np.asarray(prev["all_DL"][-1])
            Y   = np.asarray(prev["all_YAP"][-1])
            M   = np.asarray(prev["all_M"][-1])

            state = np.concatenate([xs, L0s, W, C, D, DL, Y, M])
            N_prev = N

            burning_in_bool = False
            stationary = True
            burn_in_used = 0.0
            _s, _l, tend = _apply_post_burnin(nd, post_burnin_scale,
                                              rapid_slough, tend)
            if _s is not None:
                sl_shape, sl_scale = _s, _l

    if not matches:
        if init_length != 1:
            burning_in_bool = False
            stationary = True
            burn_in_used = 0.0
            _s, _l, tend = _apply_post_burnin(nd, post_burnin_scale,
                                              rapid_slough, tend)
            if _s is not None:
                sl_shape, sl_scale = _s, _l
            tend += 301.0

        if phi_D != 0:
            N = int(init_length // (2 * 0.75 * L_target0))
            xs = np.linspace(0, init_length, N + 1)
        else:
            N = 10
            xs = np.linspace(0, 2 * 0.75 * L_target0 * N, N + 1)
        N_prev = N
        L0s  = np.full(N, L_target0) * 2 * 0.75

        xg = 0.5 * (xs[:-1] + xs[1:])
        _, W = wnt_steady_state_from_source(xg, nd['deltaW'], nd['DW'], xs[-1], source_fn, t=0.0, **source_kwargs)
        C = np.zeros(N); D = np.zeros(N); DL = np.zeros(N); M = np.zeros(N)

        Y = yap_qssa_vec(compute_cell_compression(xs, L0s), M, nd)
        state = np.concatenate([xs, L0s, W, C, D, DL, Y, M])

    # adaptive burn-in settings, in DAYS (7-multiples for the scaled writeup)
    if burn_window is None:
        burn_window = max(126.0, 10.0 * nd["tau_M"])      # 18 weeks
    if not adaptive_burn_in:
        burn_in_time = (float(burn_in_time_input) if burn_in_time_input is not None
                        else max(126.0, 6.0 * nd["tau_M"]))
        max_burn = burn_in_time

    t = 0.0

    # recording buffers
    all_t, all_xs, all_L0s = [], [], []
    all_W, all_C, all_D, all_DL = [], [], [], []
    all_YAP, all_M = [], []
    division_positions = []

    slough_count = 0

    # main time loop
    while (t < tend and not burning_in_bool) or burning_in_bool:
        step += 1
        if step > max_steps:
            print(f"step limit ({max_steps}) reached; aborting.")
            break
        # Set dt for THIS step from the PREVIOUS step's hazard, so the ODE,
        # both PDE half-steps, the event test and the clock all advance by the
        # same amount. Adapting after H is computed -- as fast_model does --
        # advances the state by one dt and t by another.
        if H_prev > 0.0:
            dt_want = min(dt_max, hazard_tol / H_prev)
            if dt_want < dt_min:
                dt_want = dt_min
                n_dt_floor += 1
            if abs(dt_want - dt) > 1e-15:
                dt = dt_want
                n_dt_changes += 1
            dt_used_min = min(dt_used_min, dt)

        # # record progress every 100 time steps
        # if int(t / dt) % 100 == 0:print(f"t = {t:.3f}, N = {N}, L={xs[-1]}, slough_count = {slough_count}")

        xs = state[:N+1]
        current_length = xs[-1]
        cell_centers = 0.5 * (xs[:-1] + xs[1:])
        wnt_src = source_fn(cell_centers, t, current_length, **source_kwargs)

        # PDE half-step for W (W now at slice 2*N+1:3*N+1)
        W_cell, _ = voronoi_backward_euler_half(nd['DW'], nd['deltaW'], state[2*N+1:3*N+1], xs, None, wnt_src, dt/2)

        sigma_now = compute_cell_compression(state[:N+1], state[N+1:2*N+1])
        Y_qssa    = yap_qssa_vec(sigma_now, state[7*N+1:8*N+1], nd)  # M at 7*N+1:8*N+1
        state[6*N+1:7*N+1] = Y_qssa  # Y at 6*N+1:7*N+1
        Y = Y_qssa
        dkk1_src = nd['phiD'] * (1.0 / (1 + (nd['xi_D'] / (Y + 1e-16)) ** nd['m_dy']))
        # D now at slice 4*N+1:5*N+1
        D_cell, _ = voronoi_backward_euler_half(nd['DD'], nd['deltaD'], state[4*N+1:5*N+1], xs, None, dkk1_src, dt/2)

        state[2*N+1:3*N+1] = W_cell
        state[4*N+1:5*N+1] = D_cell

        xs_prev = state[:N+1]
        try:
            sol = solve_ivp(rhs_cells_fast, (0, dt), state,
                            args=(N, grow_coef, L_max, kappa_spring, nd),
                            method='RK45', max_step=dt/2, rtol=1e-6, atol=1e-8)
            # sol = solve_ivp(rhs_cells_fast, (0, dt), state,
            #                 args=(N, grow_coef, L_max, kappa_spring, nd),
            #                 method='Radau', max_step=dt/2, rtol=1e-6, atol=1e-8)
        except Exception as exc:
            print(f"Error during ODE integration: {exc}")
            break

        state = sol.y[:, -1]

        # # print diagnostics as indicators of stiffness: nfev and step size
        # print(f"t={t:.3f}, nfev={sol.nfev}, last_step={sol.t[-1] - sol.t[-2]:.2e}, N={N}")

        xs_new = state[:N+1]
        if np.min(np.diff(xs_new)) < 1e-6:
            print("Minimum cell edge separation too small; aborting.")
            break

        # conservative remap after spatial edges moved
        W_cell = remesh_conservative(xs_prev, state[2*N+1:3*N+1], xs_new)
        D_cell = remesh_conservative(xs_prev, state[4*N+1:5*N+1], xs_new)
        M_cell = remesh_conservative(xs_prev, state[7*N+1:8*N+1], xs_new)
        state[2*N+1:3*N+1] = W_cell
        state[4*N+1:5*N+1] = D_cell
        state[7*N+1:8*N+1] = M_cell

        # second PDE half-step
        sigma_now = compute_cell_compression(state[:N+1], state[N+1:2*N+1])
        Y_qssa    = yap_qssa_vec(sigma_now, state[7*N+1:8*N+1], nd)
        state[6*N+1:7*N+1] = Y_qssa
        Y = Y_qssa
        dkk1_src = nd['phiD'] * (1.0 / (1 + (nd['xi_D'] / (Y + 1e-16)) ** nd['m_dy']))

        xs = state[:N+1]
        cell_centers = 0.5 * (xs[:-1] + xs[1:])
        current_length = xs[-1]
        wnt_src = source_fn(cell_centers, t + dt/2, current_length, **source_kwargs)
        W_cell, _ = voronoi_backward_euler_half(nd['DW'], nd['deltaW'], W_cell, xs_new, None, wnt_src, dt/2)
        D_cell, _ = voronoi_backward_euler_half(nd['DD'], nd['deltaD'], D_cell, xs_new, None, dkk1_src, dt/2)

        state[2*N+1:3*N+1] = W_cell
        state[4*N+1:5*N+1] = D_cell

        # competing-risk slough vs division
        event_flag = False
        xs = state[:N+1]
        L0s = state[N+1:2*N+1]
        W_cell = state[2*N+1:3*N+1]
        C_cell = state[3*N+1:4*N+1]
        D_cell = state[4*N+1:5*N+1]
        DL_cell = state[5*N+1:6*N+1]
        Y_cell = state[6*N+1:7*N+1]
        M_cell = state[7*N+1:8*N+1]

        # tip slough hazard
        if N > 0:
            if tau_tip <= 0.0:
                h_sl = 0.0
            else:
                h_sl = (sl_shape / sl_scale) * (tau_tip / sl_scale) ** (sl_shape - 1.0)
        else:
            h_sl = 0.0

        Cp = C_cell ** p_div
        rq = (np.diff(xs) / L_div) ** q_div
        G = beta_div * (Cp / (Cp + W_div_switch ** p_div)) * (rq / (rq + r_half ** q_div))
        # ensure G is finite and non-negative (numerical safety)
        G = np.where(np.isfinite(G), G, 0.0)
        G = np.clip(G, 0.0, None)
        H = h_sl + G.sum()
        H_prev = H
        hdt_max = max(hdt_max, H * dt)

        p_any = 1 - np.exp(-H * dt)

        if rng.random() < p_any:
            event_flag = True
            u = rng.random() * H
            if u < h_sl:
                # slough at tip
                slough_count += 1
                old_edges = xs.copy()
                old_W = W_cell.copy()
                old_D = D_cell.copy()
                old_M = M_cell.copy()
                xs = xs[:-1]
                L0s = L0s[:-1]
                C_cell = C_cell[:-1]
                DL_cell = DL_cell[:-1]
                Y_cell = Y_cell[:-1]
                # M_cell = M_cell[:-1]
                N -= 1
                W_cell = remesh_conservative(old_edges, old_W, xs)
                D_cell = remesh_conservative(old_edges, old_D, xs)
                M_cell = remesh_conservative(old_edges, old_M, xs)
                tau_tip = 0.0
            else:
                # cumulative propensity
                cumG = np.cumsum(G)
                totalG = cumG[-1] if cumG.size > 0 else 0.0

                # defensive guard: if totalG == 0 then division is impossible; treat as no-event
                if totalG <= 0.0:
                    # fall back to tip-slough if h_sl > 0, otherwise skip event
                    # here we already know we're in the division branch (u >= h_sl),
                    # so skip dividing safely:
                    event_flag = False
                    tau_tip += dt
                else:
                    v = u - h_sl
                    # numerical safeguard: clamp v into [0, totalG*(1-eps)]
                    eps = 1e-12
                    if v < 0.0:
                        v = 0.0
                    if v >= totalG:
                        v = totalG * (1.0 - eps)

                    # pick index, then clip to valid range [0, N-1]
                    j = int(np.searchsorted(cumG, v, side='left'))
                    if j >= cumG.size:
                        j = cumG.size - 1
                    if j < 0:
                        j = 0

                    # ensure there is a right-hand edge to index (xs has length N+1)
                    if j + 1 >= xs.size:
                        j = max(0, xs.size - 2)

                    x_new = 0.5 * (xs[j] + xs[j + 1])

                    if not burning_in_bool:
                        division_positions.append(x_new / xs[-1])

                    old_edges = xs.copy()
                    old_W = W_cell.copy()
                    old_D = D_cell.copy()
                    old_M = M_cell.copy()
                    xs = np.insert(xs, j + 1, x_new)
                    L0s = np.insert(L0s, j + 1, L_target0)
                    L0s[j] = L_target0
                    C_cell = np.insert(C_cell, j + 1, C_cell[j])
                    DL_cell = np.insert(DL_cell, j + 1, DL_cell[j])
                    Y_cell = np.insert(Y_cell, j + 1, Y_cell[j])
                    # M_cell = np.insert(M_cell, j + 1, M_cell[j])
                    N += 1
                    tau_tip += dt

                    W_cell = remesh_conservative(old_edges, old_W, xs)
                    D_cell = remesh_conservative(old_edges, old_D, xs)
                    M_cell = remesh_conservative(old_edges, old_M, xs)
        else:
            tau_tip += dt

        state = np.concatenate([xs, L0s, W_cell, C_cell, D_cell, DL_cell, Y_cell, M_cell])

        # recording
        if (t >= t_next_save or event_flag or N != N_prev) and not burning_in_bool:
            while t >= t_next_save:
                t_next_save += t_save_step
            if N > 0:
                N_prev = N
                all_t.append(t)
                all_xs.append(state[:N+1].copy())
                all_L0s.append(state[N+1:2*N+1].copy())
                all_W.append(state[2*N+1:3*N+1].copy())
                all_C.append(state[3*N+1:4*N+1].copy())
                all_D.append(state[4*N+1:5*N+1].copy())
                all_DL.append(state[5*N+1:6*N+1].copy())
                all_YAP.append(state[6*N+1:7*N+1].copy())
                all_M.append(state[7*N+1:8*N+1].copy())
            else:
                all_t.append(t)
                all_xs.append(xs.copy())
                all_L0s.append([np.nan])
                all_W.append([np.nan]); all_C.append([np.nan])
                all_D.append([np.nan]); all_DL.append([np.nan])
                all_YAP.append([np.nan]); all_M.append([np.nan])

        t += dt
        if N == 0:
            print("No cells present; terminating simulation.")
            break
        if xs[-1] >= max_length:
            print(f"Tissue extended beyond {max_length}; terminating simulation.")
            break

        # end burn-in transitions
        current_length = xs[-1]
        settled = False
        if burning_in_bool and t >= next_probe:
            next_probe = t + probe_every
            pt.append(t); pL.append(float(xs[-1]))
            pM.append(float(np.mean(state[7*N+1:8*N+1])))
            rL, sL, nw = _trend_drift(pt, pL, burn_window)
            rM, sM, _ = _trend_drift(pt, pM, burn_window)
            settled = (nw > 0 and np.isfinite(rL) and np.isfinite(rM)
                        and rL < drift_tol_rel and sL < drift_tol_sd
                        and rM < drift_tol_rel and sM < drift_tol_sd)

        if burning_in_bool and (settled or t >= max_burn):
            stationary = bool(settled)
            burn_in_used = t
            
            _s, _l, tend = _apply_post_burnin(nd, post_burnin_scale,
                                              rapid_slough, tend)
            if _s is not None:
                sl_shape, sl_scale = _s, _l

            N_prev = N
            burning_in_bool = False
            t = 0.0
            t_next_save = t_save_step

            all_t, all_xs = [0.0], [state[:N+1].copy()]
            all_L0s = [state[N+1:2*N+1].copy()]
            all_W = [state[2*N+1:3*N+1].copy()]
            all_C = [state[3*N+1:4*N+1].copy()]
            all_D = [state[4*N+1:5*N+1].copy()]
            all_DL = [state[5*N+1:6*N+1].copy()]
            all_YAP = [state[6*N+1:7*N+1].copy()]
            all_M = [state[7*N+1:8*N+1].copy()]
            division_positions = []

    if n_dt_floor > 0 and hdt_max > 3 * hazard_tol:
        print(f"WARNING: dt hit the floor {n_dt_floor} times with "
              f"max H*dt = {hdt_max:.3f}; events are under-resolved.")

    # prepare save_dict
    if minimal_save:
        final_length = float(state[:N+1][-1]) if N > 0 else 0.0
        save_dict = {
            'key_parameters': dict(
                kappa_spring0=kappa_spring, phi_D=phi_D, W_div_switch=W_div_switch,
                a_step=a_step, b_step=b_step,
                D_W=(float(D_W) if D_W is not None else base_params['D_W']),
                D_D=(float(D_D) if D_D is not None else base_params['D_D']),
                DW_nd=float(nd['DW']), DD_nd=float(nd['DD']),
                init_length=init_length,
                beta_div=float(beta_div),
                p_div=int(p_div), q_div=int(q_div), r_div_half=float(r_half),
                L_div=float(L_div),
                sigma_ref=float(sigma_ref),
                A_sigma=float(nd['A_sigma']), alpha_M=float(nd['alpha_M']),
                stationary=stationary, burn_in_used=float(burn_in_used),
                dt_max=float(dt_max), dt_min_used=float(dt_used_min),
                n_dt_changes=int(n_dt_changes), n_dt_floor=int(n_dt_floor),
                max_H_dt=float(hdt_max),
            ),
            'final_length': final_length,
            'division_positions': division_positions
        }
    elif maximal_save or rep == 0:
        save_dict = {
            'key_parameters': dict(
                kappa_spring0=kappa_spring, phi_D=phi_D, W_div_switch=W_div_switch,
                a_step=a_step, b_step=b_step,
                D_W=(float(D_W) if D_W is not None else base_params['D_W']),
                D_D=(float(D_D) if D_D is not None else base_params['D_D']),
                DW_nd=float(nd['DW']), DD_nd=float(nd['DD']),
                init_length=init_length,
                beta_div=float(beta_div),
                p_div=int(p_div), q_div=int(q_div), r_div_half=float(r_half),
                L_div=float(L_div),
                stationary=stationary, burn_in_used=float(burn_in_used),
                dt_max=float(dt_max), dt_min_used=float(dt_used_min),
                n_dt_changes=int(n_dt_changes), n_dt_floor=int(n_dt_floor),
                max_H_dt=float(hdt_max),
            ),
            'other_parameters': nd,
            'all_t':    all_t,
            'all_xs':   all_xs,
            'all_L0s':  all_L0s,
            'all_W':    all_W,
            'all_C':    all_C,
            'all_D':    all_D,
            'all_DL':   all_DL,
            'all_YAP':  all_YAP,
            'all_M':    all_M,
            'num_cells': [len(xs) for xs in all_xs],
            'lengths': [xs[-1] for xs in all_xs],
            'division_positions': division_positions
        }
    else:
        save_dict = {
            'all_t': all_t,
            'num_cells': [len(xs) for xs in all_xs],
            'lengths': [xs[-1] for xs in all_xs],
            'division_positions': division_positions
        }

    with open(fname, 'wb') as f:
        pickle.dump(save_dict, f)

    return str(fname)