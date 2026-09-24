"""
Voronoi partitioning of cells into compartments, and diffusion on the resulting mesh,
adapted from Murphy et al. 2021 (The role of mechanical interactions in EMT - Physical Biology 18(4))
"""

import numpy as np
from scipy.linalg import solve_banded

def compute_resident_points(
    interval_lengths: np.ndarray,
    interval_source_rate: float,
    gamma: float = 0.8,
    max_loops: int = 5,
    tol: float = 1e-6,
):
    """
    Build a multi-compartment voronoi partition of cells, refining
    until the spacing of resident points is sufficiently uniform
    (|log10(max delta) - log10(min delta)| <= gamma), or until max_loops is reached.

    Parameters
    ----------
    interval_lengths : (N,) array
        Lengths of the N cells.
    interval_source_rate : float
        Source rate for the first compartment of the first cell
    gamma : float
        Tolerance on the log10-range of compartment spacings.
    max_loops : int
        Maximum number of refinement iterations.
    tol : float
        Small tolerance when checking 1-point-per-cell possibility.

    Returns
    -------
    res_points : (M,) array
        Final resident points (one per compartment).
    interval_lengths_compartments : (M,) array
        Lengths of each compartment.
    compartments_per_cell_overall : (N,) int array
        Total number of compartments in each original cell.
    interval_source_rates_compartments : (sum(compartments_per_cell_overall),) array
        Source-rate for each compartment.  Only the first compartment
        of the last cell carries the rate `interval_source_rates[-2]`.
    onerespossible : bool
        True if the final partition could have been done with exactly
        one point per *original* cell.
    """
    # start with 1 compartment per cell
    compartments_per_cell_1 = np.ones(len(interval_lengths), int)

    # initial voronoi
    (res_points,
     interval_lengths_compartments,
     compartments_per_cell_2,
     onerespossible) = voronoi_partition(
        interval_lengths, tol
    )

    interval_lengths_min = interval_lengths_compartments.copy()
    loop = 1
    while loop <= max_loops:
        delta = np.diff(res_points)
        if np.abs(np.log10(delta.max()) - np.log10(delta.min())) <= gamma:
            break

        # subdivide any cell whose compartments are too large
        max_len = 0.5 * interval_lengths_min.min()
        (interval_lengths_below_size_threshold,
         compartments_per_cell_1) = compartments_below_size_threshold(
            interval_lengths, max_len
        )

        # recompute voronoi on the refined mesh
        (res_points,
         interval_lengths_compartments,
         compartments_per_cell_2,
         onerespossible) = voronoi_partition(
            interval_lengths_below_size_threshold, tol
        )

        interval_lengths_min = interval_lengths_compartments.copy()
        loop += 1
    else:
        print("Voronoi loop max reached without satisfying uniformity condition")

    # now collapse the two-stage compartments_per_cell_2 into a single count per original cell
    N = len(compartments_per_cell_1)
    compartments_per_cell_overall = np.zeros(N, int)
    idx = 0
    for i in range(N):
        n_sub = compartments_per_cell_1[i]
        total  = compartments_per_cell_2[idx:idx+n_sub].sum()
        compartments_per_cell_overall[i] = total
        idx += n_sub

    # build source-rate vector for each compartment
    total_compartments = compartments_per_cell_overall.sum()
    interval_source_rates_compartments = np.zeros(total_compartments, float)

    target_idx = 0
    interval_source_rates_compartments[target_idx] = interval_source_rate

    return (
        res_points,
        interval_lengths_compartments,
        compartments_per_cell_overall,
        interval_source_rates_compartments,
        onerespossible
    )

def compartments_below_size_threshold(interval_lengths, max_length):
    """
    Splits each interval into enough equal-length compartments so that
    no compartment exceeds max_length.
    
    Args:
      interval_lengths: 1D array-like of length N, the original cell-interval lengths.
      max_length:       scalar, maximum allowed compartment length.
      
    Returns:
      lengths_below:    1D np.array of length M >= N, the new compartment lengths.
      comps_per_cell:   1D np.array of length N, number of compartments each original interval got.
    """
    interval_lengths = np.asarray(interval_lengths, float)
    N = len(interval_lengths)
    
    comps_per_cell = np.zeros(N, int)
    lengths_below = []
    
    # loop over all but the last "fake" interval
    for i in range(N - 1):
        L = interval_lengths[i]
        # how many pieces we need
        n = int(np.ceil(L / max_length))
        comps_per_cell[i] = n
        # each piece has length L / n
        lengths_below.extend([L / n] * n)
    
    # the final interval is a "fake" cell, keep it whole
    comps_per_cell[-1] = 1
    lengths_below.append(interval_lengths[-1])
    
    return np.array(lengths_below), comps_per_cell

def voronoi_partition(interval_lengths, tol=1e-6):
    """
    Wrapper: try 1 point per cell; if that fails, subdivide into compartments.

    Parameters
    ----------
    interval_lengths : array_like, shape (N,)
        Lengths of the N cells.
    tol : float
        Tolerance for degenerate Voronoi regions.

    Returns
    -------
    res_points : ndarray, shape (M,)
        Resident points (one per compartment).
    interval_lengths_compartments : ndarray, shape (M,)
        Lengths of each compartment.
    compartments_per_cell : ndarray, shape (N,)
        Number of compartments per original cell.
    possible : bool
        True if the strict one-per-cell partition was possible (no subdivision needed).
    """
    # build the full boundary array
    x_current = np.concatenate(([0.0], np.cumsum(interval_lengths)))
    # default outputs if no subdivision
    interval_lengths_compartments = np.array(interval_lengths, float)
    compartments_per_cell         = np.ones(len(interval_lengths), int)

    # first try strict 1-point-per-cell
    possible, res_points = _one_res_point_possible(
        x_current, tol
    )

    # if that failed, subdivide
    if not possible:
        (res_points,
         interval_lengths_compartments,
         compartments_per_cell,
         possible) = _divide_into_compartments(
            x_current, tol
        )

    return res_points, interval_lengths_compartments, compartments_per_cell, possible

def _one_res_point_possible(x_current, tol=1e-6):
    """
    Attempt a strict 1-resident-point-per-cell Voronoi on boundaries x (len N+1).
    If successful, returns (True, r) where r has length N and
      0.5*(r[i] + r[i+1]) == x[i+1]  for i=0..N-2.
    Otherwise returns (False, midpoints).
    """
    x = np.asarray(x_current, float)
    N = len(x) - 1
    # allocate feasible intervals [a_i, b_i] for each r_i
    a = np.empty(N)
    b = np.empty(N)
    # first cell: r0 ∈ [x0, x1]
    a[0], b[0] = x[0], x[1]
    # forward-propagate feasible intervals
    for i in range(1, N):
        # reflect the previous interval through midpoint x[i]
        low  = 2*x[i] - b[i-1]
        high = 2*x[i] - a[i-1]
        # intersect with the physical cell: r_i ∈ [x[i], x[i+1]]
        a[i] = max(x[i], low)
        b[i] = min(x[i+1], high)
        if b[i] - a[i] < tol:
            # no feasible region — give up
            mids = 0.5*(x[:-1] + x[1:])
            return False, mids
    # pick the last point in the middle of its feasible region
    r = np.empty(N)
    r[-1] = 0.5*(a[-1] + b[-1])

    for i in range(N-2, -1, -1):
        r[i] = 2*x[i+1] - r[i+1]
    return True, r
    
def _divide_into_compartments(x_current, tol=1e-6):
    """
    Subdivide any “bad” cells so that an interval-Voronoi partition 
    (one resident point per compartment) is possible.

    Parameters
    ----------
    x_current : array_like, shape (N+1,)
        Current cell-boundary positions.
    tol : float
        Minimum permissible width for a Voronoi region.

    Returns
    -------
    res_points : ndarray, shape (M,)
        Resident points (one per compartment).
    interval_lengths_compartments : ndarray, shape (M,)
        Lengths of each compartment.
    compartments_per_cell : ndarray, shape (N,)
        Number of compartments created in each original cell.
    possible : bool
        True if the algorithm finished without degeneracy.
    """
    # make a dynamic list of boundaries
    x_comp = list(x_current)
    N_cells = len(x_comp) - 1

    # start with exactly one compartment per cell
    compartments_per_cell = [1]*N_cells

    # possible-region arrays (dynamic as we may insert)
    vor_min = []
    vor_max = []
    vor_sz  = []

    # initialize first compartment's region
    vor_min.append(x_comp[0])
    vor_max.append(x_comp[1])
    vor_sz .append(vor_max[0] - vor_min[0])

    possible = True
    cell_ctr = 1        # how many original cells processed
    intv_ctr = 1        # how many compartments processed

    # loop until every original cell has been handled
    while cell_ctr < N_cells:
        cell_ctr += 1
        intv_ctr += 1

        # ensure region arrays are long enough
        while len(vor_min) <= intv_ctr:
            vor_min.append(0.0)
            vor_max.append(0.0)
            vor_sz .append(0.0)

        # if the previous region is too small, bail
        if vor_sz[intv_ctr-1] < tol:
            possible = False
            break

        # reflect the previous region into the current cell
        left_reflect  = (x_comp[intv_ctr] - vor_max[intv_ctr-1]) + x_comp[intv_ctr]
        right_reflect = (x_comp[intv_ctr] - vor_min[intv_ctr-1]) + x_comp[intv_ctr]
        hi = x_comp[intv_ctr+1]

        if   left_reflect <= hi and right_reflect <= hi:
            mn, mx = left_reflect, right_reflect

        elif left_reflect <= hi and right_reflect > hi:
            mn, mx = left_reflect, min(hi, right_reflect)

        else:
            # CASE 3: must subdivide the *previous* compartment into two
            prev_min = vor_min[intv_ctr-1]
            prev_max = vor_max[intv_ctr-1]

            # choose new boundary so reflection hits the cell edge
            new_x = 0.5*(prev_min + x_comp[intv_ctr])
            if new_x < prev_max:
                new_x = prev_max

            # first second-compartment region in previous cell
            mn = new_x + (new_x - prev_max)
            mx = x_comp[intv_ctr]
            vor_min[intv_ctr] = mn
            vor_max[intv_ctr] = mx
            vor_sz [intv_ctr] = mx - mn

            # insert the new boundary
            x_comp.insert(intv_ctr, new_x)
            compartments_per_cell[cell_ctr-2] = 2

            # insert slots in our region arrays
            vor_min.insert(intv_ctr, 0.0)
            vor_max.insert(intv_ctr, 0.0)
            vor_sz .insert(intv_ctr, 0.0)

            # now handle the *current* cell as a fresh compartment
            intv_ctr += 1
            if len(vor_min) <= intv_ctr:
                vor_min.append(0.0); vor_max.append(0.0); vor_sz.append(0.0)

            # recalc reflection into this new compartment
            left_reflect  = (x_comp[intv_ctr] - vor_max[intv_ctr-1]) + x_comp[intv_ctr]
            right_reflect = (x_comp[intv_ctr] - vor_min[intv_ctr-1]) + x_comp[intv_ctr]
            hi = x_comp[intv_ctr+1]

            if   left_reflect <= hi and right_reflect <= hi:
                mn, mx = left_reflect, right_reflect
            elif left_reflect <= hi and right_reflect > hi:
                mn, mx = left_reflect, min(hi, right_reflect)
            else:
                # in practice this second subdivision almost never recurses
                possible = False
                break

        # write the region for the “usual” cases (1 & 2) or after case 3
        vor_min[intv_ctr] = mn
        vor_max[intv_ctr] = mx
        vor_sz [intv_ctr] = mx - mn

    # final-compartment size check
    if possible and vor_sz[intv_ctr] < tol:
        possible = False

    # now place one resident point per compartment
    M = len(vor_min)
    res_points = np.zeros(M)
    if possible:
        res_points[-1] = 0.5*(vor_min[-1] + vor_max[-1])
        for k in range(M-2, -1, -1):
            # boundary x_comp[k+1] bisects res_points[k] and res_points[k+1]
            res_points[k] = x_comp[k+1] - (res_points[k+1] - x_comp[k+1])
    else:
        # fallback to midpoints of each (sub)interval
        res_points = np.array([
            0.5*(x_comp[i] + x_comp[i+1]) for i in range(len(x_comp)-1)
        ])

    # compartment lengths = diffs of the (possibly subdivided) boundary list
    interval_lengths_compartments = np.diff(np.array(x_comp))

    return (
        res_points,
        interval_lengths_compartments,
        np.array(compartments_per_cell, int),
        possible
    )

def find_x_prev_compartments(x_current_comp, x_prev_cells, 
                             x_current_cells, comps_per_cell):
    """
    Python port of function_DISCRETE_usenumcompartments_find_x_prev_m.
    Given:
      x_current_comp   : (M+1,) compartment boundaries [0 = b0 < b1 < ... < bM]
      x_prev_cells     : (N+1,) old cell boundaries
      x_current_cells  : (N+1,) current cell boundaries
      comps_per_cell   : (N,)  # compartments in each cell
    Returns:
      x_prev_comp      : (M+1,) reconstructed old compartment boundaries
    """
    if x_prev_cells is None:
        # first call: no "previous" so pretend the old cell-mesh was the same
        x_prev_cells = x_current_cells.copy()

    # if already passed a comp-mesh (len matches), just reuse it
    if len(x_prev_cells) == len(x_current_comp):
        return x_prev_cells.copy()
    
    M = len(x_current_comp) - 1
    N = len(comps_per_cell)
    x_prev_comp = np.zeros_like(x_current_comp)
    comp_idx = 0
    # we assume x_prev_comp[0] = x_prev_cells[0] = 0
    for cell in range(N):
        n = comps_per_cell[cell]
        cell_len_curr = x_current_cells[cell+1] - x_current_cells[cell]
        cell_len_prev = x_prev_cells[cell+1]   - x_prev_cells[cell]
        for _ in range(n):
            # how big is this new compartment?
            dx_comp = x_current_comp[comp_idx+1] - x_current_comp[comp_idx]
            frac    = dx_comp / cell_len_curr
            x_prev_comp[comp_idx+1] = x_prev_comp[comp_idx] + frac * cell_len_prev
            comp_idx += 1
    # enforce last boundary
    x_prev_comp[-1] = x_prev_cells[-1]
    return x_prev_comp

def calculate_prop_nonum(D, res_pts, x_end, tol=1e-12):
    """
    Jump rates on a 1D Voronoi mesh using ghost-points for zero-flux BC
    (second-order accurate at both ends).
    """
    y = np.asarray(res_pts, float)
    M = y.size
    if M < 2:
        return np.zeros(M), np.zeros(M)

    # build an extended array with one ghost at each boundary:
    #   left ghost at  -y[0]
    #   right ghost at 2*x_end - y[-1]
    y_ext = np.empty(M+2, float)
    y_ext[0]    = -y[0]
    y_ext[1:-1] = y
    y_ext[-1]   = 2*x_end - y[-1]

    # precompute all spacings
    ds = np.diff(y_ext)               # ds[i] = y_ext[i+1] - y_ext[i]

    T_plus  = np.zeros(M, float)
    T_minus = np.zeros(M, float)

    # loop over i=0...M-1
    #   left jump uses ds[i], right jump uses ds[i+1], and span = ds[i]+ds[i+1]
    for i in range(M):
        hm   = ds[i]       # y_i - y_{i-1}
        hp   = ds[i+1]     # y_{i+1} - y_i
        span = hm + hp     # y_{i+1} - y_{i-1}

        # protect against pathological small or zero distances
        hm  = max(hm,  tol)
        hp  = max(hp,  tol)
        span = max(span, tol)

        T_plus[i]  = 2*D / (hp   * span)
        T_minus[i] = 2*D / (hm   * span)

    # enforce zero-flux at the physical ends
    T_minus[0]  = 0.0
    T_plus[-1]  = 0.0

    return T_plus, T_minus

def voronoi_backward_euler_half(D, k_deg, c_cells, x_cells, x_prev_cells,
                                    source_rates_cells, dt_half):
    """
    One half-step of diffusion + influx + decay + dilution on the evolving mesh,
    via backward-Euler for the diffusion+decay piece.

    Inputs:
      D                  diffusion coefficient
      k_deg              decay rate
      c_cells            (N,) current cell concentrations
      x_cells            (N+1,) current cell boundaries
      x_prev_cells       (N+1,) previous cell boundaries
      source_rates_cells (N,) source term per cell
      dt_half            half-timestep
    Returns:
      c_cells_new        (N,) updated cell concentrations
      x_curr_comp        (M+1,) current compartment boundaries (for next call)
    """
    # (1) build compartments
    lengths          = np.diff(x_cells)
    res_pts, lengths_comp, comps, _ = voronoi_partition(lengths)
    M                = len(lengths_comp)

    # (2) lift to compartments
    c_comp           = np.repeat(c_cells,             comps)
    src_comp         = np.repeat(source_rates_cells, comps)

    # (3) get old/new comp-meshes
    x_curr_comp      = np.concatenate(([x_cells[0]], np.cumsum(lengths_comp)))
    x_prev_comp      = find_x_prev_compartments(
                           x_curr_comp, x_prev_cells, x_cells, comps)

    # (4) jump-rates
    prop_fwd, prop_bwd = calculate_prop_nonum(D, res_pts, x_cells[-1])
    prop_bwd[0]    = 0.0
    prop_fwd[-1]   = 0.0

    # (5) build backward-Euler tridiagonal system
    dx_comp        = np.diff(x_curr_comp)     # (M,)
    dx_prev        = np.diff(x_prev_comp)     # (M,)

    # explicit source + dilution
    src_term       = src_comp
    dil_term       = - (c_comp / dx_comp) * (dx_comp - dx_prev)

    # correct for irregular mesh:
    h = dx_comp                      # (M,) array of compartment lengths
    # build RHS = M c^n + dt*(M*source + M*dilution)
    rhs = h*c_comp + dt_half*( h*src_term + h*dil_term )

    # build banded ab for (M - dt*A) c_new = rhs
    ab = np.zeros((3, M))
    # upper diag ab[0,i] = -dt * A_{i-1,i} = -dt * (T^-_i * h_i)
    ab[0,1:]  = -dt_half * prop_bwd[1:] * h[1:]
    # main diag ab[1,i] = M_{ii} - dt*A_{ii} 
    #               = h_i + dt*(h_i*(T^+_i + T^-_i) + k_deg*h_i)
    ab[1,:]   = h + dt_half*( h*(prop_fwd + prop_bwd) + k_deg*h )
    # lower diag ab[2,i] = -dt * A_{i+1,i} = -dt * (T^+_i * h_i)
    ab[2,:-1] = -dt_half * prop_fwd[:-1] * h[:-1]

    # solve
    c_comp_new = solve_banded((1,1), ab, rhs)

    # (6) collapse back to cell level
    c_cells_new = np.empty_like(c_cells)
    idx = 0
    for j, n in enumerate(comps):
        if n == 1:
            c_cells_new[j] = c_comp_new[idx]
        else:
            seg = slice(idx, idx+n)
            c_cells_new[j] = np.dot(c_comp_new[seg], dx_comp[seg]) / dx_comp[seg].sum()
        idx += n

    return c_cells_new, x_curr_comp

def collapse_compartments_to_cells(c_comp, lengths_comp, comps_per_cell):
    """
    Given
      c_comp          (M,)   concentrations in each of M compartments
      lengths_comp    (M,)   lengths of each compartment
      comps_per_cell  (N,)   for each of the N biological cells, how many compartments it got
    Return
      W_cell (N,)  = length-weighted average of c_comp over each cell's compartments
    """
    W_cell = np.empty(len(comps_per_cell), float)
    idx = 0
    for i, n_comp in enumerate(comps_per_cell):
        if n_comp == 1:
            # trivial case: one compartment == one cell
            W_cell[i] = c_comp[idx]
        else:
            # take length-weighted average
            comp_slice = slice(idx, idx + n_comp)
            vols = lengths_comp[comp_slice]
            conc = c_comp[comp_slice]
            W_cell[i] = np.dot(conc, vols) / vols.sum()
        idx += n_comp
    return W_cell
