"""
Linear equalization optimizers.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 30, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from collections.abc import Callable
from functools import partial
from typing     import Any, Optional

import numpy as np
# pylint: disable=redefined-builtin
from numpy import (
    argmax, array, array_equal, concatenate, dot, einsum, identity, ix_,
    log10, maximum, minimum, ones, pad, sqrt, sum, vectorize, where, zeros)
from numpy.linalg import lstsq, solve as np_solve
from numpy.typing import NDArray
from scipy.linalg import convolution_matrix, solve, toeplitz

from pychopmarg.common  import Rmat, Rvec, NormMode
from pychopmarg.noise   import NoiseCalc
from pychopmarg.utility import calc_Hffe


def scale_taps(w: Rvec, w_min: Optional[Rvec] = None, w_max: Optional[Rvec] = None) -> Rvec:
    """
    Scale tap weights proportionally, to just fit inside the given min/max limits.

    Args:
        w: The tap weights to scale.

    Keyword Args:
        w_min: Minimum tap weights.
            Default: None (Use ``-ones(len(w))``.)
        w_max: Maximum tap weights.
            Default: None (Use ``ones(len(w))``.)

    Returns:
        Tap weights, scaled proportionally to just fit inside min/max limits.
    """

    if w_min is None:
        w_min = -ones(len(w))
    if w_max is None:
        w_max = ones(len(w))

    assert len(w) == len(w_min) == len(w_max), ValueError(
        f"Lengths of: `w` ({len(w)}), `w_min` ({len(w_min)}), and `w_max` ({len(w_max)}), must be equal!")

    def filt_scalars(x: float) -> bool:
        "Filter scaling values."
        return 1 > x > 0
    vfilt_scalars = vectorize(filt_scalars)

    w_scalars_min = w_min / w
    w_scalars_max = w_max / w
    w_scalars = minimum(
        where(vfilt_scalars(w_scalars_min), w_scalars_min, 1),
        where(vfilt_scalars(w_scalars_max), w_scalars_max, 1),
    )
    w_scale = np.min(w_scalars)

    return w_scale * w


def clip_taps(
    w: Rvec, curs_ix: int,
    w_min: Optional[Rvec] = None, w_max: Optional[Rvec] = None,
    obey_spec: bool = True
) -> Rvec:
    """
    Clip tap weights to the given min/max limits, as per (178A-26).

    Args:
        w: The tap weights to clip.
        curs_ix: The index, in ``w``, of the cursor tap.

    Keyword Args:
        w_min: Minimum tap weights.
            Default: None (Use ``-ones(len(w))``.)
        w_max: Maximum tap weights.
            Default: None (Use ``ones(len(w))``.)
        obey_spec: Allow cursor tap weight to grow w/o bound, as per D1.2, when True.
            Default: True

    Returns:
        Tap weights, clipped accordingly.
    """

    if w_min is None:
        w_min = -ones(len(w))
    if w_max is None:
        w_max = ones(len(w))

    assert len(w) == len(w_min) == len(w_max), ValueError(
        f"Lengths of: `w` ({len(w)}), `w_min` ({len(w_min)}), and `w_max` ({len(w_max)}), must be equal!")

    if obey_spec:
        # With cursor tap weight limits normalized to current cursor tap weight, which IS what D1.2 (and D1.1) specify.
        # Note that this allows the filter gain to grow without bound, since:
        #     a) the cursor tap weight is never clipped, and
        #     b) the other taps grow ratiometrically w/ the cursor tap.
        #
        #   1. Is that correct/intended?
        #
        #   2. What effect might the resultant variation in noise amplification thru the FFE
        #      have on our otherwise apples-to-apples comparison of tap weight "bestness"?
        w_lim_curs_val = w[curs_ix]
        w_lim = minimum(w_max * w_lim_curs_val, maximum(w_min * w_lim_curs_val, w))
    else:
        # With cursor tap weight bound to absolute limits, which is NOT what D1.2 (or D1.1) specifies,
        # but WAS the intent in some previous presentations on this topic.
        w_lim_curs_val = min(w_max[curs_ix], max(w_min[curs_ix], w[curs_ix]))
        w_lim = minimum(w_max * w_lim_curs_val, maximum(w_min * w_lim_curs_val, w))
        w_lim[curs_ix] = w_lim_curs_val

    return w_lim


def przf(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    pulse_resp: Rvec, nspui: int, nTaps: int, nPreTaps: int, nDFETaps: int,
    tap_mins: Rvec, tap_maxs: Rvec, b_min: Rvec, b_max: Rvec,
    norm_mode: NormMode = NormMode.P8023dj, unit_amp: bool = False
) -> tuple[Rvec, Rvec, Rvec]:
    """
    Optimize FFE tap weights, via *Pulse Response Zero Forcing* (PRZF).

    Args:
        pulse_resp: The pulse response to be filtered.
        nspui: Number of samples per unit interval.
        nTaps: The total number of FFE filter taps, including the cursor.
        nPreTaps: The number of pre-cursor taps.
        nDFETaps: Number of DFE taps.
        tap_mins: Minimum allowed tap values.
        tap_maxs: Maximum allowed tap values.
        b_min: Minimum allowed DFE tap values.
        b_max: Maximum allowed DFE tap values.

    Keyword Args:
        norm_mode: The tap weight normalization mode to use.
            Default: ``P8023dj``
        unit_amp: Enforce unit pulse response amplitude when True.
            (For comparing results to ``mmse()``.)
            Default: False

    Returns:
        A triple consisting of

            - The optimum FFE tap weights.
            - The optimum DFE tap weights.
            - The pulse response samples used in optimization.

    Notes:
        1. The algorithm implemented below is a slightly modified version of:

            Mellitz, R., Lusted, K., *RX FFE Implementation Algorithm for COM 4.1*,
            IEEE P802.3dj Task Force, August 31, 2023.

    ToDo:
        1. Add sampling time as an input parameter?
    """

    assert len(pulse_resp) >= nTaps * nspui, ValueError(
        "\n\t".join(
            [f"The pulse response length ({len(pulse_resp)}) must be at least:",
             f"`nspui` ({nspui}) * `nTaps` ({nTaps}) = {nspui * nTaps}!"]))
    assert nTaps == 0 and nPreTaps == 0 or nTaps > nPreTaps, ValueError(
        f"`nTaps` ({nTaps}) must be greater than `nPreTaps` ({nPreTaps})!")
    assert nTaps == 0 and nPreTaps == 0 or (nPreTaps + nDFETaps) < (nTaps - 1), ValueError(
        f"The sum of `nPreTaps` ({nPreTaps}) and `nDFETaps` ({nDFETaps}) must be less than: `nTaps` ({nTaps}) - 1!")

    # Construct null filter, as default.
    dw = nPreTaps
    wn: Rvec = zeros(nTaps)
    if nTaps > 0:
        wn[dw] = 1.0

    # Sample the given pulse response, assuming cursor coincides w/ maximum.
    dh, first_samp = divmod(argmax(pulse_resp), nspui)
    h = pulse_resp[first_samp::nspui]
    bn = minimum(b_max, maximum(b_min, h[dh + 1: dh + 1 + nDFETaps]))  # default DFE tap weights
    len_h = len(h)
    d = dh + nPreTaps

    if nTaps > 0:
        # Create the appropriate forcing vector.
        h_norm = h.copy()
        if unit_amp:
            h_norm /= max(h_norm)
        fv: Rvec = zeros(len_h)
        fv[dh] = h_norm[dh]                         # Don't force the cursor to zero.
        dfe_ixs = slice(dh + 1, dh + nDFETaps + 1)  # indices of DFE taps
        # Bound first `nDFETaps` post-cursor taps to DFE's correction limits.
        fv[dfe_ixs] = minimum(b_max * h_norm[dh],
                              maximum(b_min * h_norm[dh],
                                      h_norm[dfe_ixs]))
        fv = pad(fv, (nPreTaps, 0))[:len_h]  # Adding expected delay, `dw`, due to Rx FFE pre-cursor taps.

        # Find the optimum FFE tap weights.
        H = convolution_matrix(h, nTaps, mode='full')[:len_h]
        h0 = H[d]
        Hb = H[d + 1: d + 1 + nDFETaps]
        w, _, _, _ = lstsq(H, fv, rcond=None)

        # Check and enforce FFE tap weight limits, according to given normalization mode.
        match norm_mode:
            case NormMode.P8023dj:
                wn = clip_taps(w, dw, tap_mins, tap_maxs)
                wn /= dot(h0, wn)
            case NormMode.Scaled:
                wn = scale_taps(w, tap_mins, tap_maxs)
            case NormMode.Unaltered:
                wn = w
            case NormMode.UnitDcGain:
                wn = w / sum(w)
            case _:
                raise RuntimeError(
                    f"Received unknown normalization mode: {norm_mode}!")

        # Calculate desired DFE tap weight values and enforce limits.
        bn = Hb @ wn.flatten()
        bn = minimum(b_max, maximum(b_min, bn))

    return wn, bn, h[dh - nPreTaps: dh - nPreTaps + nTaps]


def _mmse_solve(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    H: Rmat, R: Rmat, d: int, Nb: int, dw: int,
    b_min: Rvec, b_max: Rvec, w_min: Rvec, w_max: Rvec,
    norm_mode: NormMode, varX: float, Rlm: float, L: int
) -> dict[str, Any]:
    """
    MMSE Rx FFE/DFE solution, w/ DFE and FFE limits enforced, for the given FFE taps, as per (178A-33..36)
    and MATLAB COM's ``MMSE_FOM()``.

    Args:
        H: Pulse response convolution matrix, one column per FFE tap.
        R: ``H.T @ H + Rnn / varX`` (178A-33), for the same taps.
        d: Index (in ``H``'s rows) of the cursor.
        Nb: Number of DFE taps.
        dw: Index (in ``H``'s columns) of the FFE's cursor tap.
        b_min: Minimum DFE tap weight values.
        b_max: Maximum DFE tap weight values.
        w_min: Minimum FFE tap weight values (relative to the cursor tap), for the same taps.
        w_max: Maximum FFE tap weight values (relative to the cursor tap), for the same taps.
        norm_mode: The tap weight normalization mode to use.
        varX: Signal power.
        Rlm: Relative level mismatch.
        L: Number of modulation levels.

    Returns:
        A dictionary containing the FOM ("fom"), MSE ("mse"), FFE and DFE tap weights ("w_lim", "b_lim"),
        and intermediate values.
    """
    Nw = H.shape[1]
    h0 = H[d]
    Hb = H[d + 1: d + 1 + Nb]
    Ib = identity(Nb)
    zb = zeros(Nb)
    A = concatenate((concatenate((R, -Hb.T, -h0.reshape((Nw, 1))), axis=1),
                     concatenate((-Hb, Ib, zeros((Nb, 1))), axis=1),
                     concatenate((h0, zb, zeros(1))).reshape((1, Nw + Nb + 1))))
    y = concatenate((h0, zeros(Nb), ones(1)))
    x = solve(A, y)
    w = x[:Nw]
    lam = x[-1]

    # Check DFE tap weights, enforcing limits if necessary.
    b = x[Nw: Nw + Nb]
    b_lim = maximum(b_min, minimum(b_max, b))
    delta_w = zeros(len(w))
    hit_b_limit = False
    if not array_equal(b_lim, b):
        hit_b_limit = True
        _A = concatenate((concatenate((R, -h0.reshape((Nw, 1))), axis=1),
                          concatenate((h0, zeros(1))).reshape((1, -1))))
        _y = concatenate((h0 + Hb.T @ b_lim, ones(1)))
        _x = solve(_A, _y)
        _w = _x[:Nw]
        delta_w = _w - w
        lam = _x[-1]
        w = _w

    # Clip to limits if apropos.
    if norm_mode == NormMode.P8023dj:
        w_lim = clip_taps(w, dw, w_min, w_max)
    else:
        w_lim = w

    # Make final adjustments if necessary.
    if not array_equal(w_lim, w):
        w_lim /= dot(h0, w_lim)  # Normalize for unit pulse response amplitude.
        b = Hb @ w_lim.flatten()
        b_lim = maximum(b_min, minimum(b_max, b))

    mse = varX * (w_lim @ R @ w_lim.T + 1 + b_lim @ b_lim - 2 * w_lim @ h0 - 2 * w_lim @ Hb.T @ b_lim).flatten()[0]
    fom = 20 * log10(Rlm / (L - 1) / sqrt(mse))
    return {"fom": fom, "mse": mse, "w_lim": w_lim, "b_lim": b_lim, "lam": lam, "delta_w": delta_w,
            "hit_b_limit": hit_b_limit, "h0": h0, "Hb": Hb, "A": A, "y": y, "x": x, "b": b}


def _mmse_fom_batch(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    H_all: Rmat, R_all: Rmat, cols: NDArray[np.int_], d: int, Nb: int, dw: int,
    b_min: Rvec, b_max: Rvec, w_min: Rvec, w_max: Rvec,
    norm_mode: NormMode, varX: float, Rlm: float, L: int
) -> Rvec:
    """
    The FOM of ``_mmse_solve()`` for each of several sets of FFE tap positions (the rows of ``cols``), at once.

    Args:
        H_all: Pulse response convolution matrix, one column per FFE tap position.
        R_all: ``H_all.T @ H_all + Rnn / varX``, for the same positions.
        cols: FFE tap positions (indices into the columns of ``H_all``), one set per row.
        d: Index (in ``H_all``'s rows) of the cursor.
        Nb: Number of DFE taps.
        dw: Index (in each row of ``cols``) of the FFE's cursor tap.
        b_min: Minimum DFE tap weight values.
        b_max: Maximum DFE tap weight values.
        w_min: Minimum FFE tap weight values (relative to the cursor tap), for every position.
        w_max: Maximum FFE tap weight values (relative to the cursor tap), for every position.
        norm_mode: The tap weight normalization mode to use.
        varX: Signal power.
        Rlm: Relative level mismatch.
        L: Number of modulation levels.

    Returns:
        The FOM (dB) for each row of ``cols``.
    """
    K, n = cols.shape
    h0 = H_all[d][cols]                                         # K x n
    Hb = H_all[d + 1: d + 1 + Nb][:, cols].transpose(1, 0, 2)   # K x Nb x n
    R = R_all[cols[:, :, None], cols[:, None, :]]               # K x n x n
    m = n + Nb + 1
    A = zeros((K, m, m))
    A[:, :n, :n] = R
    A[:, :n, n: n + Nb] = -Hb.transpose(0, 2, 1)
    A[:, :n, -1] = -h0
    A[:, n: n + Nb, :n] = -Hb
    A[:, n: n + Nb, n: n + Nb] = identity(Nb)
    A[:, -1, :n] = h0
    y = concatenate((h0, zeros((K, Nb)), ones((K, 1))), axis=1)
    x = np_solve(A, y[..., None])[..., 0]
    w = x[:, :n]
    b = x[:, n: n + Nb]

    # DFE limits: re-solve the FFE, w/ the DFE taps clipped, where any was.
    b_lim = maximum(b_min, minimum(b_max, b))
    hit = (b_lim != b).any(axis=1)
    if hit.any():
        _A = zeros((K, n + 1, n + 1))
        _A[:, :n, :n] = R
        _A[:, :n, -1] = -h0
        _A[:, -1, :n] = h0
        _y = concatenate((h0 + einsum('kbn,kb->kn', Hb, b_lim), ones((K, 1))), axis=1)
        w[hit] = np_solve(_A[hit], _y[hit][..., None])[..., 0][:, :n]

    # FFE limits, relative to the cursor tap; then renormalize, and recompute the DFE.
    if norm_mode == NormMode.P8023dj:
        w_curs = w[:, dw: dw + 1]
        w_lim = minimum(w_max[cols] * w_curs, maximum(w_min[cols] * w_curs, w))
    else:
        w_lim = w
    clipped = (w_lim != w).any(axis=1)
    if clipped.any():
        w_lim[clipped] /= (h0 * w_lim).sum(axis=1)[clipped, None]
        b_lim[clipped] = maximum(b_min, minimum(b_max, einsum('kbn,kn->kb', Hb, w_lim)))[clipped]

    wRw = einsum('kn,knm,km->k', w_lim, R, w_lim)
    wh0 = (w_lim * h0).sum(axis=1)
    wHbb = einsum('kn,kbn,kb->k', w_lim, Hb, b_lim)
    mse = varX * (wRw + 1 + (b_lim**2).sum(axis=1) - 2 * wh0 - 2 * wHbb)
    return 20 * log10(Rlm / (L - 1) / sqrt(mse))


def float_tap_cols(
    n_fix: int, Nw: int, n_groups: int, n_taps: int, foms: Callable[[NDArray[np.int_]], Rvec]
) -> list[int]:
    """
    Choose the Rx FFE's floating tap positions, as MATLAB COM's ``FOM_rxffe_floating_taps()``: one group at a time,
    each placed where it gives the best FOM, along w/ the groups already placed, without overlapping them (178A.1.8.1).

    Args:
        n_fix: Number of fixed-position taps (``Nfix``), which are taps ``0 .. n_fix - 1``.
        Nw: Full span of the FFE (``Nmax``): floating taps may occupy positions ``n_fix .. Nw - 1``.
        n_groups: Number of floating tap groups (``Nwg``).
        n_taps: Number of contiguous taps per group (``Nwf``).
        foms: The FOM given by each of several sets of FFE tap positions (the rows of its argument).

    Returns:
        The FFE tap positions: the fixed ones, followed by the floating ones, in order.
    """
    chosen: list[int] = []
    for _ in range(n_groups):
        starts = [start for start in range(n_fix, Nw - n_taps + 1)
                  if not any(col in chosen for col in range(start, start + n_taps))]
        if not starts:  # No room for another group.
            break
        cands = array([list(range(n_fix)) + sorted(chosen + list(range(start, start + n_taps))) for start in starts])
        start = starts[int(argmax(foms(cands)))]  # (The earliest, on a tie, as MATLAB's `max()`.)
        chosen += list(range(start, start + n_taps))
    return list(range(n_fix)) + sorted(chosen)


def mmse(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements,too-many-branches
    theNoiseCalc: NoiseCalc, Nw: int, dw: int, Nb: int, Rlm: float, L: int,
    b_min: Rvec, b_max: Rvec, w_min: Rvec, w_max: Rvec,
    ts_sweep: float = 0.5, norm_mode: NormMode = NormMode.P8023dj,
    ts_window: int = 0, ts_center: int = 0,
    n_fix: Optional[int] = None, n_float_groups: int = 0, n_float_taps: int = 0
) -> dict[str, Any]:
    """
    Optimize Rx FFE tap weights, via *Minimum Mean Squared Error* (MMSE).

    Args:
        theNoiseCalc: Initialized instance of ``NoiseCalc`` class.
        Nw: Number of taps in Rx FFE.
        dw: Number of pre-cursor taps in the Rx FFE.
        Nb: Number of DFE taps.
        Rlm: Relative level mismatch.
        L: Number of modulation levels.
        b_min: Minimum DFE tap weight values.
        b_max: Maximum DFE tap weight values.
        w_min: Minimum FFE tap weight values.
        w_max: Maximum FFE tap weight values.

    Keyword Args:
        ts_sweep: The cursor sampling time "search radius" around the peak pulse response amplitude (UI).
            Default: 0.5 (i.e. - ``ts`` within [-UI/2, +UI/2] of peak location)
        norm_mode: The tap weight normalization mode to use.
            Default: ``P8023dj``
        ts_window: Cursor timing search window (vector samples); 0 = full search. (See Note 2.)
            Default: 0
        ts_center: Cursor timing search starting point, relative to the peak (vector samples). (See Note 2.)
            Default: 0
        n_fix: Number of fixed-position Rx FFE taps (``Nfix``); the rest of the ``Nw`` (= ``Nmax``) positions
            are for floating taps. (See Note 3.)
            Default: None (``Nw``, i.e. - no floating taps)
        n_float_groups: Number of floating tap groups (``Nwg``).
            Default: 0
        n_float_taps: Number of taps per floating tap group (``Nwf``).
            Default: 0

    Returns:
        A dictionary containing

            * the optimized FFE tap weights, in the "rx_taps" key,
            * the optimized DFE tap weights, in the "dfe_tap_weights" key, and
            * several other values of general utility for plotting/analysis/debugging.

    Raises:
        ValueError: If ``dw`` > ``Nw``, or there is insufficient room at the beginning/end \
        of the pulse response vector for ``ts`` sweeping.

    Notes:
        1. The optimization technique encoded here is taken from the following references

            [1] Healey, A., Hegde, R., *Reference receiver framework for 200G/lane electrical interfaces and PHYs*,
            IEEE P802.3dj Task Force, Jan. 2024

            [2] D1.2 of P802.3dj, IEEE P802.3dj Task Force, Aug. 2024

        2. With ``ts_window`` > 0, the cursor timing search works outward from ``ts_center``,
        as MATLAB COM's "middle" search (``TS_SRCH_MODE``) does from the peak, with ``LOCAL_SEARCH`` = ``ts_window``:
        offsets are tried in order of their distance from ``ts_center``, and one that's ``ts_window``
        or more samples beyond the best so far on its side of ``ts_center`` is skipped.
        Otherwise, every offset in the search radius is tried.
        The number of offsets tried is returned in the "n_ts_evals" key.

        3. With floating taps, their positions are chosen at each sampling time (``float_tap_cols()``), and the
        returned "rx_taps" span all ``Nw`` positions, w/ zeros at those unused. Their positions are returned in
        the "rx_tap_cols" key. (``w_min``/``w_max`` also span all ``Nw`` positions.)
    """

    assert Nw == 0 or Nw > dw, ValueError(
        f"`Nw` ({Nw}) must be greater than `dw` ({dw})!")

    # Initialize certain things, from the given `NoiseCalc` instance.
    nspui       = theNoiseCalc.nspui
    vic_pr      = theNoiseCalc.vic_pulse_resp
    curs_ix     = int(argmax(vic_pr))
    ts_sweep_ix = int(ts_sweep * nspui)

    # Confirm sufficient room for `ts` sweeping.
    assert curs_ix >= ts_sweep_ix, ValueError(
        f"Insufficient room at beginning of pulse response: {curs_ix}!\nvic_pr: {vic_pr}")
    assert curs_ix < len(vic_pr) - ts_sweep_ix, ValueError(
        f"Insufficient room at end of pulse response: {len(vic_pr) - curs_ix}!")

    # Initialize and run the search for optimum `ts` and Rx FFE tap weights.
    max_fom = -1000
    rslt = {}
    varX = theNoiseCalc.varX
    _n_fix = Nw if n_fix is None else n_fix
    if Nw:
        ts_ofsts = list(range(-ts_sweep_ix, ts_sweep_ix + 1))  # Inclusive, as MATLAB COM's `sample_adjustment`.
        if ts_window:  # Work outward from `ts_center` (a stable sort, like MATLAB's).
            ts_center = min(max(ts_center, ts_ofsts[0]), ts_ofsts[-1])
            ts_ofsts.sort(key=lambda k: abs(k - ts_center))
        side_best: dict[int, tuple[float, int]] = {}  # best (FOM, offset) on each side of `ts_center`
        n_ts_evals = 0
        for ts_ofst in ts_ofsts:
            if ts_window:
                sides = [side for side in (-1, 1) if (ts_ofst - ts_center) * side >= 0]
                if any(side in side_best and abs(side_best[side][1] - ts_ofst) >= ts_window for side in sides):
                    continue
            n_ts_evals += 1
            ts_ix = curs_ix + ts_ofst
            theNoiseCalc.ts_ix = ts_ix
            # Keep every pre-cursor sample, as D3.3 (178A-32) and MATLAB COM's `MMSE()` do,
            # so that early pre-cursor ISI contributes to the MSE.
            dh, first_samp = divmod(ts_ix, nspui)  # `dh`: number of pre-cursor samples in `h`
            h = vic_pr[first_samp::nspui].copy()
            d = dh + dw
            first_col = concatenate((h, zeros(Nw - 1)))
            H_all = convolution_matrix(first_col, Nw, mode='full')[:len(first_col)]
            Rn = theNoiseCalc.Rn()[:Nw]
            R_all = H_all.T @ H_all + toeplitz(Rn) / varX

            if _n_fix < Nw and n_float_groups:
                cols = float_tap_cols(
                    _n_fix, Nw, n_float_groups, n_float_taps,
                    partial(_mmse_fom_batch, H_all, R_all, d=d, Nb=Nb, dw=dw, b_min=b_min, b_max=b_max,
                            w_min=w_min, w_max=w_max, norm_mode=norm_mode, varX=varX, Rlm=Rlm, L=L))
            else:
                cols = list(range(_n_fix))
            sol = _mmse_solve(H_all[:, cols], R_all[ix_(cols, cols)], d, Nb, dw,
                              b_min, b_max, w_min[cols], w_max[cols], norm_mode, varX, Rlm, L)
            fom = sol["fom"]
            H = H_all[:, cols]
            w_lim = sol["w_lim"]
            b_lim = sol["b_lim"]
            w_full = zeros(Nw)
            w_full[cols] = w_lim

            # Calculate residual ISI.
            hISI = concatenate((array([sol["h0"] @ w_lim - 1]), sol["Hb"] @ w_lim, H[d + 1 + Nb:] @ w_lim))
            varISI = varX * (hISI**2).sum()                                                         # (93A-31)

            if ts_window:
                for side in sides:
                    if side not in side_best or fom > side_best[side][0]:
                        side_best[side] = (fom, ts_ofst)
            if fom > max_fom:
                max_fom = fom
                rslt["fom"] = fom
                rslt["mse"] = sol["mse"]
                rslt["lambda"] = sol["lam"]  # Should be `mse / varX`.
                rslt["delta_w"] = sol["delta_w"]
                rslt["hit_b_limit"] = sol["hit_b_limit"]
                rslt["rx_taps"] = w_full
                rslt["rx_tap_cols"] = cols
                rslt["dfe_tap_weights"] = b_lim
                rslt["vic_pulse_resp"] = vic_pr  # Note: Does not include Rx FFE/DFE!
                rslt["cursor_ix"] = ts_ix
                rslt["varTx"] = theNoiseCalc.variance(theNoiseCalc.Stn(calc_Hffe(theNoiseCalc.f, theNoiseCalc.Tb, w_full, Nw - dw - 1, hasCurs=True)))
                rslt["varISI"] = varISI
                rslt["varJ"] = theNoiseCalc.variance(theNoiseCalc.Sjn)
                rslt["varXT"] = sum([theNoiseCalc.variance(theNoiseCalc.Sxn(pr)) for pr in theNoiseCalc.agg_pulse_resps])
                rslt["varN"] = theNoiseCalc.variance(theNoiseCalc.Srn)
                rslt["quant_step"] = theNoiseCalc.quant_step_size  # From `Rn()`, above.
                # DEBUGGING:
                rslt["h"] = h
                rslt["h0"] = sol["h0"]
                rslt["H"] = H
                rslt["d"] = d
                rslt["R"] = R_all[ix_(cols, cols)]
                rslt["A"] = sol["A"]
                rslt["y"] = sol["y"]
                rslt["x"] = sol["x"]
                rslt["b"] = sol["b"]
                rslt["theNoiseCalc"] = theNoiseCalc
                rslt["varX"] = varX
                rslt["Rn"] = Rn
        rslt["n_ts_evals"] = n_ts_evals
        theNoiseCalc.ts_ix = rslt["cursor_ix"]  # Leave the noise calculator at the chosen sampling time.
        # Apply desired normalization to tap weights.
        w = rslt["rx_taps"]
        match norm_mode:
            case NormMode.P8023dj:
                w_lim = w  # Done above as part of optimization loop.
            case NormMode.Scaled:
                w_lim = scale_taps(w, w_min, w_max)
            case NormMode.Unaltered:
                w_lim = w
            case NormMode.UnitDcGain:
                w_lim = w / sum(w)
            case _:
                raise RuntimeError(
                    f"Received unknown normalization mode: {norm_mode}!")
        rslt["rx_taps"] = w_lim
    else:
        theNoiseCalc.ts_ix = int(curs_ix)
        d, first_samp = divmod(int(curs_ix), nspui)
        h = vic_pr[first_samp::nspui]
        H = convolution_matrix(h, 1, mode='full')[:len(h)]
        h0 = H[d]
        Hb = H[d + 1: d + 1 + Nb]
        Rn = theNoiseCalc.Rn()[:1]
        R = H.T @ H + toeplitz(Rn) / varX
        w_lim = array([1.0 / vic_pr[curs_ix]])
        b_lim = maximum(b_min, minimum(b_max, h[d + 1: d + 1 + Nb]))
        mse = varX * (w_lim @ R @ w_lim.T + 1 + b_lim @ b_lim - 2 * w_lim @ h0 - 2 * w_lim @ Hb.T @ b_lim).flatten()[0]
        fom = 20 * log10(Rlm / (L - 1) / sqrt(mse))
        rslt["fom"] = fom
        rslt["mse"] = mse
        rslt["lambda"] = None
        rslt["delta_w"] = None
        rslt["hit_b_limit"] = False
        rslt["rx_taps"] = w_lim
        rslt["dfe_tap_weights"] = b_lim
        rslt["vic_pulse_resp"] = vic_pr  # Note: Does not include Rx FFE/DFE!
        rslt["h"] = h
        rslt["cursor_ix"] = curs_ix
        rslt["varTx"] = theNoiseCalc.variance(theNoiseCalc.Stn())
        rslt["varISI"] = 0
        rslt["varJ"] = theNoiseCalc.variance(theNoiseCalc.Sjn)
        rslt["varXT"] = sum([theNoiseCalc.variance(theNoiseCalc.Sxn(pr)) for pr in theNoiseCalc.agg_pulse_resps])
        rslt["varN"] = theNoiseCalc.variance(theNoiseCalc.Srn)
        rslt["quant_step"] = theNoiseCalc.quant_step_size  # From `Rn()`, above.

    return rslt
