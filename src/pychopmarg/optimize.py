"""
Linear equalization optimizers.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 30, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any, Optional
import warnings

import numpy as np
from numpy        import(
    argmax, array, array_equal, concatenate, dot, identity, insert,
    log10, maximum, minimum, ones, pad, sqrt, sum, vectorize, where, zeros)
from numpy.linalg import matrix_rank, lstsq
from scipy.linalg import LinAlgWarning, convolution_matrix, solve, solve_toeplitz, toeplitz

from pychopmarg.common  import Rvec
from pychopmarg.noise   import NoiseCalc


def scale_taps(w: Rvec, w_min: Optional[Rvec] = None, w_max: Optional[Rvec] = None) -> Rvec:
    """
    Scale tap weights proportionally, to just fit inside the given min/max limits.

    Args:
        w: The tap weights to scale.

    Keyword Args:
        w_min: Minimum tap weights.
            Default: None (Use `-ones(len(w))`.)
        w_max: Maximum tap weights.
            Default: None (Use `ones(len(w))`.)

    Returns:
        w_lim: Tap weights, scaled proportionally, to just fit inside min/max limits.
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
    w_min: Optional[Rvec] = None, w_max: Optional[Rvec] = None
) -> Rvec:
    """
    Clip tap weights to the given min/max limits, as per (178A-26).

    Args:
        w: The tap weights to clip.
        curs_ix: The index, in `w`, of the cursor tap.

    Keyword Args:
        w_min: Minimum tap weights.
            Default: None (Use `-ones(len(w))`.)
        w_max: Maximum tap weights.
            Default: None (Use `ones(len(w))`.)

    Returns:
        w_lim: Tap weights, clipped accordingly.
    """

    if w_min is None:
        w_min = -ones(len(w))
    if w_max is None:
        w_max = ones(len(w))

    assert len(w) == len(w_min) == len(w_max), ValueError(
        f"Lengths of: `w` ({len(w)}), `w_min` ({len(w_min)}), and `w_max` ({len(w_max)}), must be equal!")

    w_lim_curs_val = min(w_max[curs_ix], max(w_min[curs_ix], w[curs_ix]))
    w_lim = minimum(w_max * w_lim_curs_val, maximum(w_min * w_lim_curs_val, w))
    w_lim[curs_ix] = w_lim_curs_val

    return w_lim


def przf(
    pulse_resp: Rvec, nspui: int, nTaps: int, nPreTaps: int, nDFETaps: int,
    tap_mins: Rvec, tap_maxs: Rvec, b_min: Rvec, b_max: Rvec
) -> Rvec:
    """
    Optimize FFE tap weights, via _Pulse Response Zero Forcing_ (PRZF).

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

    Returns:
        A triple consisting of:
            - FFE tap weights: The optimum FFE tap weights.
            - DFE tap weights: The optimum DFE tap weights.
            - pr_samps: The pulse response samples used in optimization.

    Notes:
        1. The algorithm implemented below is a slightly modified version of:
            Mellitz, R., Lusted, K., "RX FFE Implementation Algorithm for COM 4.1", IEEE P802.3dj Task Force, August 31, 2023.

    ToDo:
        1. Add DFE tap weight max limiting.
        2. Add sampling time as an input parameter?
    """

    assert len(pulse_resp) >= nTaps * nspui, ValueError(
        f"The pulse response length ({len(pulse_resp)}) must be at least: `nspui` ({nspui}) * `nTaps` ({nTaps}) = {nspui * nTaps}!")
    assert nTaps > nPreTaps, ValueError(
        f"`nTaps` ({nTaps}) must be greater than `nPreTaps` ({nPreTaps})!")
    assert (nPreTaps + nDFETaps) < (nTaps - 1), ValueError(
        f"The sum of `nPreTaps` ({nPreTaps}) and `nDFETaps` ({nDFETaps}) must be less than: `nTaps` ({nTaps}) - 1!")

    # Sample the given pulse response, assuming cursor coincides w/ maximum.
    dh, first_samp = divmod(argmax(pulse_resp), nspui)
    h = pulse_resp[first_samp::nspui]
    len_h = len(h)
    d = dh + nPreTaps

    # Create the appropriate forcing vector.
    fv = zeros(len_h)
    no_force_ixs = slice(dh, dh + nDFETaps + 1)  # indices of cursor and DFE taps
    fv[no_force_ixs] = h[no_force_ixs]           # Don't force the cursor, or DFE taps, to zero.
    fv = pad(fv, (nPreTaps, 0))[:len_h]          # Adding expected delay, `dw`, due to Rx FFE pre-cursor taps.

    # Find the optimum FFE tap weights, enforcing min/max limits and scaling for unit pulse response amplitude.
    H = convolution_matrix(h, nTaps, mode='full')[:len_h]
    h0 = H[d]
    Hb = H[d + 1: d + 1 + nDFETaps]
    try:
        w, _, _, _ = lstsq(H, fv, rcond=None)
        wn = clip_taps(w, nPreTaps, tap_mins, tap_maxs)
    except Exception:
        print(f"H.shape: {H.shape}; fv.shape: {fv.shape}; w.shape: {w.shape}")
        raise
    wn /= dot(h0, wn)

    # Calculate desired DFE tap weight values and enforce limits.
    bn = Hb @ wn.flatten()
    bn = minimum(b_max, maximum(b_min, bn))

    return wn, bn, h[dh - nPreTaps: dh - nPreTaps + nTaps]


def mmse(theNoiseCalc: NoiseCalc, Nw: int, dw: int, Nb: int, Rlm: float, L: int,
         b_min: Rvec, b_max: Rvec, w_min: Rvec, w_max: Rvec, ts_sweep: float = 0.5) -> dict[str, Any]:
    """
    Optimize linear equalization, via _Minimum Mean Squared Error_ (MMSE).

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
            Default: 0.5 (i.e. - `ts` within [-UI/2, +UI/2] of peak location)

    Returns:
        A dictionary containing:
            - the optimized FFE tap weights, in the "rx_taps" key,
            - the optimized DFE tap weights, in the "dfe_tap_weights" key, and
            - several other values of general utility for plotting/analysis/debugging.

    Raises:
        ValueError if:
            - `dw` > `Nw`, or
            - there is insufficient room at the beginning/end of the pulse response vector, for `ts` sweeping.

    Notes:
        1. The optimization technique encoded here is taken from the following references:
            [1] Healey, A., Hegde, R., _Reference receiver framework for 200G/lane electrical interfaces and PHYs_, IEEE P802.3dj Task Force, Jan. 2024
            [2] D1.1 of P802.3dj, IEEE P802.3dj Task Force, Aug. 2024
    """

    assert Nw > dw, ValueError(
        f"`Nw` ({Nw}) must be greater than `dw` ({dw})!")

    # Initialize certain things, from the given `NoiseCalc` instance.
    nspui       = theNoiseCalc.nspui
    vic_pr      = theNoiseCalc.vic_pulse_resp
    curs_ix     = argmax(vic_pr)
    ts_sweep_ix = int(ts_sweep * nspui)

    # Confirm sufficient room for `ts` sweeping.
    assert curs_ix >= ts_sweep_ix, ValueError(
        f"Insufficient room at beginning of pulse response: {curs_ix}!")
    assert curs_ix <= len(vic_pr) - ts_sweep_ix, ValueError(
        f"Insufficient room at end of pulse response: {len(vic_pr) - curs_ix}!")

    # Initialize and run the search for optimum `ts` and Rx FFE tap weights.
    max_fom = -1000
    ts_ix_best = 0
    rslt = {}
    for ts_ix in range(curs_ix - ts_sweep_ix, curs_ix + ts_sweep_ix):
        theNoiseCalc.ts_ix = ts_ix
        dh, first_samp = divmod(ts_ix, nspui)
        h = vic_pr[first_samp::nspui]
        d = dw + dh
        H = toeplitz(concatenate((h, zeros(Nw - 1))), insert(zeros(Nw - 1), 0, h[0]))
        h0 = H[d]
        Hb = H[d + 1: d + 1 + Nb]
        varX = theNoiseCalc.varX
        Rn = theNoiseCalc.Rn(theNoiseCalc.agg_pulse_resps)[:Nw]
        R = H.T @ H + toeplitz(Rn) / varX
        Ib = identity(Nb)
        zb = zeros(Nb)

        A = concatenate((concatenate(( R, -Hb.T,         -h0.reshape((Nw, 1))), axis=1),
                         concatenate((-Hb, ones((Nb, 1)), zeros((Nb, 1))),      axis=1),
                         concatenate(( h0, zeros(2))).reshape((1, Nw + 2))))
        y = concatenate((h0, zeros(Nb), ones(1)))
        x = solve(A, y)
        w = x[:Nw]

        # Check DFE tap weights, enforcing limits if necessary.
        b = x[Nw: Nw + Nb]
        b_lim = maximum(b_min, minimum(b_max, b))
        if not array_equal(b_lim, b):
            _A = concatenate((concatenate((R, -h0.reshape((Nw, 1))), axis=1),
                              concatenate((h0, zeros(1))).reshape((1, -1))))
            _y = concatenate((h0 + Hb.T @ b_lim, ones(1)))
            _x = solve(_A, _y)
            w = _x[:Nw]
        
        # Check and enforce tap weight limits.
        w_lim = clip_taps(w, dw, w_min, w_max)
        if not array_equal(w_lim, w):
            w_lim /= dot(h0, w_lim)
            b = Hb @ w_lim.flatten()
            b_lim = maximum(b_min, minimum(b_max, b))

        mse = varX * (w_lim @ R @ w_lim.T + 1 + b_lim @ b_lim - 2 * w_lim @ h0 - 2 * w_lim @ Hb.T @ b_lim).flatten()[0]
        fom = 20 * log10(Rlm / (L - 1) / sqrt(mse))
        if fom > max_fom:
            max_fom = fom
            rslt["fom"] = fom
            rslt["mse"] = mse
            rslt["rx_taps"] = w_lim
            rslt["dfe_tap_weights"] = b_lim
            rslt["vic_pulse_resp"] = vic_pr
            rslt["cursor_ix"] = ts_ix
            df = theNoiseCalc.f[1] - theNoiseCalc.f[0]
            rslt["varTx"] = sum(theNoiseCalc.Stn) * df
            rslt["varISI"] = 0
            rslt["varJ"] = sum(theNoiseCalc.Sjn) * df
            rslt["varXT"] = sum(sum(array(list(map(theNoiseCalc.Sxn, theNoiseCalc.agg_pulse_resps))), axis=0)) * df
            rslt["varN"] = 0
            # DEBUGGING:
            rslt["h"] = h
            rslt["h0"] = h0
            rslt["H"] = H
            rslt["d"] = d
            rslt["R"] = R
            rslt["A"] = A
            rslt["y"] = y
            rslt["x"] = x
            # rslt["_A"] = _A
            # rslt["_y"] = _y
            # rslt["_x"] = _x
            rslt["b"] = b
            rslt["theNoiseCalc"] = theNoiseCalc
            rslt["varX"] = varX
            rslt["Rn"] = Rn

    return rslt