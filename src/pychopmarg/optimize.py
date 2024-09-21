"""
Linear equalization optimizers.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 30, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any
import warnings

import numpy as np
from numpy        import argmax, array, array_equal, concatenate, dot, identity, insert, log10, maximum, minimum, ones, sqrt, zeros
from scipy.linalg import LinAlgWarning, convolution_matrix, solve, toeplitz

from pychopmarg.common  import Rvec
from pychopmarg.noise   import NoiseCalc

def przf(pulse_resp: Rvec, nspui: int, nTaps: int, nPreTaps: int, nDFETaps: int) -> Rvec:
    """
    Optimize FFE tap weights, via _Pulse Response Zero Forcing_ (PRZF).

    Args:
        pulse_resp: The pulse response to be filtered.
        nspui: Number of samples per unit interval.
        nTaps: The total number of FFE filter taps, including the cursor.
        nPreTaps: The number of pre-cursor taps.
        nDFETaps: Number of DFE taps.

    Returns:
        weights: The optimum tap weights.
    """

    assert len(pulse_resp) >= nTaps * nspui, ValueError(
        f"The pulse response length ({len(pulse_resp)}) must be at least: `nspui` ({nspui}) * `nTaps` ({nTaps}) = {nspui * nTaps}!")
    assert nTaps > nPreTaps, ValueError(
        f"`nTaps` ({nTaps}) must be greater than `nPreTaps` ({nPreTaps})!")
    assert (nPreTaps + nDFETaps) < (nTaps - 1), ValueError(
        f"The sum of `nPreTaps` ({nPreTaps}) and `nDFETaps` ({nDFETaps}) must be less than: `nTaps` ({nTaps}) - 1!")

    curs_uis, curs_ofst = divmod(argmax(pulse_resp), nspui)
    pr_samps = pulse_resp[curs_ofst::nspui]
    if curs_uis < nPreTaps:
        pr_samps = pad(pr_samps, (nPreTaps - curs_uis, 0))
    else:
        pr_samps = pr_samps[curs_uis - nPreTaps:]
    fv = zeros(nTaps)
    fv[nPreTaps: nPreTaps + nDFETaps + 1] = pr_samps[nPreTaps: nPreTaps + nDFETaps + 1]
    vv = convolution_matrix(pr_samps[:nTaps], nTaps, mode='same')
    return solve(vv, fv)


def mmse(theNoiseCalc: NoiseCalc, Nw: int, dw: int, Nb: int, Rlm: float, L: int,
         b_min: Rvec, b_max: Rvec, w_min: Rvec, w_max: Rvec) -> dict[str, Any]:
    """
    Optimize linear equalization, via _Minimum Mean Squared Error_ (MMSE).

    Args:
        theNoiseCalc: Initialized instance of ``NoiseCalc`` class.
        Nw: Number of taps in Rx FFE.
        dw: Number of pre-cursor taps in the Rx FFE.
        Nb: Number of DFE taps.
        Rlm: Relative level mismatch.
        L: Number of modulation levels.

    Notes:
        1. The optimization technique encoded here is taken from the following reference:
            [1] Healey, A., Hegde, R., _Reference receiver framework for 200G/lane electrical interfaces and PHYs_, IEEE P802.3dj Task Force, Jan. 2024

    """

    vic_pr = theNoiseCalc.vic_pulse_resp
    nspui  = theNoiseCalc.nspui

    max_fom = -1000
    ts_ix_best = 0
    rslt = {}
    max_ix = argmax(vic_pr)
    half_UI = int(nspui // 2)
    for ts_ix in range(max(0, max_ix - half_UI), min(len(vic_pr), max_ix + half_UI)):
        theNoiseCalc.ts_ix = ts_ix
        h = vic_pr[ts_ix % nspui:: nspui]
        d = dw + len(h)
        H = toeplitz(concatenate((h, zeros(Nw - 1))), insert(zeros(Nw - 1), 0, h[0]))
        h0 = H[d + 1]
        Hb = H[d + 2: d + Nb + 2]
        R = H.T @ H + toeplitz(theNoiseCalc.Rn(theNoiseCalc.agg_pulse_resps)[:Nw]) / theNoiseCalc.varX
        Ib = identity(Nb)
        zb = zeros(Nb)
        A = concatenate((concatenate((R, -Hb.T, -h0.reshape((Nw, 1))), axis=1),
                         concatenate((-Hb, ones((Nb, 1)), zeros((Nb, 1))), axis=1),
                         concatenate((h0, zeros(2))).reshape((1, Nw + 2))))
        y = concatenate((h0, zeros(1), ones(1)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = solve(A, y)
        w = x[:Nw]
        b = x[Nw: Nw + Nb]
        b_lim = maximum(b_min, minimum(b_max, b))
        if not array_equal(b_lim, b):
            _A = concatenate((concatenate((R, -h0.reshape((Nw, 1))), axis=1),
                              concatenate((h0, zeros(1))).reshape((1, -1))))
            _y = concatenate((h0 + Hb.T @ b_lim, ones(1)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _x = solve(_A, _y)
            w = _x[:Nw]
        w_lim = maximum(w_min, minimum(w_max, w))
        if not array_equal(w_lim, w):
            w_lim /= dot(h0[:Nw], w_lim.flatten())
            b = Hb @ w_lim.flatten()
            b_lim = maximum(b_min, minimum(b_max, b))
        mse = theNoiseCalc.varX * (w_lim @ R @ w_lim.T + 1 + b_lim @ b_lim - 2 * w_lim @ h0 - 2 * w_lim @ Hb.T @ b_lim).flatten()[0]
        fom = 20 * log10(Rlm / (L - 1) / sqrt(mse.flatten()[0]))
        if fom > max_fom:
            rslt["ts_ix"] = ts_ix
            rslt["fom"] = fom
            rslt["w"] = w_lim
            rslt["b"] = b_lim
            rslt["mse"] = mse
            rslt["rx_taps"] = w_lim
            rslt["dfe_tap_weights"] = b_lim
            rslt["cursor_ix"] = ts_ix
            rslt["As"] = vic_pr[ts_ix]
            df = theNoiseCalc.f[1] - theNoiseCalc.f[0]
            # rslt["varTx"] = sum(theNoiseCalc.Stn(Av, snr_tx) * df)
            rslt["varTx"] = 0
            rslt["varISI"] = 0
            rslt["varJ"] = 0
            rslt["varXT"] = 0
            rslt["varN"] = 0
            rslt["vic_pulse_resp"] = vic_pr
            max_fom = fom

    return rslt
