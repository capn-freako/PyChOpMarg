"""
Linear equalization optimizers.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 30, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any

from numpy        import argmax, array, concatenate, identity, insert, zeros
from scipy.linalg import solve, toeplitz

from pychopmarg.noise import NoiseCalc


def przf():
    """
    Optimize linear equalization, via _Pulse Response Zero Forcing_ (PRZF).
    """
    pass


def mmse(theNoiseCalc: NoiseCalc, Nw: int, dw: int, Nb: int) -> dict[str, Any]:
    """
    Optimize linear equalization, via _Minimum Mean Squared Error_ (MMSE).

    Args:
        theNoiseCalc: Initialized instance of ``NoiseCalc`` class.
        Nw: Number of taps in Rx FFE.
        dw: Number of pre-cursor taps in the Rx FFE.
        Nb: Number of DFE taps.

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
        try:
            h0 = H[d + 1]
        except Exception:
            print(f"H.shape: {H.shape}; d: {d}; Nw: {Nw}; dw: {dw}")
            raise
        Hb = H[d + 2: d + Nb + 2]
        R = H.T @ H + theNoiseCalc.Rnn(theNoiseCalc.agg_pulse_resps) / theNoiseCalc.varX
        Ib = identity(Nb)
        zb = zeros(Nb)
        A = array([[ R, -Hb.T, -h0.T],
                   [-Hb, Ib,    zb.T],
                   [ h0, zb,    0]])
        y = array([h0.T, zb.T, 1]).T
        w, b, _ = solve(A, y)
        b_lim = max(b_min, min(b_max, b))
        if b_lim != b:
            _A = array([[R, -h0.T],
                        [h0, 0]])
            _y = array([h0.T + Hb.T @ b_lim]).T
            w, _ = solve(_A, _y)
        w_lim = max(w_min, min(w_max, w))
        if w_lim != w:
            w_lim /= h0 @ w_lim
            b = Hb @ w_lim
            b_lim = max(b_min, min(b_max, b))
        mse = varX * (w_lim.T @ R @ w_lim + 1 + b_lim.T @ b_lim - 2 * w_lim.T @ h0.T - 2 * w_lim.T @ Hb.T @ b_lim)
        fom = 20 * log10(Rlm / (L - 1) / sqrt(mse))
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
