"""
Linear equalization optimizers.

Original author: David Banas <capn.freako@gmail.com>

Original date:   August 30, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from numpy        import argmax, array, identity, zeros
from scipy.linalg import solve

from pychopmarg.noise import NoiseCalc


def przf():
    """
    Optimize linear equalization, via _Pulse Response Zero Forcing_ (PRZF).
    """
    pass


def mmse(theNoiseCalc: NoiseCalc, Nw: int, dw: int, Nb: int):
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

    vic_pr = theNoiseCalc.vic_pulse_response
    nspui  = theNoiseCalc.nspui

    max_fom = -1000
    ts_ix_best = 0
    max_ix = argmax(vic_pr)
    for ts_ix in range(max(0, max_ix - nspui // 2), min(len(vic_pr), max_ix + nspui // 2)):
        theNoiseCalc.ts_ix = ts_ix
        h = array([ts_ix % nspui:: nspui])
        d = dw + len(h)
        H = toeplitz(concatenate((h, zeros(Nw - 1))), concatenate((h[0], zeros(Nw - 1))))
        h0 = H[d + 1]
        Hb = H[d + 2: d + Nb + 2]
        R = H.T @ H + theNoiseCalc.Rnn / theNoiseCalc.varX
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
            ts_ix_best = ts_ix
            max_fom = fom
            w_best = w_lim
            b_best = b_lim
            mse_best = mse
