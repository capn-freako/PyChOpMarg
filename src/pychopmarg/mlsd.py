"""
Maximum likelihood sequence detection (MLSD) reference receiver, as per P802.3dj/D3.3 178A.1.10.

Original author: David Banas <capn.freako@gmail.com>

Original date:   October 15, 2026

Copyright (c) 2026 David Banas; all rights reserved World wide.

COM for the MLSD-based reference receiver modifies the DFE-based COM (``COM_DFE``, w/ ``Nb`` = 1)
by the advantage the MLSD has over the DFE (178A-49):

    COM = COM_DFE + 20 * log10(-Pan^-1(DER_MLSD) / As)

All probability mass functions (PMFs) here are on uniform, zero-centered voltage grids (odd length).
"""

from typing import Optional

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

from pychopmarg.common import Rvec


def conv_full(p1: Rvec, p2: Rvec) -> Rvec:
    "Full convolution of two zero-centered PMFs (the result is zero-centered too), w/ FFT round-off clipped."
    p = np.maximum(fftconvolve(p1, p2), 0.0)
    return p / p.sum()


def conv_same(p1: Rvec, p2: Rvec) -> Rvec:
    "Convolution of two zero-centered PMFs on the grid of the first, w/ FFT round-off clipped."
    p = np.maximum(fftconvolve(p1, p2, mode='same'), 0.0)
    return p / p.sum()


def trim_pmf(p: Rvec, tol: float = 1e-30) -> Rvec:
    "Trim a zero-centered PMF symmetrically, dropping tails holding less than ``tol`` (keeping it zero-centered)."
    c = np.cumsum(p)
    n_lo = int(np.searchsorted(c, tol))                     # Leading bins holding < `tol`.
    n_hi = int(np.searchsorted(c[-1] - c[::-1], tol))       # Trailing bins holding < `tol`.
    n = min(n_lo, n_hi, (len(p) - 1) // 2)
    return p[n: len(p) - n] if n else p


def scale_pmf(p: Rvec, a: float) -> Rvec:
    """
    PMF of ``a * Y``, for ``Y`` w/ the zero-centered PMF ``p``, on the same grid: ``p(y / a) / |a|``,
    as in 178A-54/55 (the note in 178A.1.10.2). Mass is rebinned through the cumulative distribution
    (linearly interpolated between bin edges), so the result is a PMF.
    """
    n = len(p)
    if a == 0:
        rslt = np.zeros(n)
        rslt[n // 2] = 1.0
        return rslt
    if a < 0:
        return scale_pmf(p[::-1], -a)
    edges = np.arange(n + 1) - n / 2                        # Bin edges, in bins, centered on zero.
    cdf = np.concatenate(([0.0], np.cumsum(p)))
    rslt = np.diff(np.interp(edges / a, edges, cdf, left=0.0, right=cdf[-1]))
    return rslt / rslt.sum()


def gauss_pmf(n: int, ystep: float, var: float) -> Rvec:
    "Zero-mean Gaussian PMF of variance ``var`` on a zero-centered grid of ``n`` (odd) bins of ``ystep``."
    y = (np.arange(n) - n // 2) * ystep
    if var <= 0:
        p = np.zeros(n)
        p[n // 2] = 1.0
        return p
    p = np.exp(-y**2 / (2 * var))
    return p / p.sum()


def inv_cdf(p: Rvec, ystep: float, prob: float) -> float:
    "The value of ``y`` at which the zero-centered PMF ``p``'s cumulative distribution first reaches ``prob``."
    return float((int(np.argmax(np.cumsum(p) >= prob)) - len(p) // 2) * ystep)


def cdf_at(p: Rvec, ystep: float, y: float) -> float:
    """
    Cumulative distribution of the zero-centered PMF ``p`` at ``y``, as MATLAB COM's ``CDF_ev()``:
    its value at the first grid point at or above ``y``.
    """
    n = len(p)
    k = int(np.ceil(y / ystep - 1e-9)) + n // 2
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(np.cumsum(p)[k])


def solve_g_an(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    p_rest: Rvec, ystep: float, var_G: float, var_rn: float, DER0: float, dCOM_an: float,
    rel_tol: float = 1e-6
) -> tuple[float, Rvec]:
    """
    The added noise scale factor ``g_an``, and the noise and interference PMF ``p_an`` it gives,
    as per 178A.1.10.1: ``p_an`` is that of COM_DFE, but w/ the Gaussian noise variance
    ``var_G + g_an * var_rn`` (178A-53), and ``g_an`` makes ``20 * log10(Pan^-1(DER0) / P^-1(DER0))`` = ``dCOM_an``.

    Args:
        p_rest: PMF of every non-Gaussian noise and interference term of COM_DFE (all but ``pG``).
        ystep: Voltage grid step.
        var_G: Variance of COM_DFE's Gaussian noise, ``sigma_G^2``.
        var_rn: Variance of the Rx noise at the FFE output: the integral of ``Srn * |Hrxffe|^2``.
        DER0: Target detector error ratio.
        dCOM_an: Penalty for added receiver noise (dB).

    Keyword Args:
        rel_tol: Relative tolerance on ``g_an``.
            Default: 1e-6

    Returns:
        ``g_an``, and ``p_an`` (on ``p_rest``'s grid).

    Raises:
        ValueError: If no added Rx noise can produce the penalty (e.g. - ``var_rn`` is 0).
    """
    n = len(p_rest)

    def p_an(g: float) -> Rvec:
        return conv_same(p_rest, gauss_pmf(n, ystep, var_G + g * var_rn))

    target = inv_cdf(p_an(0.0), ystep, DER0) * 10**(dCOM_an / 20)  # (Both negative.)
    g_lo, g_hi = 0.0, 1.0
    for _ in range(200):
        if inv_cdf(p_an(g_hi), ystep, DER0) <= target:
            break
        g_lo, g_hi = g_hi, 2 * g_hi
    else:
        raise ValueError(f"No added Rx noise (variance {var_rn:.3g}) gives a {dCOM_an} dB penalty!")
    while g_hi - g_lo > rel_tol * g_hi:
        g = (g_lo + g_hi) / 2
        if inv_cdf(p_an(g), ystep, DER0) <= target:
            g_hi = g
        else:
            g_lo = g
    return g_hi, p_an(g_hi)


def der_mlsd(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    p_an: Rvec, ystep: float, As: float, b: float, rho: Rvec, L: int,
    rel_tol: float = 1e-4, max_j: Optional[int] = None
) -> tuple[float, int]:
    """
    Detector error ratio of the MLSD, ``DER_MLSD``, as per (178A-50), its infinite sum truncated
    (as in MATLAB COM) once a term adds less than ``rel_tol`` of the sum so far.

    Args:
        p_an: Noise and interference PMF, w/ the added Rx noise (178A.1.10.1).
        ystep: Voltage grid step.
        As: Signal amplitude (V).
        b: First DFE tap weight, ``b_lim(1)``.
        rho: Normalized autocorrelation of the noise and residual ISI at the FFE output, ``Rni(n) / Rni(0)``
            (178A.1.10.3), for lags 0 and up.
        L: Number of modulation levels.

    Keyword Args:
        rel_tol: Relative size of the last term kept.
            Default: 1e-4
        max_j: Longest sequence considered.
            Default: None (``len(rho) - 1``)

    Returns:
        ``DER_MLSD``, and the number of terms summed.
    """
    max_j = min(max_j or len(rho) - 1, len(rho) - 1)
    p_1mb = trim_pmf(scale_pmf(p_an, 1 - b))
    p_j = trim_pmf(conv_full(p_an, trim_pmf(scale_pmf(p_an, b))))                         # (178A-54)
    der = 0.0
    j = 0
    for j in range(1, max_j + 1):
        u = np.array([1.0] + [(-1)**(i - 1) * (1 - b) for i in range(2, j + 1)] + [(-1)**(j + 1) * b])  # (178A-51)
        V = toeplitz(rho[: j + 1])
        x = As * (u @ u)**1.5 / np.sqrt(u @ V @ u)
        term = ((L - 1) / L)**(j - 1) * cdf_at(p_j, ystep, -x)                                 # (178A-50)
        der += term
        if term <= rel_tol * der:
            break
        p_j = trim_pmf(conv_full(p_j, p_1mb))                                                   # (178A-55)
    return der, j
