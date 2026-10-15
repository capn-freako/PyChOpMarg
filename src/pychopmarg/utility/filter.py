"""
Filtering utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import cast
import warnings

import numpy as np  # type: ignore
import skrf  as rf  # type: ignore

from pychopmarg.common import Rvec, Cvec, PI, TWOPI


def from_dB(x: float) -> float:
    """Convert from (dB) to real, assuming square law applies."""
    return pow(10, x / 20)


def calc_Hctle(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    f: Rvec, fz: float, fp1: float, fp2: float, fLF: float, gDC: float, gDC2: float
) -> Cvec:
    """
    Return the voltage transfer function, H(f), of the Rx CTLE,
    according to (93A-22).

    Args:
        f: Frequencies at which to calculate ``Hctle`` (Hz).
        fz: First stage zero frequency (Hz).
        fp1: First stage lower pole frequency (Hz).
        fp2: First stage upper pole frequency (Hz).
        fLF: Second stage pole/zero frequency (Hz).
        gDC: First stage d.c. gain (dB).
        gDC2: Second stage d.c. gain (dB).

    Returns:
        The complex voltage transfer function, H(f), for the CTLE.
    """
    g1 = from_dB(gDC)
    g2 = from_dB(gDC2)
    num = (g1 + 1j * f / fz) * (g2 + 1j * f / fLF)
    den = (1 + 1j * f / fp1) * (1 + 1j * f / fp2) * (1 + 1j * f / fLF)
    return num / den


def calc_Hffe(
    freqs: Rvec, td: float,
    tap_weights: Rvec, n_post: int,
    hasCurs: bool = False
) -> Cvec:
    """
    Calculate the voltage transfer function, H(f), for a digital FFE, according to (93A-21).

    Args:
        freqs: Frequencies at which to calculate ``Hffe`` (Hz).
        td: Tap delay time (s).
        tap_weights: The filter tap weights.
        n_post: The number of post-cursor taps.

    Keyword Args:
        hasCurs: ``tap_weights`` includes the cursor tap weight when True.
            Default: False (Cursor tap weight will be calculated.)

    Returns:
        The complex voltage transfer function, H(f), for the FFE.

    The simple expression being returned is defended, as follows:

    1. Take axiomatically that what we want to return is the sum of the rows of the following matrix:

        .. math::
            \\begin{bmatrix}
                b_0 e^{-j 2 \\pi T 0 f_0} & b_0 e^{-j 2 \\pi T 0 f_1} & b0 e^{-j 2 \\pi T 0 f_2} ... \\\\
                b_1 e^{-j 2 \\pi T 1 f_0} & b_1 e^{-j 2 \\pi T 1 f_1} & b1 e^{-j 2 \\pi T 1 f_2} ... \\\\
                b_2 e^{-j 2 \\pi T 2 f_0} & b_2 e^{-j 2 \\pi T 2 f_1} & b2 e^{-j 2 \\pi T 2 f_2} ... \\\\
                \\vdots
            \\end{bmatrix}

    2. Now, note that each columnar sum is a dot product of the vectors:

        - :math:`\\{b_n\\}`, and

        - :math:`\\{e^{-j 2 \\pi n T \\cdot f_m}\\}`, where:

        - :math:`f_m = m \\Delta f = \\frac{m}{NT}`, giving:

        .. math::
            H(f) = [ b_0, b_1, b_2, ... ] e^{-j 2 \\pi T \\mathbf{F}}

        where:

        .. math::
            \\mathbf{F} = \\begin{bmatrix}
                f0 \\cdot 0 & f1 \\cdot 0 & f2 \\cdot 0 ... \\\\
                f0 \\cdot 1 & f1 \\cdot 1 & f2 \\cdot 1 ... \\\\
                f0 \\cdot 2 & f1 \\cdot 2 & f2 \\cdot 2 ... \\\\
                \\vdots
            \\end{bmatrix} =
            \\begin{bmatrix}
                0 \\\\
                1 \\\\
                2 \\\\
                \\vdots
            \\end{bmatrix} [f_0, f_1, f_2, ...] = \\mathbf{n}^T \\mathbf{f}

        giving:

        .. math::
            H(f) = \\mathbf{b} e^{-j 2 \\pi T \\mathbf{n}^T \\mathbf{f}}

    3. Finally, comparing the final expression above to the Python code reveals a match:

        .. code-block:: python

            return bs @ np.exp(np.outer(np.arange(len(bs)), -1j * TWOPI * td * freqs))

    Note that **F** may be pre-calculated, and needn't be recalculated,
    once the system time/frequency vectors have been established.
    Doing so yields a significant performance improvement in cases with many Tx FFE combinations.
    """

    bs = tap_weights
    if not hasCurs:
        b0 = 1 - abs(tap_weights).sum()
        bs = np.insert(bs, -n_post, b0)
    return bs @ np.exp(np.outer(np.arange(len(bs)), -1j * TWOPI * td * freqs))  # 50% perf. improvement


def calc_Hdfe(freqs: Rvec, td: float, tap_weights: Rvec) -> Cvec:
    """
    Calculate the voltage transfer function, H(f), for a *Decision Feedback Equalizer* (DFE).

    Args:
        freqs: Frequencies at which to calculate ``Hdfe`` (Hz).
        td: Tap delay time (s).
        tap_weights: The vector of filter tap weights.

    Returns:
        The complex voltage transfer function, H(f), for the DFE.
    """

    bs = tap_weights.flatten()
    return 1 / (1 - sum(b * np.exp(-1j * TWOPI * (n + 1) * td * freqs) for n, b in enumerate(bs)))


def null_filter(nTaps: int, nPreTaps: int = 0) -> Rvec:
    """
    Construct a null filter w/ ``nTaps`` taps and (optionally) ``nPreTaps`` pre-cursor taps.

    Args:
        nTaps: Total number of taps, including the cursor tap.

    Keyword Args:
        nPreTaps: Number of pre-cursor taps.
            Default: 0

    Returns:
        The filter tap weight vector, including the cursor tap weight.
    """

    assert nTaps > 0, ValueError(
        f"`nTaps` ({nTaps}) must be greater than zero!")
    assert nPreTaps < nTaps, ValueError(
        f"`nPreTaps` ({nPreTaps}) must be less than `nTaps` ({nTaps})!")

    taps = np.zeros(nTaps)
    taps[nPreTaps] = 1.0

    return taps


def raised_cosine(x: Cvec) -> Cvec:
    "Apply raised cosine window to input."
    len_x = len(x)
    w = (np.array([np.cos(PI * n / len_x) for n in range(len_x)]) + 1) / 2
    return w * x


def calc_H21(freqs: Rvec, s2p: rf.Network, g1: float, g2: float) -> Cvec:
    """
    Return the voltage transfer function, H21(f), of a terminated two
    port network, according to (93A-18).

    Args:
        freqs: Frequencies at which to calculate the response (Hz).
        s2p: Two port network of interest.
        g1: Reflection coefficient looking out of the left end of the channel.
        g2: Reflection coefficient looking out of the right end of the channel.

    Returns:
        Complex voltage transfer function at given frequencies.

    Raises:
        ValueError: If given network is not two port.
    """

    assert s2p.s[0].shape == (2, 2), ValueError("Network must be 2-port!")
    _s2p: rf.Network = cast(rf.Network, s2p.extrapolate_to_dc().interpolate(
        freqs[freqs <= s2p.f[-1]], kind='cubic', coords='polar', basis='t', assume_sorted=True))
    pad_len = len(freqs) - len(_s2p.f)
    s11 = np.pad(_s2p.s11.s.flatten(),                (0, pad_len), mode='edge')
    s12 = np.pad(raised_cosine(_s2p.s12.s.flatten()), (0, pad_len), mode='edge')
    s21 = np.pad(raised_cosine(_s2p.s21.s.flatten()), (0, pad_len), mode='edge')
    s22 = np.pad(_s2p.s22.s.flatten(),                (0, pad_len), mode='edge')
    dS = s11 * s22 - s12 * s21
    return (s21 * (1 - g1) * (1 + g2)) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * dS)


def calc_vtf(s2p: rf.Network, g1: float, g2: float) -> Cvec:
    """
    Return the voltage transfer function of a terminated two port network, according to (93A-18),
    at the network's own frequencies (i.e. - w/o interpolation, unlike ``calc_H21()``).

    Args:
        s2p: Two port network of interest.
        g1: Reflection coefficient looking out of the left end of the channel.
        g2: Reflection coefficient looking out of the right end of the channel.

    Returns:
        Complex voltage transfer function at ``s2p.f``.
    """
    assert s2p.s[0].shape == (2, 2), ValueError("Network must be 2-port!")
    s11, s12, s21, s22 = s2p.s[:, 0, 0], s2p.s[:, 0, 1], s2p.s[:, 1, 0], s2p.s[:, 1, 1]
    dS = s11 * s22 - s12 * s21
    return (s21 * (1 - g1) * (1 + g2)) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * dS)


def _interp1_extrap(x: Rvec, y: Rvec, xq: Rvec) -> Rvec:
    "Linear interpolation, w/ linear extrapolation from the end segments (as MATLAB's ``interp1(..., 'extrap')``)."
    yq = np.interp(xq, x, y)
    lo = xq < x[0]
    hi = xq > x[-1]
    yq[lo] = y[0] + (xq[lo] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    yq[hi] = y[-1] + (xq[hi] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return yq


def _trend_mean(x: Rvec) -> float:
    "Mean of ``x``, excluding outliers more than one standard deviation from its median (as MATLAB COM)."
    return float(np.mean(x[abs(x - np.median(x)) < np.std(x, ddof=1)]))


def interp_vtf(H: Cvec, fin: Rvec, fout: Rvec) -> Cvec:
    """
    Interpolate a transfer function onto a new frequency grid, as MATLAB COM's ``interp_Sparam()`` does,
    with its default policies: magnitude ``linear_trend_to_DC`` and phase ``extrap_cubic_to_dc_linear_to_inf``.

    Args:
        H: Transfer function, at ``fin``.
        fin: Its frequencies (Hz), ascending.
        fout: Frequencies to interpolate onto (Hz), ascending from 0.

    Returns:
        ``H`` at ``fout``.

    Notes:
        1. Magnitude is interpolated linearly, extended to d.c. along a linear fit to its first 10 points, and
        above ``fin[-1]`` to the value at ``fout[-1]`` of a linear fit to its upper half (clamped to
        [eps, ``|H(fin[-1])|``]).

        2. Phase (unwrapped) is interpolated linearly. If ``fin[0]`` > 0, its first 10 points are first
        replaced, following the low frequency group delay trend, and it's extended above ``fin[-1]``
        following the high frequency group delay trend; otherwise, it's extrapolated from its end segments.
    """
    eps = np.finfo(float).eps
    fin = np.asarray(fin, dtype=float)
    fout = np.asarray(fout, dtype=float)

    # Magnitude
    H_mag = np.maximum(abs(H), eps)
    fin_x, H_mag_x = fin, H_mag
    if fin[0] > 0:
        p = np.polyfit(fin[:10], H_mag[:10], 1)
        fin_x = np.concatenate(([0.0], fin_x))
        H_mag_x = np.concatenate(([np.polyval(p, 0.0)], H_mag_x))
    if fin[-1] < fout[-1]:
        mid = int(np.floor(len(fin) / 2 + 0.5)) - 1  # MATLAB's `round(length(fin)/2)`, as a 0-based index
        p = np.polyfit(fin[mid:], H_mag[mid:], 1)
        hf_trend_val = min(max(np.polyval(p, fout[-1]), eps), H_mag[-1])
        fin_x = np.concatenate((fin_x, [fout[-1]]))
        H_mag_x = np.concatenate((H_mag_x, [hf_trend_val]))
    H_mag_i = _interp1_extrap(fin_x, H_mag_x, fout)

    # Phase
    H_ph = np.unwrap(np.angle(H))
    if np.mean(np.diff(H_ph)) > 0:
        warnings.warn("Anti-causal response found; a finer frequency step is required for this channel.",
                      stacklevel=2)
    if fin[0] != 0:
        group_delay = -np.diff(H_ph) / np.diff(fin)
        lf_trend = _trend_mean(group_delay[:50])
        for k in range(9, -1, -1):  # Correct outliers in the first 10 phase samples.
            H_ph[k] = H_ph[k + 1] + lf_trend * (fin[k + 1] - fin[k])
        H_ph_i = _interp1_extrap(fin, H_ph, fout)
        hf_trend = -_trend_mean(group_delay[-51:])
        hf = fout > fin[-1]
        if np.any(hf):
            last = np.argmax(hf) - 1
            H_ph_i[hf] = H_ph_i[last] + (fout[hf] - fout[last]) * hf_trend
    else:
        H_ph_i = _interp1_extrap(fin, H_ph, fout)

    return H_mag_i * np.exp(1j * H_ph_i)
