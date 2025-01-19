"""
Filtering utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

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
    return 1 / (1 - sum(list(map(lambda n_b: n_b[1] * np.exp(-1j * TWOPI * (n_b[0] + 1) * td * freqs),
                                 enumerate(bs)))))


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
    _s2p = s2p.extrapolate_to_dc().interpolate(
        freqs[freqs <= s2p.f[-1]], kind='cubic', coords='polar', basis='t', assume_sorted=True)
    pad_len = len(freqs) - len(_s2p.f)
    s11 = np.pad(_s2p.s11.s.flatten(),                (0, pad_len), mode='edge')
    s12 = np.pad(raised_cosine(_s2p.s12.s.flatten()), (0, pad_len), mode='edge')
    s21 = np.pad(raised_cosine(_s2p.s21.s.flatten()), (0, pad_len), mode='edge')
    s22 = np.pad(_s2p.s22.s.flatten(),                (0, pad_len), mode='edge')
    dS = s11 * s22 - s12 * s21
    return (s21 * (1 - g1) * (1 + g2)) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * dS)
