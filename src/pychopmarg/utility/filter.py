"""
Filtering utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024 (Copied from `pybert.utility`.)

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

import numpy as np  # type: ignore

from pychopmarg.common import Rvec, Cvec, TWOPI


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
        f: Frequencies at which to calculate `Hctle(f)` (Hz).
        fz: First stage zero frequency (Hz).
        fp1: First stage lower pole frequency (Hz).
        fp2: First stage upper pole frequency (Hz).
        fLF: Second stage pole/zero frequency (Hz).
        gDC: CTLE first stage d.c. gain (dB).
        gDC2: CTLE second stage d.c. gain (dB).

    Returns:
        The complex voltage transfer function, H(f), for the CTLE.

    Raises:
        None
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
        freqs: Frequencies at which to calculate `Hffe` (Hz).
        td: Tap delay time (s).
        tap_weights: The filter tap weights.
        n_post: The number of post-cursor taps.

    Keyword Args:
        hasCurs: `tap_weights` includes the cursor tap weight when True.
            Default: False (Cursor tap weight will be calculated.)

    Returns:
        The complex voltage transfer function, H(f), for the FFE.

    Raises:
        None
    """

    bs = list(np.array(tap_weights).flatten())
    if not hasCurs:
        b0 = 1 - sum(list(map(abs, tap_weights)))
        bs.insert(-n_post, b0)
    return sum(list(map(lambda n_b: n_b[1] * np.exp(-1j * TWOPI * n_b[0] * td * freqs),
                        enumerate(bs))))


def calc_Hdfe(freqs: Rvec, td: float, tap_weights: Rvec) -> Cvec:
    """
    Calculate the voltage transfer function, H(f), for a _Decision Feedback Equalizer_ (DFE).

    Args:
        freqs: Frequencies at which to calculate `Hdfe` (Hz).
        td: Tap delay time (s).
        tap_weights: The vector of filter tap weights.

    Returns:
        The complex voltage transfer function, H(f), for the DFE.
    """

    bs = list(np.array(tap_weights).flatten())
    return 1 / (1 - sum(list(map(lambda n_b: n_b[1] * np.exp(-1j * TWOPI * (n_b[0] + 1) * td * freqs),
                                 enumerate(bs)))))


def null_filter(nTaps: int, nPreTaps: int = 0) -> Rvec:
    """
    Construct a null filter w/ `nTaps` taps and (optionally) `nPreTaps` pre-cursor taps.

    Args:
        nTaps: Total number of taps, including the cursor tap.

    Keyword Args:
        nPreTaps: Number of pre-cursor taps.
            Default: 0

    Returns:
        taps: The filter tap weight vector, including the cursor tap weight.
    """

    assert nTaps > 0, ValueError(
        f"`nTaps` ({nTaps}) must be greater than zero!")
    assert nPreTaps < nTaps, ValueError(
        f"`nPreTaps` ({nPreTaps}) must be less than `nTaps` ({nTaps})!")

    taps = np.zeros(nTaps)
    taps[nPreTaps] = 1.0

    return taps
