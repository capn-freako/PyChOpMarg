"""
Filtering utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

import numpy as np  # type: ignore
import skrf  as rf  # type: ignore

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
    """

    # bs = list(np.array(tap_weights).flatten())
    # if not hasCurs:
    #     b0 = 1 - sum(list(map(abs, tap_weights)))
    #     bs.insert(-n_post, b0)
    bs = tap_weights.flatten()
    if not hasCurs:
        b0 = 1 - abs(tap_weights).sum()
        bs = np.insert(bs, -n_post, b0)
    if False:
        return sum(list(map(lambda n_b: n_b[1] * np.exp(-1j * TWOPI * n_b[0] * td * freqs),
                            enumerate(bs))))
    else:
        return bs @ np.exp(np.outer(np.arange(len(bs)), -1j * TWOPI * td * np.array(freqs)))  # 50% perf. improvement

    # Row sum:
    # (b0 * e(-j 2pi T 0*f0)) (b0 * e(-j 2pi T 0*f1)) (b0 * e(-j 2pi T 0*f2))
    # (b1 * e(-j 2pi T 1*f0)) (b1 * e(-j 2pi T 1*f1)) (b1 * e(-j 2pi T 1*f2))
    # (b2 * e(-j 2pi T 2*f0)) (b2 * e(-j 2pi T 2*f1)) (b2 * e(-j 2pi T 2*f2))

    # Transposing the above:

    # (b0 * e(-j 2pi T 0*f0)) + (b1 * e(-j 2pi T 1*f0)) + (b2 * e(-j 2pi T 2*f0))
    # (b0 * e(-j 2pi T 0*f1)) + (b1 * e(-j 2pi T 1*f1)) + (b2 * e(-j 2pi T 2*f1))
    # (b0 * e(-j 2pi T 0*f2)) + (b1 * e(-j 2pi T 1*f2)) + (b2 * e(-j 2pi T 2*f2))
    # =
    # b `dot` e(-j 2pi T f0 * n), n = [0,1,2]
    # b `dot` e(-j 2pi T f1 * n)
    # b `dot` e(-j 2pi T f2 * n)
    # Transposing:
    # b `dot` e(-j 2pi T f0 * n)   b `dot` e(-j 2pi T f1 * n)   b `dot` e(-j 2pi T f2 * n)
    # =
    # b @ e(-j 2pi T f0 * 0)   e(-j 2pi T f1 * 0)   e(-j 2pi T f2 * 0)
    #     e(-j 2pi T f0 * 1)   e(-j 2pi T f1 * 1)   e(-j 2pi T f2 * 1)
    #     e(-j 2pi T f0 * 2)   e(-j 2pi T f1 * 2)   e(-j 2pi T f2 * 2)
    # =
    # b @ e(n.T @ -j*2pi*T*f)
    # n = np.array([list(range(len(bs))),])

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

    bs = list(np.array(tap_weights).flatten())
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
    s2p = s2p.extrapolate_to_dc()
    s2p.interpolate_self(freqs)
    s11 = s2p.s11.s.flatten()
    s12 = s2p.s12.s.flatten()
    s21 = s2p.s21.s.flatten()
    s22 = s2p.s22.s.flatten()
    dS = s11 * s22 - s12 * s21
    return (s21 * (1 - g1) * (1 + g2)) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * dS)
