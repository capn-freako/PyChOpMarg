"""
General purpose utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any, TypeVar

import numpy as np  # type: ignore
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from pychopmarg.common import Rvec

T = TypeVar('T', Any, Any)


def all_combs(xss: list[list[T]]) -> list[list[T]]:
    """
    Generate all combinations of input.

    Args:
        xss: The lists of candidates for each position in the final output.

    Returns:
        All possible combinations of inputs.
    """
    if not xss:
        return [[]]
    head, *tail = xss
    yss = all_combs(tail)
    return [[x, *ys] for x in head for ys in yss]  # type: ignore


def mk_combs(trips: list[tuple[float, float, float]]) -> list[Rvec]:
    """
    Make all possible combinations of tap weights, given a list of "(min, max, step)" triples.

    Args:
        trips: A list of "(min, max, step)" triples, one per weight.

    Returns:
        A list of NDArrays of tap weights, including all possible combinations.
    """
    ranges = []
    for trip in trips:
        if trip[2]:  # non-zero step?
            ranges.append(list(np.arange(trip[0], trip[1] + trip[2], trip[2])))
        else:
            ranges.append([0.0])
    # return list(map(lambda xs: np.array(xs), all_combs(ranges)))
    return list(map(np.array, all_combs(ranges)))
    # rslt = []
    # combs = all_combs(ranges)
    # for xs in combs:
    #     try:
    #         ys = np.array(xs)[::-1]
    #     except:
    #         print(f"xs: {xs}")
    #         print(f"len(combs): {len(combs)}")
    #         print(f"Empty combs: {len(list(filter(lambda x: not x, combs)))}")
    #         raise
    #     rslt.append(ys)
    # return rslt


def from_irfft(x: Rvec, t_irfft: Rvec, t: Rvec, nspui: int) -> Rvec:
    """
    Interpolate ``irfft()`` output to ``t`` and subsample at fBaud.

    Args:
        x: ``irfft()`` results to be interpolated and subsampled.
        t_irfft: Time index vector for ``x``.
        t: Desired new time index vector (same units as ``t_irfft``).
        nspui: Number of samples per unit interval in ``t``.

    Returns:
        Interpolated and subsampled vector.

    Raises:
        IndexError: If length of input doesn't match length of ``t_irfft`` vector.

    Notes:
        1. Input vector is shifted, such that its peak occurs at ``0.1 * max(t)``, before interpolating.
        This is done to:

            - ensure that we don't omit any non-causal behavior,
              which ends up at the end of an IFFT output vector
              when the peak is very near the beginning, and
            - to ensure that the majority of our available time span
              is available for capturing reflections.

        2. The sub-sampling phase is adjusted, so as to ensure that we catch the peak.
    """

    assert len(x) == len(t_irfft), IndexError(
        f"Length of input ({len(x)}) must match length of `t_irfft` vector ({len(t_irfft)})!")

    t_pk = 0.1 * t[-1]                         # target peak location time
    targ_ix = np.where(t_irfft >= t_pk)[0][0]  # target peak vector index, in `x`
    curr_ix = np.argmax(x)                     # current peak vector index, in `x`
    _x = np.roll(x, targ_ix - curr_ix)         # `x` with peak repositioned

    krnl = interp1d(t_irfft, _x, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    y = krnl(t)
    _, curs_ofst = divmod(np.argmax(y), nspui)  # Ensure that we capture the peak in the next step.
    return y[curs_ofst::nspui]                  # Sampled at fBaud, w/ peak captured.


def print_taps(ws: list[float]) -> str:
    """Return formatted tap weight values."""
    n_ws = len(ws)
    if n_ws == 0:
        return ""
    res = f"{ws[0]:5.2f}"
    if n_ws > 1:
        if n_ws > 8:
            for w in ws[1:8]:
                res += f" {w:5.2f}"
            res += f"\n{ws[8]:5.2f}"
            for w in ws[9:]:
                res += f" {w:5.2f}"
        else:
            for w in ws[1:]:
                res += f" {w:5.2f}"
    return res
