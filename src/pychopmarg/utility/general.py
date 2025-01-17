"""
General purpose utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from pathlib import Path
import re
from typing import Any, Dict, Optional, TypeVar

import numpy as np  # type: ignore
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from pychopmarg.common import *
from pychopmarg.config.template import COMParams

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
            ranges.append([0.0])  # type: ignore
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


def fwhm(pr: Rvec) -> float:
    """
    Measure the full width at half maximum of the given pulse response.
    
    Args:
        pr: Pulse response to measure.
        
    Returns:
        fwhm: Full width at half max of largest peak in given signal.

    Notes:
        1. Used to characterize the _bandwidth_ of a given channel.
    """
    pk_loc = np.where(pr == max(pr))[0][0]
    half_max = 0.5 * pr[pk_loc]
    left_hm = pk_loc
    while left_hm > 0 and pr[left_hm] > half_max:
        left_hm -= 1
    right_hm = pk_loc
    while right_hm < (len(pr) - 1) and pr[right_hm] > half_max:
        right_hm += 1
    return right_hm - left_hm


def reflectivity(pr: Rvec) -> float:
    """
    Measure the _reflectivity_ of a channel with the given pulse response.
    
    Args:
        pr: Pulse response of channel.
        
    Returns:
        ref: Reflectivity of channel.

    Notes:
        1. Use sum of: delta-x weighted by power at delta-x.
    """
    pk_loc = np.where(pr == max(pr))[0][0]
    return sum([dn * y**2 for dn, y in enumerate(pr[pk_loc:])])
    

def get_channel_sets(path: Path) -> dict[ChnlGrpName, list[ChnlSet]]:
    """
    Return all available groups of channel sets in the given path.

    Args:
        path: The folder in which to begin searching.
            (Assumed to contain some number of sub-directories,
            in which the actual channel sets are contained.)

    Returns:
        Dictionary of channel groups, each containing a list of channel sets.

    Notes:
        1. A "channel set" is a dictionary containing a thru channel and some number
            of NEXT and FEXT aggressors.
    """
    
    chnl_groups = list(filter(lambda p: p.is_dir(), path.iterdir()))
    chnl_groups.sort()
    channels = {}
    for chnl_grp in chnl_groups:
        channels[chnl_grp.name] = []
        thru_chnls = list(chnl_grp.glob("*[tT][hH][rR][uU]*.[sS]4[pP]"))  # No global option for case insensitive glob().
        thru_chnls.sort()
        for thru_chnl in thru_chnls:
            nexts = list(chnl_grp.glob(re.sub("thru", "[nN][eE][xX][tT][0-9]", thru_chnl.name, flags=re.IGNORECASE)))
            nexts.sort()
            fexts = list(chnl_grp.glob(re.sub("thru", "[fF][eE][xX][tT][0-9]", thru_chnl.name, flags=re.IGNORECASE)))
            fexts.sort()
            channels[chnl_grp.name].append({  # Here we're constructing a "channel set" dictionary.
                "THRU": [thru_chnl],
                "NEXT": nexts,
                "FEXT": fexts
            })
    return channels


def dBm_Hz(x: Rvec) -> Rvec:
    "Convert (V^2/Hz) to (dBm/Hz), assuming 100 Ohm system impedance."
    return 10 * np.log10(1e3 * x / 100)


def mag_dB(x: Cvec) -> Rvec:
    "Return the magnitude in dB of a complex amplitude vector."
    return 20 * np.log10(np.abs(x))


