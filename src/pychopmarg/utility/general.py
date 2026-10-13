"""
General purpose utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from pathlib import Path
import re
from typing import Any, TypeVar

import numpy as np  # type: ignore
from scipy.interpolate import interp1d

from pychopmarg.common import Rvec, Cvec, ChnlSet, ChnlGrpName

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
            # (Not `arange(min, max + step, step)`, which rounding can extend past `max`: [0:0.1:0.2] => 0, ..., 0.3.)
            n_vals = int(round((trip[1] - trip[0]) / trip[2])) + 1
            ranges.append(list(trip[0] + trip[2] * np.arange(n_vals)))
        else:
            ranges.append([0.0])  # type: ignore
    return list(map(np.array, all_combs(ranges)))


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
    return sum(dn * y**2 for dn, y in enumerate(pr[pk_loc:]))


def get_channel_sets(path: Path, recursive: bool = False) -> dict[ChnlGrpName, list[ChnlSet]]:
    """
    Return all available groups of channel sets in the given path.

    Args:
        path: The folder in which to begin searching.
            (Assumed to contain some number of sub-directories,
            in which the actual channel sets are contained.)

    Keyword Args:
        recursive: Search each group's sub-directories, too, when True.
            (Each thru channel's aggressors are found in its own directory.)
            Default: False

    Returns:
        Dictionary of channel groups, each containing a list of channel sets.

    Notes:
        1. A "channel set" is a dictionary containing a thru channel and some number
            of NEXT and FEXT aggressors.
    """

    chnl_groups = list(filter(lambda p: p.is_dir(), path.iterdir()))
    chnl_groups.sort()
    channels: dict[ChnlGrpName, list[ChnlSet]] = {}
    for chnl_grp in chnl_groups:
        channels[chnl_grp.name] = []
        thru_pat = "*[tT][hH][rR][uU]*.[sS]4[pP]"  # No global option for case insens. glob().
        thru_chnls = list(chnl_grp.rglob(thru_pat) if recursive else chnl_grp.glob(thru_pat))
        thru_chnls.sort()
        aggressors: dict[Path, dict[str, list[Path]]] = {}
        for chnl_dir in sorted({thru.parent for thru in thru_chnls}):
            aggressors.update(_find_aggressors([thru for thru in thru_chnls if thru.parent == chnl_dir]))
        for thru_chnl in thru_chnls:
            channels[chnl_grp.name].append({  # Here we're constructing a "channel set" dictionary.
                "THRU": [thru_chnl],
                "NEXT": aggressors[thru_chnl]["NEXT"],
                "FEXT": aggressors[thru_chnl]["FEXT"],
            })
    return channels


def _find_aggressors(thru_chnls: list[Path]) -> dict[Path, dict[str, list[Path]]]:
    """
    Assign the NEXT/FEXT Touchstone files in a directory to its thru channels.

    Args:
        thru_chnls: The thru channels, all in the same directory.

    Returns:
        For each thru channel, its "NEXT" and "FEXT" aggressor files (sorted).

    Notes:
        1. An aggressor is a "*.s4p" file in the same directory whose name contains "NEXT" or "FEXT" (the last
        occurrence decides which; any case) but not "thru", and begins with the part of the thru channel's name
        before "thru" (less any trailing "_", "-", ".", or " "). Contributors name them variously; e.g., for
        thru ``<stem>_thru<x>.s4p``: ``<stem>_next<n><x>.s4p``, ``<stem><...>_NEXT<n>.s4p``,
        ``<stem>_xtalk<n>_Next.s4p``, or ``<stem>_NEXT_<different suffix>.s4p``.
        2. Each aggressor goes to the thru channel(s) whose name prefix it matches the longest of.
    """

    def prefix(name: str) -> str:
        return name[:_THRU_RE.search(name).start()].rstrip("_-. ").lower()  # type: ignore

    rslt: dict[Path, dict[str, list[Path]]] = {thru: {"NEXT": [], "FEXT": []} for thru in thru_chnls}
    if not thru_chnls:
        return rslt
    for fname in sorted(thru_chnls[0].parent.iterdir()):
        kinds = _XTALK_RE.findall(fname.name)
        if not kinds or _THRU_RE.search(fname.name) or fname.suffix.lower() != ".s4p":
            continue
        matches = [thru for thru in thru_chnls if fname.name.lower().startswith(prefix(thru.name))]
        if not matches:
            continue
        longest = max(len(prefix(thru.name)) for thru in matches)
        for thru in matches:
            if len(prefix(thru.name)) == longest:
                rslt[thru][kinds[-1].upper()].append(fname)
    return rslt


_THRU_RE = re.compile("thru", re.IGNORECASE)
_XTALK_RE = re.compile("next|fext", re.IGNORECASE)


def dBm_Hz(x: Rvec) -> Rvec:
    "Convert (V^2/Hz) to (dBm/Hz), assuming 100 Ohm system impedance."
    return 10 * np.log10(1e3 * x / 100)


def mag_dB(x: Cvec) -> Rvec:
    "Return the magnitude in dB of a complex amplitude vector."
    return 20 * np.log10(np.abs(x))
