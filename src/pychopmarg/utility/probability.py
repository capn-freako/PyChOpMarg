"""
Statistical utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any, Dict, Optional

import numpy as np  # type: ignore

from pychopmarg.common import Rvec


def filt_pr_samps(pr_samps: Rvec, As: float, rel_thresh: float = 0.001) -> Rvec:
    """
    Filter a list of pulse response samples for minimum magnitude.

    Args:
        pr_samps: The pulse response samples to filter.
        As: Signal amplitude, as per 93A.1.6.c.

    Keyword Args:
        rel_thresh: Filtration threshold (As).
            Default: 0.001 (i.e. - 0.1%, as per Note 2 of 93A.1.7.1)

    Returns:
        The subset of ``pr_samps`` passing filtration.
    """
    thresh = As * rel_thresh
    return np.array(list(filter(lambda x: abs(x) >= thresh, pr_samps)))


def delta_pmf(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    h_samps: Rvec, L: int = 4,
    curs_ix: Optional[int] = None, y: Optional[Rvec] = None,
    dbg_dict: Optional[Dict[str, Any]] = None
) -> tuple[Rvec, Rvec]:
    """
    Calculate the "delta-pmf" for a set of pulse response samples,
    as per (93A-40).

    Args:
        h_samps: Vector of pulse response samples.

    Keyword Args:
        L: Number of modulation levels.
            Default: 4
        curs_ix: Cursor index override.
            Default: None (Means use `argmax()` to find cursor.)
        y: y-values override vector.
            Default: None (Means calculate appropriate y-value vector here.)
        dbg_dict: Optional dictionary into which debugging values may be stashed,
            for later analysis.
            Default: None

    Returns:
        A pair consisting of

            - the voltages corresponding to the bins, and
            - their probabilities.

    Raises:
        ValueError: If the given pulse response contains any NaNs.
        ValueError: If a needed shift exceeds half the result vector length.

    Notes:
        1. The input set of pulse response samples is filtered,
        as per Note 2 of 93A.1.7.1, unless a y-values override
        vector is provided, in which case it is assumed that
        the caller has already done the filtering.
    """

    assert not any(np.isnan(h_samps)), ValueError(
        f"Input contains NaNs at: {np.where(np.isnan(h_samps))[0]}")

    if y is None:
        if curs_ix is None:
            curs_ix = int(np.argmax(h_samps))
        curs_val = h_samps[curs_ix]
        max_y = 1.1 * curs_val
        npts = 2 * min(int(max_y / 0.00001), 10_000) + 1  # Note 1 of 93A.1.7.1; MUST BE ODD!
        y = np.linspace(-max_y, max_y, npts)
        ystep = 2 * max_y / (npts - 1)
        h_samps_filt = filt_pr_samps(h_samps, max_y)
    else:
        npts = len(y)
        ystep = y[1] - y[0]
        h_samps_filt = h_samps

    delta = np.zeros(npts)
    delta[npts // 2] = 1

    if dbg_dict is not None:
        dbg_dict.update({"h_samps":      h_samps})
        dbg_dict.update({"h_samps_filt": h_samps_filt})
        dbg_dict.update({"ystep":        ystep})

    def pn(hn: float) -> Rvec:
        """
        (93A-39)
        """
        if dbg_dict:
            dbg_dict.update({"hn":     hn})
            dbg_dict.update({"shifts": []})
        _rslt = np.zeros(npts)
        for el in range(L):
            _shift = int((2 * el / (L - 1) - 1) * hn / ystep)
            if dbg_dict:
                dbg_dict["shifts"].append(_shift)
            assert abs(_shift) < npts // 2, ValueError(
                f"Wrap around: _shift: {_shift}, npts: {npts}.")
            _rslt += np.roll(delta, _shift)
        return 1 / L * _rslt

    rslt = delta
    for hn in h_samps_filt:
        _pn = pn(hn)
        rslt = np.convolve(rslt, _pn, mode='same')
    rslt /= sum(rslt)  # Enforce a PMF.

    return y, rslt
