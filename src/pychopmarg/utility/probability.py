"""
Statistical utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from typing import Any, Optional

import numpy as np  # type: ignore
from numpy.typing import NDArray

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
    return pr_samps[abs(pr_samps) > thresh]


def delta_pmf(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    h_samps: Rvec, L: int = 4,
    curs_ix: Optional[int] = None, y: Optional[Rvec] = None,
    dbg_dict: Optional[dict[str, Any]] = None
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
        ValueError: If the samples' combined extent (the PMF's support) exceeds the ``y`` grid's.

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

    if dbg_dict is not None:
        dbg_dict.update({"h_samps":      h_samps})
        dbg_dict.update({"h_samps_filt": h_samps_filt})
        dbg_dict.update({"ystep":        ystep})

    # Pre-calculated constant values.
    zero_ix = npts // 2
    sig_step = 2.0 / (L - 1)
    sig_shifts = (np.arange(L) * sig_step - 1) / ystep

    # Each sample's shifts (bins), w/ zeros filtered out, as per MATLAB code.                      # (93A-39)
    all_shifts = np.round(np.outer(np.asarray(h_samps_filt, dtype=float), sig_shifts)).astype(int)
    max_shifts = np.abs(all_shifts).max(axis=1, initial=0)  # (0 for a sample w/ no nonzero shifts)
    max_shift = np.cumsum(max_shifts)
    if len(max_shift) and max_shift[-1] > zero_ix:  # The PMF's support would exceed the `y` grid.
        bad = max_shift[np.argmax(max_shift > zero_ix)]
        raise ValueError(
            f"The samples' combined extent ({bad * ystep:.3g}) exceeds the y grid's ({zero_ix * ystep:.3g}).")

    # PMF calculation proper - a more efficient form of convolution, given our apriori knowledge of the input:
    # the sum of the PMF's shifted copies, normalized. The PMF is nonzero only within `[lo, hi)`, which
    # the check above keeps within the grid; so, each copy is added only over its image of that support.
    # (Elsewhere, the sum's terms are exact zeros, which change nothing.) The normalizing sum is still taken
    # over the whole grid, to keep NumPy's pairwise summation, and so the result, bit-identical.
    # (The support only grows; so, zeroing each new one keeps both buffers zero outside it.)
    rslt = np.zeros(npts)
    rslt[zero_ix] = 1.0
    _rslt: NDArray = np.zeros(npts)
    lo, hi = zero_ix, zero_ix + 1
    for row in all_shifts[max_shifts > 0].tolist():
        shifts = [shift for shift in row if shift]
        new_lo, new_hi = lo + min(min(shifts), 0), hi + max(max(shifts), 0)
        _rslt[new_lo:new_hi] = 0.0
        src = rslt[lo:hi]
        for shift in shifts:
            _rslt[lo + shift: hi + shift] += src
        lo, hi = new_lo, new_hi
        np.divide(_rslt[lo:hi], _rslt.sum(), out=rslt[lo:hi])  # Enforce a PMF.

    return y, rslt


def calc_hJ(pulse_resp: Rvec, cursor_ix: int, nspui: int) -> Rvec:
    """
    Calculate the slope of the pulse response around each of its samples at the cursor's sampling phase,
    as per (178A-23) (and MATLAB COM's ``h_J``): every UI, pre-cursors included, unfiltered.

    Args:
        pulse_resp: The pulse response of interest.
        cursor_ix: Cursor index.
        nspui: Number of samples per UI.

    Returns:
        The slopes (V/UI), one per UI.

    Notes:
        1. Zero padding keeps the first sample valid at sampling phase zero (as in ``NoiseCalc.Sjn``).
    """

    ixs = cursor_ix % nspui + nspui * np.arange(len(pulse_resp) // nspui)
    _pulse_resp = np.pad(pulse_resp, 1)
    return (_pulse_resp[ixs + 2] - _pulse_resp[ixs]) / (2 / nspui)  # (178A-23)


def loc_curs(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    pulse_resp: Rvec, nspui: int, dfe_max: Rvec, dfe_min: Rvec,
    max_range: int = 1, eps: float = 0.001
) -> int:
    """
    Locate the cursor position for the given pulse response,
    according to (93A-25) and (93A-26) (i.e. - Muller-Mueller criterion).

    Args:
        pulse_resp: The pulse response of interest.
        nspui: Number of samples per UI.
        dfe_max: Vector of maximum DFE tap weight values.
        dfe_min: Vector of minimum DFE tap weight values.

    Keyword Args:
        max_range: The search radius, from the peak (UI).
            Default: 1
        eps: Threshold for declaring floating point value to be zero.
            Default: 0.001

    Returns:
        The index in the given pulse response vector of the cursor.

    Raises:
        ValueError: If no candidate within `max_range` of the peak has a positive value and a full UI on
            either side of it in the pulse response (i.e. - a degenerate pulse response).

    Notes:
        1. As per v3.70 of the COM MATLAB code, we only minimize the
        residual of (93A-25); we don't require solving it exactly.
        (We do, however, give priority to exact solutions.)
    """

    # Minimize Muller-Mueller criterion, within `max_range` of peak,
    # giving priority to exact solutions, as per the spec.
    peak_loc = int(np.argmax(pulse_resp))
    ix_best: Optional[int] = None
    res_min = 1e6
    zero_res_ixs = []
    # Each candidate needs a full UI on either side (a negative index would silently wrap around to the
    # far end of the pulse response), and a positive value: it's the cursor, and the divisor of (93A-26).
    for ix in range(max(nspui, peak_loc - nspui * max_range),
                    min(len(pulse_resp) - nspui, peak_loc + nspui * max_range)):
        if pulse_resp[ix] <= 0:
            continue
        # Anticipate the DFE first tap value, observing its limits:
        b_1 = min(dfe_max[0],
                  max(dfe_min[0],
                      pulse_resp[ix + nspui] / pulse_resp[ix]))                          # (93A-26)
        # And include the effect of that tap when checking the Muller-Mueller condition:
        res = abs(pulse_resp[ix - nspui] - (pulse_resp[ix + nspui] - b_1 * pulse_resp[ix]))  # (93A-25)
        if res < eps:        # "Exact" match?
            zero_res_ixs.append(ix)
        elif res < res_min:  # Keep track of best solution, in case no exact matches.
            ix_best = ix
            res_min = res
    if zero_res_ixs:  # Give priority to "exact" matches if there were any.
        pre_peak_ixs = list(filter(lambda x: x <= peak_loc, zero_res_ixs))
        if pre_peak_ixs:
            return pre_peak_ixs[-1]  # Standard says to use first one prior to peak in event of multiple solutions.
        return zero_res_ixs[0]       # They're all post-peak; so, return first (i.e. - closest to peak).
    if ix_best is None:
        raise ValueError(f"No valid cursor candidate within {max_range} UI of the pulse response's peak "
                         f"(sample {peak_loc} of {len(pulse_resp)}, value {pulse_resp[peak_loc]:.3g}).")
    return ix_best                   # No exact solutions; so, return that which yields minimum error.
