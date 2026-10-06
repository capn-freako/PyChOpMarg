"""Tests for `pychopmarg.utility.probability`."""

import numpy as np
import pytest

from pychopmarg.utility import loc_curs

NSPUI = 32
NO_DFE = (np.zeros(1), np.zeros(1))


def _pulse(peak_ix: int, n_samps: int = 40 * NSPUI) -> np.ndarray:
    "A smooth, single-peaked pulse response (~2 UI wide), peaking at sample `peak_ix`."
    t = (np.arange(n_samps) - peak_ix) / NSPUI
    return np.exp(-(t ** 2))


class TestLocCurs:
    def test_finds_mm_point_near_peak(self):
        "A symmetric pulse meets the Mueller-Muller criterion (h(-1) = h(1)) exactly at its peak."
        assert loc_curs(_pulse(400), NSPUI, *NO_DFE) == 400

    def test_peak_near_start_doesnt_wrap(self):
        """With the peak less than 1 UI from the start, candidates' pre-cursor samples lie at negative indices,
        which NumPy wraps to the end of the pulse response; those candidates must be skipped."""
        pr = _pulse(10)
        pr[-NSPUI:] = pr[8 + NSPUI]  # Makes sample 8's *wrapped* "pre-cursor" an exact Mueller-Muller match.
        assert loc_curs(pr, NSPUI, *NO_DFE) >= NSPUI

    def test_peak_near_end_doesnt_overrun(self):
        "With the peak less than 2 UI from the end, candidates' post-cursor samples would overrun it."
        pr = _pulse(40 * NSPUI - NSPUI - 5)
        curs_ix = loc_curs(pr, NSPUI, *NO_DFE)
        assert curs_ix + NSPUI < len(pr)

    def test_skips_non_positive_candidates(self):
        """A candidate at or below zero can't be the cursor (and would divide by zero in (93A-26)), even
        if it happens to meet the Mueller-Muller criterion."""
        pr = _pulse(400)
        ix = 390
        # A negative dip that, with tap 1 anticipated at its -1 limit, makes `ix` an exact Mueller-Muller match.
        pr[ix] = -(pr[ix + NSPUI] - pr[ix - NSPUI])
        curs_ix = loc_curs(pr, NSPUI, np.ones(1), -np.ones(1))
        assert pr[curs_ix] > 0

    def test_degenerate_pulse_raises(self):
        "No positive candidate anywhere near the peak: raise, rather than return a meaningless cursor."
        with pytest.raises(ValueError):
            loc_curs(-_pulse(400), NSPUI, *NO_DFE)
