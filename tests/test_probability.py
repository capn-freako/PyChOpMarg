"""Tests for `pychopmarg.utility.probability`."""

import numpy as np
import pytest

from pychopmarg.utility import calc_hJ, delta_pmf, loc_curs

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


class TestDeltaPmf:
    Y = np.arange(-50, 51) * 0.01  # A 101-bin grid, +/-0.5 V.

    def test_support_and_probabilities(self):
        "One NRZ sample of 0.2 V: half the probability at each of +/-0.2 V."
        _, p = delta_pmf(np.array([0.2]), L=2, y=self.Y)
        assert p[np.isclose(self.Y, 0.2)] == pytest.approx(0.5)
        assert p[np.isclose(self.Y, -0.2)] == pytest.approx(0.5)

    def test_support_reaching_grid_edge(self):
        "A PMF whose support reaches exactly to the grid's edges keeps all of its probability there."
        _, p = delta_pmf(np.array([0.3, 0.2]), L=2, y=self.Y)
        assert p[np.isclose(self.Y, 0.5)] == pytest.approx(0.25)
        assert p[np.isclose(self.Y, -0.5)] == pytest.approx(0.25)
        assert p.sum() == pytest.approx(1.0)

    def test_is_convolution_of_per_sample_pmfs(self):
        """(93A-40): the convolution of each sample's PMF, equiprobable over its L levels' (93A-39) nonzero shifts.
        (Its convolution is carried out only over the growing support of the PMF, which mustn't lose probability.)"""
        rng = np.random.default_rng(0)
        y = np.arange(-500, 501) * 0.001
        h = rng.normal(0, 0.02, 30)
        h[[3, 17]] = 0.0002  # Too small to shift: skipped.
        L = 4
        expected = np.zeros(len(y))
        expected[500] = 1.0
        for hn in h:
            shifts = [s for s in np.round((np.arange(L) * 2 / (L - 1) - 1) * hn / 0.001).astype(int) if s]
            if shifts:
                kernel = np.zeros(2 * max(map(abs, shifts)) + 1)
                for s in shifts:
                    kernel[len(kernel) // 2 + s] += 1 / len(shifts)
                expected = np.convolve(expected, kernel, mode="same")
        _, p = delta_pmf(h, L=L, y=y)
        assert np.allclose(p, expected, rtol=0, atol=1e-15)
        assert p.sum() == pytest.approx(1.0)

    def test_support_beyond_grid_raises(self):
        """Support beyond the grid is an error. (`np.roll()` once silently wrapped that probability around to the
        grid's opposite edge, corrupting COM's interference PMFs whenever they outgrew it.)"""
        with pytest.raises(ValueError, match="exceeds the y grid"):
            delta_pmf(np.array([0.3, 0.3]), L=2, y=self.Y)


class TestCalcHJ:
    def test_slopes_at_every_ui(self):
        """(178A-23): one central-difference slope (V/UI) per UI at the cursor's phase, pre-cursors included, and
        however small. (They once started at the cursor, and skipped samples below 0.1% of As.)"""
        pr = _pulse(400)
        hJ = calc_hJ(pr, 400, NSPUI)
        assert len(hJ) == len(pr) // NSPUI
        ixs = 400 % NSPUI + NSPUI * np.arange(1, len(hJ) - 1)
        assert np.allclose(hJ[1:-1], (pr[ixs + 1] - pr[ixs - 1]) * NSPUI / 2)
        assert hJ[400 // NSPUI - 1] > 0  # Rising pre-cursor slope
        assert np.all(np.delete(hJ, 400 // NSPUI) != 0)  # No small-sample filtering (0 only at the peak)

    def test_phase_zero(self):
        "At sampling phase zero, the first slope uses zero padding rather than wrapping or dropping it."
        pr = _pulse(400)
        hJ = calc_hJ(pr, 13 * NSPUI, NSPUI)
        assert len(hJ) == len(pr) // NSPUI
        assert hJ[0] == pytest.approx(pr[1] * NSPUI / 2)
