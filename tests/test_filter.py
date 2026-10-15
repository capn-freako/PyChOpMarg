"""Tests for `pychopmarg.utility.filter`."""

import numpy as np
import pytest

from pychopmarg.utility import interp_vtf

FIN = np.arange(1, 5001) * 10e6  # 10 MHz ... 50 GHz, w/o d.c.
FOUT = np.arange(0, 20001) * 10e6  # 0 ... 200 GHz


def test_pure_delay_survives_interpolation_and_extrapolation():
    "A pure delay has constant group delay; so, its phase follows the trend everywhere, including d.c. and beyond."
    tau = 3e-9
    H = np.exp(-2j * np.pi * FIN * tau)
    assert np.allclose(np.angle(interp_vtf(H, FIN, FOUT) * np.exp(2j * np.pi * FOUT * tau)), 0, atol=1e-6)


def test_magnitude_trends_to_dc_and_beyond():
    """Magnitude, as MATLAB COM's `linear_trend_to_DC`: d.c. from a linear fit to the first 10 points; above the
    data, a linear ramp to the value at the last output frequency of a linear fit to the upper half, clamped
    to [eps, last value]."""
    mag = 1 - FIN / 100e9  # Linear: 1 at d.c., 0.5 at 50 GHz; trend reaches 0 at 100 GHz.
    H = mag * np.exp(-2j * np.pi * FIN * 1e-9)
    H_i = abs(interp_vtf(H, FIN, FOUT))
    assert H_i[0] == pytest.approx(1.0)
    assert H_i[FOUT == 50e9][0] == pytest.approx(0.5)
    assert H_i[-1] == pytest.approx(np.finfo(float).eps)  # Trend at 200 GHz is negative; so, clamped to eps.
    above = FOUT > FIN[-1]
    assert np.allclose(np.diff(H_i[above]), np.diff(H_i[above])[0])  # A straight ramp


def test_data_with_dc_is_used_as_is():
    "With a d.c. point, magnitude and phase are just interpolated there (no trend fitting at the low end)."
    fin = np.concatenate(([0.0], FIN))
    H = (1 - fin / 100e9) * np.exp(-2j * np.pi * fin * 1e-9)
    assert interp_vtf(H, fin, FOUT)[:5001] == pytest.approx(H)
