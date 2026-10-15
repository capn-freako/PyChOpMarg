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


def test_ctle_second_stage_pole_may_differ_from_its_zero():
    """The CTLE (178A-14's Hctf): (g1 + jf/fz1)(g2 + jf/fz2) / ((1 + jf/fp1)(1 + jf/fp2)(1 + jf/fp3)); w/o `fp3`,
    it's (93A-22), w/ fp3 = fz2."""
    from pychopmarg.utility import calc_Hctle  # pylint: disable=import-outside-toplevel
    f = np.linspace(0, 100e9, 101)
    fz1, fp1, fp2, fz2, fp3, gdc, gdc2 = 42.5e9, 42.5e9, 106.25e9, 1.3e9, 2.0e9, -6.0, -2.0
    g1, g2 = 10**(gdc / 20), 10**(gdc2 / 20)
    want = ((g1 + 1j * f / fz1) * (g2 + 1j * f / fz2)
            / ((1 + 1j * f / fp1) * (1 + 1j * f / fp2) * (1 + 1j * f / fp3)))
    assert np.allclose(calc_Hctle(f, fz1, fp1, fp2, fz2, gdc, gdc2, fp3=fp3), want)
    assert np.allclose(calc_Hctle(f, fz1, fp1, fp2, fz2, gdc, gdc2),
                       calc_Hctle(f, fz1, fp1, fp2, fz2, gdc, gdc2, fp3=fz2))
    # High-frequency asymptote of the second stage: fp3 / fz2.
    assert abs(calc_Hctle(np.array([1e15]), 1e30, 1e30, 1e30, fz2, 0.0, gdc2, fp3=fp3)[0]) == pytest.approx(fp3 / fz2, rel=1e-3)
