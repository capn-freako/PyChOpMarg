"""Tests for the COM configuration spreadsheet parser (``pychopmarg.excel``)."""

from pathlib import Path

import pytest

from pychopmarg.excel import get_com_params

CFG = Path(__file__).parent.parent / "config" / "config_com_ieee8023dj_PyChOpMarg_vs_MATLAB.xls"


@pytest.fixture(scope="module")
def params():
    return get_com_params(CFG)


def test_ctle_gains_are_attenuations(params):
    """The sheet enters `g_DC` ([2:1:12]) and `g_DC_HP` ([0:0.5:3]) as positive attenuations, by convention;
    they're parsed as non-positive gains (dB)."""
    assert params.g_DC == [-float(n) for n in range(2, 13)]
    assert params.g_DC2 == [-0.5 * n for n in range(7)]


def test_validates(params):
    "The parsed parameters, including the sheet's per-device (2-D) `C_d`/`L_s`, pass `COMParams` validation."
    assert params.C_d.shape == (2, 3)
