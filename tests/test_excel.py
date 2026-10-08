"""Tests for the COM configuration spreadsheet parser (``pychopmarg.excel``)."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
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



@pytest.mark.parametrize("name", ["config.xlsx", "CONFIG.XLSX"])
def test_xlsx_reads_same_as_xls(tmp_path, params, name):
    "The same configuration, saved as *.xlsx, parses to identical parameters."
    xlsx = tmp_path / name
    pd.read_excel(CFG, header=None).to_excel(xlsx, header=False, index=False)
    xlsx_params = get_com_params(xlsx)
    for field in dataclasses.fields(params):
        assert np.array_equal(np.asarray(getattr(xlsx_params, field.name), dtype=object),
                              np.asarray(getattr(params, field.name), dtype=object)), field.name
