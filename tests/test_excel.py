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


@pytest.mark.parametrize("rows, dfe_min, dfe_max", [
    ({"b_max(2..N_b)": 0.3}, [0.0, -0.3, -0.3], [0.85, 0.3, 0.3]),                         # `b_min` defaults to `-b_max`.
    ({"b_max(2..N_b)": 0.3, "b_min(2..N_b)": -0.1}, [0.0, -0.1, -0.1], [0.85, 0.3, 0.3]),  # Explicit `b_min`.
    ({"b_max(2..N_b)": 0.3, "b_min(1)": None}, [-0.85, -0.3, -0.3], [0.85, 0.3, 0.3]),    # No `b_min(1)` row.
    ({"b_max(2..N_b)": None}, [0.0, -0.2, -0.2], [0.85, 0.2, 0.2]),                         # No `b_max(2..N_b)` row.
    ({"b_max(2..N_b)": "[0.3 0.25]"}, [0.0, -0.3, -0.25], [0.85, 0.3, 0.25]),               # Per-tap `b_max`.
])
def test_dfe_limits(tmp_path, rows, dfe_min, dfe_max):
    """Each DFE tap's minimum is the sheet's `b_min` (tap 1: `b_min(1)`; taps 2..N_b: `b_min(2..N_b)`), or else `-b_max`.
    Taps 2..N_b's maximum is `b_max(2..N_b)` (a scalar, or one per tap), or else 0.2. (As MATLAB COM.)"""
    sheet = pd.read_excel(CFG, header=None)
    sheet.loc[sheet[0] == "N_b", 1] = 3
    if rows["b_max(2..N_b)"] is None:  # Omit `b_max(2..N_b)`.
        sheet = sheet[sheet[0] != "b_max(2..N_b)"]
    else:
        sheet.loc[sheet[0] == "b_max(2..N_b)", 1] = rows["b_max(2..N_b)"]
    if "b_min(2..N_b)" in rows:
        sheet.loc[len(sheet), [0, 1]] = ["b_min(2..N_b)", rows["b_min(2..N_b)"]]
    if rows.get("b_min(1)", 0.0) is None:  # Omit `b_min(1)`.
        sheet = sheet[sheet[0] != "b_min(1)"]
    cfg = tmp_path / "dfe.xlsx"
    sheet.to_excel(cfg, header=False, index=False)
    params = get_com_params(cfg)
    assert list(params.dfe_min) == pytest.approx(dfe_min)
    assert list(params.dfe_max) == pytest.approx(dfe_max)
