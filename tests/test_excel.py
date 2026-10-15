"""Tests for the COM configuration spreadsheet parser (``pychopmarg.excel``)."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pychopmarg.excel import get_com_params

# The tests' own copy of the COM configuration, independent of `config/`'s comparison sheet.
CFG = Path(__file__).parent / "data" / "config_com_ieee8023dj_test.xls"


@pytest.fixture(scope="module")
def params():
    return get_com_params(CFG)


def test_ctle_gains_used_as_written(params):
    "`g_DC` ([-12:1:-2]) and `g_DC_HP` ([-3:0.5:0]) are used as written, as MATLAB COM does."
    assert params.g_DC == [-float(n) for n in range(12, 1, -1)]
    assert params.g_DC2 == [-0.5 * n for n in range(6, -1, -1)]


def test_positive_ctle_gains_warn_and_fail_validation(tmp_path):
    "Positive CTLE gains (likely missing minus signs) get a warning, then fail `COMParams` validation."
    sheet = pd.read_excel(CFG, header=None)
    sheet.loc[sheet[0] == "g_DC", 1] = "[2:1:12]"
    cfg = tmp_path / "positive_gdc.xlsx"
    sheet.to_excel(cfg, header=False, index=False)
    with pytest.warns(UserWarning, match="g_DC.*Missing minus signs"), pytest.raises(ValueError, match=r"`g_DC\[0\]`"):
        get_com_params(cfg)


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


def test_per_end_package_lengths(params):
    "The sheet's `z_p (TX)`/`(RX)`/`(NEXT)`/`(FEXT)` rows set each end's package segment lengths (mm), as in MATLAB."
    assert params.zp_tx == params.zp_next == params.zp_fext == [34.0, 1.8]
    assert params.zp_rx == [32.0, 1.8]


@pytest.mark.parametrize("zp_tx, zp_sel, expected", [
    ("[12 30 ; 1.8 1.8]", 2, [30.0, 1.8]),  # One row per segment, one column per test case.
    ("[12 30 ; 1.8 1.8]", 1, [12.0, 1.8]),
    ("[12 30]", 2, [30.0]),                 # A single segment.
])
def test_zp_select_picks_test_case(tmp_path, zp_tx, zp_sel, expected):
    "`z_p select` (1-based) picks the test case (column) of each `z_p (...)` row."
    sheet = pd.read_excel(CFG, header=None)
    for col in (0, 9):
        for end in ("TX", "RX", "NEXT", "FEXT"):  # (Every row must have the same number of test cases.)
            sheet.loc[sheet[col] == f"z_p ({end})", col + 1] = zp_tx
        sheet.loc[sheet[col] == "z_p select", col + 1] = zp_sel
    cfg = tmp_path / "zp.xlsx"
    sheet.to_excel(cfg, header=False, index=False)
    params = get_com_params(cfg)
    assert params.zp_tx == params.zp_rx == params.zp_next == params.zp_fext == expected


def test_zp_select_lists_every_test_case(tmp_path):
    "With several `z_p select` cases, `zp_cases` holds each one's lengths, in order; `zp_tx`, etc., the first's."
    sheet = pd.read_excel(CFG, header=None)
    for col in (0, 9):
        for end in ("TX", "RX", "NEXT", "FEXT"):
            sheet.loc[sheet[col] == f"z_p ({end})", col + 1] = "[12 30 ; 1.8 1.8]"
        sheet.loc[sheet[col] == "z_p select", col + 1] = "[2 1]"
    cfg = tmp_path / "zp_cases.xlsx"
    sheet.to_excel(cfg, header=False, index=False)
    params = get_com_params(cfg)
    ends = ("zp_tx", "zp_rx", "zp_next", "zp_fext")
    assert params.zp_cases == [dict.fromkeys(ends, [30.0, 1.8]), dict.fromkeys(ends, [12.0, 1.8])]
    assert params.zp_tx == [30.0, 1.8]


@pytest.mark.parametrize("rows, N_qb, P_qc", [
    ([], 0, None),                              # Off, by default.
    ([("N_qb", 6)], 6, None),                   # `P_qc` defaults to 2 * DER_0, as in MATLAB COM.
    ([("N_qb", 6), ("P_qc", 1e-7)], 6, 1e-7),
])
def test_quantization_noise_params(tmp_path, rows, N_qb, P_qc):
    "`N_qb` and `P_qc` are read from the sheet; without them, quantization noise is off, as in MATLAB COM."
    sheet = pd.read_excel(CFG, header=None)
    for name, val in rows:
        new_row = pd.DataFrame([[name, val] + [np.nan] * (sheet.shape[1] - 2)], columns=sheet.columns)
        sheet = pd.concat([sheet, new_row], ignore_index=True)
    cfg = tmp_path / "quant.xlsx"
    sheet.to_excel(cfg, header=False, index=False)
    params = get_com_params(cfg)
    assert params.N_qb == N_qb
    assert params.P_qc == pytest.approx(P_qc if P_qc is not None else 2 * params.DER_0)
