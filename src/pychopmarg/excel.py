"""
MS Excel importing utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   January 17, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

from functools import reduce
from pathlib import Path
import re
import warnings
from typing import Any, TypeVar
from collections.abc import Callable

import numpy as np
from numpy import arange, array, concatenate, ones, zeros
from numpy.typing import NDArray
import pandas as pd

from pychopmarg.config.ieee_8023dj import IEEE_8023dj
from pychopmarg.config.template    import COMParams

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")

# Field name translation table (XLS => ``COMParams``)
com_fields: dict[str, str] = {
    "z_p select": "zp_sel",
    "z_p (TX)": "zp_tx",
    "z_p (NEXT)": "zp_next",
    "z_p (FEXT)": "zp_fext",
    "z_p (RX)": "zp_rx",
    "z_bp (TX)": "zbp_tx",
    "z_bp (NEXT)": "zbp_next",
    "z_bp (FEXT)": "zbp_fext",
    "z_bp (RX)": "zbp_rx",
    "c(0)": "c0_min",
    "b_max(1)": "b_max1",
    "b_min(1)": "b_min1",
    "b_max(2..N_b)": "b_maxN",
    "b_min(2..N_b)": "b_minN",
    "ffe_pre_tap_len": "dw",
    "f_b": "fb",
    "Delta_f": "fstep",
    "g_DC_HP": "g_DC2",
    "f_HP_PZ": "f_LF",
    "R_LM": "RLM",
    "sigma_RJ": "sigma_Rj",
    "package_tl_tau": "tau",
}
ignored_fields = [
    "Operational control",
    "COM Pass threshold",
    "Include PCB",
    "Table 92–12 parameters",
    "Parameter",
    "PMD_type",
    "Histogram_Window_Weight",
]
# MATLAB COM rows that don't affect COM (reports, plots, other analyses, input checks).
report_fields = {
    "DIAGNOSTICS", "DISPLAY_WINDOW", "Display frequency domain", "CSV_REPORT", "SAVE_FIGURE_to_CSV", "RESULT_DIR",
    "SAVE_FIGURES", "SAVE_TD", "BREAD_CRUMBS", "TDR", "ERL", "EW", "Sigma BBN step", "f_min",
}
# MATLAB COM options PyChOpMarg doesn't model, w/ the values it matches (see ``docs/SpecDeviations.md``, B5).
# A sheet setting any other value gets a warning.
matlab_only_fields: dict[str, tuple[Any, ...]] = {
    "ts_anchor": (1,),
    "Raised_Cosine": (0,),
    "FORCE_TR": (1,),
    "INC_PACKAGE": (1,),
    "IDEAL_TX_TERM": (0,),
    "IDEAL_RX_TERM": (0,),
    "INCLUDE_CTLE": (1,),
    "INCLUDE_TX_RX_FILTER": (1,),
    "RX_CALIBRATION": (0,),
    "CTLE_type": ("CL120d",),
    "Port Order": ("[1 3 2 4]",),
    "ffe_tap_step_size": (0,),
    "ffe_main_cursor_min": (1,),
    "PKG_Tx_FFE_preset": (0,),
    "AC_CM_RMS": (0,),
}


class UsageDict(dict):
    "A ``dict`` that records which keys have been looked up (w/ ``[]``, ``get()``, or ``in``), in ``used``."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.used: set[Any] = set()

    def __getitem__(self, key):
        self.used.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.used.add(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        self.used.add(key)
        return super().get(key, default)


def same_setting(val: Any, ok: Any) -> bool:
    "Is the sheet value ``val`` the setting ``ok``? (Numbers compare as numbers; anything else, as whitespace-free text.)"
    try:
        return float(val) == float(ok)
    except (TypeError, ValueError):
        return "".join(str(val).split()).lower() == "".join(str(ok).split()).lower()


def first(f: Callable[[T1], T2]) -> Callable[[tuple[T1, T3]], tuple[T2, T3]]:
    "Translation of Haskell ``first`` function."
    return lambda pr: (f(pr[0]), pr[1])


def second(f: Callable[[T1], T2]) -> Callable[[tuple[T3, T1]], tuple[T3, T2]]:
    "Translation of Haskell ``second`` function."
    return lambda pr: (pr[0], f(pr[1]))


def compose(*functions):
    "Function composition w/ NO TYPE CHECKING!"
    return reduce(lambda f, g: lambda x: f(g(x)), functions)


def apply2(
    f: Callable[[T1], T2],
    g: Callable[[T3], T4]
) -> Callable[[tuple[T1, T3]], tuple[T2, T4]]:
    "Translation of Haskell ``***`` operator."
    return compose(first(f), second(g))


def alternative(
    f: Callable[[T1], T2],
    g: Callable[[T1], T2],
    x: T1
) -> T2:
    """
    Try ``f`` and if it fails apply ``g``.

    Args:
        f: First function to try.
        g: Second function to try.
        x: Function argument.

    Notes:
        1. It would be preferable to omit ``x`` from the argument list
        and return a function, instead of a value, but we can't do that,
        because Python doesn't allow a try/except block inside a lambda.
    """
    try:
        rslt = f(x)
    except Exception:  # pylint: disable=broad-exception-caught
        rslt = g(x)
    return rslt


def parse_Mrange(mStr: str) -> NDArray:
    "Parse a string containing an M-code range."
    start, step, stop = list(map(float, mStr.strip("[] ").split(":")))
    if start == stop == 0:
        return array([0.0])
    return arange(start, stop + step / 2, step)


def parse_Mfloat(mStr: str) -> NDArray:
    "Parse a string containing a single M-code number."
    return array([float(mStr)])


def parse_Mfloats(mStr: str) -> NDArray:
    "Parse a string containing a repeated M-code number."
    pattern = r"([-]?[0-9.]+)\*ones\(1,\s*([0-9]+)\)"
    # match = re.search(pattern, mStr, re.DEBUG)  # Leave as example.
    match = re.search(pattern, mStr)
    if match:
        return array([float(match.group(1))] * int(match.group(2)))
    raise RuntimeError(f"Couldn't parse: '{mStr}'.")


def parse_Mlist(mStr: str) -> NDArray:
    "Parse a string containing an M-code list of numbers."
    tokens = [s.strip(",") for s in mStr.strip("[] ").split()]
    return concatenate([alternative(parse_Mfloat, parse_Mfloats, s) for s in tokens])


def parse_Marray(mStr: str) -> NDArray:
    "Parse a string containing either an M-code range or list of numbers."
    return array(list(filter(lambda x: x is not None,
                             alternative(parse_Mrange, parse_Mlist, mStr))))


def parse_Mmatrix(mStr: str) -> NDArray:
    "Parse a string containing an M-code matrix of numbers."
    rslt = array(list(map(parse_Marray, mStr.strip("[]").split(";"))))
    if rslt.shape[0] == 1:
        return rslt.flatten()
    return rslt


def match_ignored_field_name_prefix(mName: str) -> bool:
    "Match MATLAB field name prefix to any ignored field name."
    for nm in ignored_fields:
        if mName.startswith(nm):
            return True
    return False


def cfg_trans(com_cfg: NDArray) -> dict[str, Any]:
    "Translate/filter the names/values of the given 2D NumPy array."
    return dict(map(apply2(lambda s: com_fields.get(s, s),  # type: ignore
                           lambda s: parse_Mmatrix(s) if isinstance(s, str) else s),
                    filter(lambda pr: not pd.isna(pr[0]) and not match_ignored_field_name_prefix(pr[0]),
                           com_cfg)))


# Kept global, so I can inspect it later while debugging:
com_params_dict: dict[Any, Any] = {}  # pylint: disable=global-statement


def get_com_params(cfg_file: Path) -> COMParams:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    "Read a COM configuration spreadsheet (``*.xls`` or ``*.xlsx``) and return an equivalent ``COMParams`` instance."

    global com_params_dict  # pylint: disable=global-statement

    com_cfg = pd.read_excel(cfg_file)  # (Engine chosen by extension: `xlrd` for *.xls, `openpyxl` for *.xlsx.)

    _com_cfg = com_cfg.to_numpy()
    com_cfg1 = _com_cfg[:, range(2)]
    com_cfg2 = _com_cfg[:, range(9, 11)]

    sheet_params = UsageDict(cfg_trans(com_cfg1))
    sheet_params.update(cfg_trans(com_cfg2))
    # Rows `cfg_trans()` ignores (`ignored_fields`), but some of which we read, as is.
    raw_params = {name.strip(): val for name, val in np.concatenate((com_cfg1, com_cfg2))
                  if isinstance(name, str) and not pd.isna(val)}
    com_params_dict = UsageDict(vars(IEEE_8023dj))
    com_params_dict.update(sheet_params)

    # Package transmission lines (MATLAB COM's names): "package_Z_c" has one row per segment, of [Tx Rx] impedances.
    if "package_tl_gamma0_a1_a2" in sheet_params:
        com_params_dict["gamma0"], com_params_dict["a1"], com_params_dict["a2"] = (
            np.asarray(sheet_params["package_tl_gamma0_a1_a2"], dtype=float).ravel().tolist())
    z_c_rx = None
    if "package_Z_c" in sheet_params:
        z_c = np.atleast_1d(np.asarray(sheet_params["package_Z_c"], dtype=float))
        if z_c.ndim == 1:  # One segment: [Tx Rx], or one impedance for both.
            z_c = z_c.reshape(1, -1)
        com_params_dict["z_c"] = z_c[:, 0].tolist()
        if not np.array_equal(z_c[:, -1], z_c[:, 0]):
            z_c_rx = z_c[:, -1].tolist()

    # EQ search control, from the "I/O control" columns (MATLAB COM's settings of the same names).
    io_ctl = UsageDict()
    if _com_cfg.shape[1] > 6:
        io_ctl.update({name.strip(): val for name, val in _com_cfg[:, range(5, 7)]
                       if isinstance(name, str) and not pd.isna(val)})
    if "T_r" in io_ctl and "T_r" not in sheet_params:  # (MATLAB COM's sheets keep it in the I/O column.)
        com_params_dict["T_r"] = float(io_ctl["T_r"])
    search_ctl: dict[str, Any] = {}
    if "Local Search" in io_ctl:
        search_ctl["local_search"] = float(io_ctl["Local Search"])
    if "TS_SRCH_MODE" in io_ctl:
        search_ctl["ts_srch_mode"] = str(io_ctl["TS_SRCH_MODE"]).strip().lower()
    if "sample_adjustment" in io_ctl:
        adj = np.atleast_1d(parse_Mmatrix(io_ctl["sample_adjustment"]) if isinstance(io_ctl["sample_adjustment"], str)
                            else io_ctl["sample_adjustment"]).astype(int).ravel()
        search_ctl["ts_sample_adj"] = (int(adj[0]), int(adj[-1]))

    # Set the Tx tap ranges/steps: c(-4), ..., c(-1), c(1), ..., c(3), as MATLAB COM (cursor excluded).
    N_TX_PRE     = 4
    N_TX_POST    = 3
    N_TX_TAPS    = N_TX_PRE + N_TX_POST
    tx_taps_min  = [0.] * N_TX_TAPS
    tx_taps_max  = [0.] * N_TX_TAPS
    tx_taps_step = [0.] * N_TX_TAPS

    def set_tap(ix, weights):
        if isinstance(weights, (list, np.ndarray)):
            tx_taps_min[ix] = min(weights)
            tx_taps_max[ix] = max(weights)
            tx_taps_step[ix] = (tx_taps_max[ix] - tx_taps_min[ix]) / (len(weights) - 1)
        else:  # Assume scalar.
            tx_taps_min[ix] = weights
            tx_taps_max[ix] = weights
            tx_taps_step[ix] = 0

    for n in range(N_TX_PRE):
        mKey = f"c(-{n + 1})"
        if mKey in com_params_dict:
            set_tap(N_TX_PRE - (n + 1), com_params_dict[mKey])
    for n in range(N_TX_POST):
        pKey = f"c({n + 1})"
        if pKey in com_params_dict:
            set_tap(N_TX_PRE + n, com_params_dict[pKey])

    # Set Rx FFE config. Without its rows, there's none (only the cursor tap), as in MATLAB COM.
    if "dw" not in sheet_params:
        assert "ffe_post_tap_len" not in sheet_params, ValueError(
            "Either both or neither of: 'ffe_pre_tap_len' & 'ffe_post_tap_len' must be given in the configuration spreadsheet!")
        com_params_dict["dw"] = 0
        rx_taps_min = rx_taps_max = array([1.0])
    else:
        assert "ffe_post_tap_len" in sheet_params, ValueError(
            "Either both or neither of: 'ffe_pre_tap_len' & 'ffe_post_tap_len' must be given in the configuration spreadsheet!")
        nRxPreTaps = com_params_dict["dw"]
        nRxPostTaps = com_params_dict["ffe_post_tap_len"]
        nRxTaps = nRxPreTaps + 1 + nRxPostTaps  # Includes cursor tap.
        rx_taps_min = array([-0.7] * nRxTaps)
        rx_taps_max = array([0.7] * nRxTaps)
        # Set cursor max/min to 1.0 (i.e. - no clipping, because relative).
        rx_taps_min[nRxPreTaps] = rx_taps_max[nRxPreTaps] = 1.0
        if "ffe_pre_tap1_max" in com_params_dict:
            rx_taps_max[nRxPreTaps - 1] = com_params_dict["ffe_pre_tap1_max"]
            rx_taps_min[nRxPreTaps - 1] = -com_params_dict["ffe_pre_tap1_max"]
        if "ffe_post_tap1_max" in com_params_dict:
            rx_taps_max[nRxPreTaps + 1] = com_params_dict["ffe_post_tap1_max"]
            rx_taps_min[nRxPreTaps + 1] = -com_params_dict["ffe_post_tap1_max"]
        if "ffe_tapn_max" in com_params_dict:
            rx_taps_max[:nRxPreTaps - 1] = com_params_dict["ffe_tapn_max"] * ones(nRxPreTaps - 1)
            rx_taps_max[nRxPreTaps + 2:] = com_params_dict["ffe_tapn_max"] * ones(nRxTaps - nRxPreTaps - 2)
            rx_taps_min[:nRxPreTaps - 1] = -com_params_dict["ffe_tapn_max"] * ones(nRxPreTaps - 1)
            rx_taps_min[nRxPreTaps + 2:] = -com_params_dict["ffe_tapn_max"] * ones(nRxTaps - nRxPreTaps - 2)

    # Make sure CTLE d.c. gains are both lists. They're used as written (dB), as MATLAB COM does, so must be <= 0;
    # a positive one is probably an attenuation entered without its sign (and fails `COMParams` validation).
    for key, sheet_name in (("g_DC", "g_DC"), ("g_DC2", "g_DC_HP")):
        if not isinstance(com_params_dict[key], (list, np.ndarray)):
            com_params_dict[key] = [com_params_dict[key]]
        com_params_dict[key] = [float(g) for g in com_params_dict[key]]
        if any(g > 0 for g in com_params_dict[key]):
            warnings.warn(f"`{sheet_name}` in '{cfg_file}' has positive values ({com_params_dict[key]}); CTLE d.c. "
                          "gains are <= 0 dB, and MATLAB COM uses them as written. Missing minus signs?",
                          stacklevel=2)

    # Set Rx DFE min./max.
    dfe_max = dfe_min = array([])  # Default is an empty array and used if we don't find `N_b` in our parse results.
    if "N_b" in com_params_dict:
        N_b = int(com_params_dict["N_b"])
        if N_b > 0:
            dfe_max = zeros(N_b)
            dfe_min = zeros(N_b)
            if "b_max1" in com_params_dict:
                dfe_max[0] = float(com_params_dict["b_max1"])
            dfe_min[0] = float(com_params_dict.get("b_min1", -dfe_max[0]))  # As MATLAB COM, defaults to `-b_max`.
            b_maxN, b_minN = com_params_dict.get("b_maxN", 0.2), com_params_dict.get("b_minN")
            if N_b > 1:  # Taps 2..N_b, as MATLAB COM: `b_max` defaults to 0.2, `b_min` to `-b_max`; each may be per tap.
                dfe_max[1:] = np.asarray(b_maxN, dtype=float).ravel()
                dfe_min[1:] = -dfe_max[1:] if b_minN is None else np.asarray(b_minN, dtype=float).ravel()

    rslt = COMParams(
        com_params_dict["fb"],
        com_params_dict["fstep"],
        com_params_dict["L"],
        com_params_dict["M"],
        com_params_dict["DER_0"],
        com_params_dict["T_r"],
        com_params_dict["RLM"],
        com_params_dict["A_v"],
        com_params_dict["A_fe"],
        com_params_dict["A_ne"],
        com_params_dict["R_0"],
        com_params_dict["A_DD"],
        com_params_dict["SNR_TX"],
        com_params_dict["eta_0"],
        com_params_dict["sigma_Rj"],
        com_params_dict["f_z"],
        com_params_dict["f_p1"],
        com_params_dict["f_p2"],
        com_params_dict["f_LF"],
        com_params_dict["g_DC"],
        com_params_dict["g_DC2"],
        tx_taps_min,
        tx_taps_max,
        tx_taps_step,
        com_params_dict["c0_min"],
        com_params_dict["f_r"],
        dfe_min,
        dfe_max,
        rx_taps_min,
        rx_taps_max,
        com_params_dict["dw"],
        com_params_dict["R_d"],
        com_params_dict["C_d"],
        com_params_dict["C_b"],
        com_params_dict["C_p"],
        com_params_dict["L_s"],
        com_params_dict["z_c"],
        com_params_dict["z_p"],
        com_params_dict["gamma0"],
        com_params_dict["a1"],
        com_params_dict["a2"],
        com_params_dict["tau"],
        **search_ctl,
    )
    if "z_pB" in com_params_dict:
        rslt.z_pB = com_params_dict["z_pB"]
    rslt.z_c_rx = z_c_rx
    if "f_p3" in sheet_params:
        rslt.f_p3 = float(sheet_params["f_p3"])

    # Floating Rx FFE taps, as MATLAB COM: `N_bg` groups of `N_bf` taps (default 6), within +/-`bmaxg` (default 0.2),
    # as late as post-cursor tap `N_f`. (MATLAB COM uses them for the DFE instead, without an Rx FFE.)
    n_bg = int(sheet_params.get("N_bg", 0))
    n_f = sheet_params.get("N_f")
    n_bf = int(sheet_params.get("N_bf", 6))
    bmaxg = float(sheet_params.get("bmaxg", 0.2))
    if n_bg and len(rx_taps_max):
        nfix = len(rx_taps_max)
        nmax = rslt.dw + int(n_f) + 1
        rslt.rx_taps_min = np.concatenate((rx_taps_min, -bmaxg * ones(nmax - nfix)))
        rslt.rx_taps_max = np.concatenate((rx_taps_max, bmaxg * ones(nmax - nfix)))
        rslt.Nfix, rslt.Nwg, rslt.Nwf = nfix, n_bg, n_bf

    # MLSD: MATLAB COM's "MLSD" (or "MLSE") switch, wherever it appears; `add_rx_noise` defaults to the
    # "COM Pass threshold", as in MATLAB COM (and D3.3: the minimum COM limit).
    mlsd_flag = sheet_params.get("MLSD", sheet_params.get("MLSE", io_ctl.get("MLSD", io_ctl.get("MLSE", 0))))
    rslt.mlsd = bool(float(mlsd_flag))
    rslt.dCOM_an = float(sheet_params.get("add_rx_noise", raw_params.get("COM Pass threshold", 3.0)))

    # Host boards: MATLAB COM's "Include PCB", w/ its board rows (and its defaults, where they're missing).
    rslt.include_pcb = int(float(raw_params.get("Include PCB", 0)))
    for key in ("zbp_tx", "zbp_rx", "zbp_next", "zbp_fext"):
        if key in sheet_params:
            setattr(rslt, key, float(sheet_params[key]))

    def per_end(val: Any) -> list[float]:
        "A [Tx, Rx] pair, from a scalar (used at both ends) or a pair."
        vals = np.atleast_1d(np.asarray(val, dtype=float)).ravel().tolist()
        return vals * 2 if len(vals) == 1 else vals[:2]

    if "board_Z_c" in sheet_params:
        rslt.brd_Z_c = per_end(sheet_params["board_Z_c"])
    if "board_tl_gamma0_a1_a2" in sheet_params:
        rslt.brd_gamma0_a1_a2 = np.asarray(sheet_params["board_tl_gamma0_a1_a2"], dtype=float).ravel().tolist()
    if "board_tl_tau" in sheet_params:
        rslt.brd_tau = float(sheet_params["board_tl_tau"])
    for key, name in (("brd_C0", "C_0"), ("brd_C1", "C_1")):
        if name in sheet_params:
            setattr(rslt, key, per_end(sheet_params[name]))

    # Quantization noise: off unless the sheet sets `N_qb`, and `P_qc` defaults to 2 * DER_0, as in MATLAB COM.
    rslt.N_qb = int(sheet_params.get("N_qb", 0))
    rslt.P_qc = float(sheet_params.get("P_qc", 2 * rslt.DER_0))

    # Per-end package lengths: MATLAB COM's "z_p (TX)", etc. have one row per package segment and one column per
    # test case; "z_p select" (1-based) lists the cases to evaluate. `zp_tx`, etc. hold the first; `zp_cases`, all of
    # them, when there are several.
    zp_sel = np.atleast_1d(com_params_dict.get("zp_sel", 1)).astype(int)
    cases: list[dict[str, list[float]]] = [{} for _ in zp_sel]
    for key in ("zp_tx", "zp_rx", "zp_next", "zp_fext"):
        if key in com_params_dict:
            zp = np.asarray(com_params_dict[key], dtype=float)
            if zp.ndim < 2:  # One segment, possibly several cases.
                zp = zp.reshape(1, -1)
            for case, sel in zip(cases, zp_sel):
                case[key] = zp[:, sel - 1].tolist()
            setattr(rslt, key, cases[0][key])
    if len(cases) > 1 and cases[0]:
        rslt.zp_cases = cases

    # Warn of sheet rows we don't read, and of MATLAB COM options set to values we don't model.
    used = com_params_dict.used | sheet_params.used | io_ctl.used
    unread = []
    for name, val in list(raw_params.items()) + list(io_ctl.items()):
        if name in matlab_only_fields:
            if not any(same_setting(val, ok) for ok in matlab_only_fields[name]):
                warnings.warn(f"'{name}' is {val!r} in '{cfg_file}', but PyChOpMarg doesn't model that option; "
                              f"it matches MATLAB COM w/ '{name}' = {matlab_only_fields[name][0]!r}.", stacklevel=2)
        elif not (com_fields.get(name, name) in used or name in report_fields or match_ignored_field_name_prefix(name)):
            unread.append(name)
    if unread:
        warnings.warn(f"PyChOpMarg doesn't read these rows of '{cfg_file}', and ignores them: "
                      f"{', '.join(repr(nm) for nm in unread)}.", stacklevel=2)
    return rslt
