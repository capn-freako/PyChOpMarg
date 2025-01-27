"""
MS Excel importing utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   January 17, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

from functools import reduce
from pathlib import Path
import re
from typing import Any, Callable, TypeVar

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
    "ffe_pre_tap_len": "dw",
    "f_b": "fb",
    "Delta_f": "fstep",
    "g_DC_HP": "g_DC2",
    "f_HP_PZ": "f_LF",
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
    except:  # noqa=E722 pylint: disable=bare-except
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
    tokens = list(map(lambda s: s.strip(","),
                      mStr.strip("[] ").split()))
    return concatenate(list(map(lambda s: alternative(parse_Mfloat, parse_Mfloats, s),
                                tokens)))


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
    return dict(map(apply2(lambda s: com_fields[s] if s in com_fields else s,  # type: ignore
                           lambda s: parse_Mmatrix(s) if isinstance(s, str) else s),
                    filter(lambda pr: not pd.isna(pr[0]) and not match_ignored_field_name_prefix(pr[0]),
                           com_cfg)))


# Kept global, so I can inspect it later while debugging:
com_params_dict = {}  # pylint: disable=global-statement


def get_com_params(cfg_file: Path) -> COMParams:  # noqa=E501 pylint: disable=too-many-locals,too-many-branches,too-many-statements
    "Read a COM configuration XLS file and return an equivalent ``COMParams`` instance."

    global com_params_dict  # pylint: disable=global-statement

    if cfg_file.suffix == "xlsx":  # You'll need to manually export to `*.xls` from within Excel.
        raise RuntimeError("Currently, *.XLSX files must first be manually converted to *.XLS.")
        # com_cfg = pd.read_excel(cfg_file, engine_kwargs={"read_only": True})  # Doesn't work, currently.
    com_cfg = pd.read_excel(cfg_file)

    _com_cfg = com_cfg.to_numpy()
    com_cfg1 = _com_cfg[:, range(0, 2)]
    com_cfg2 = _com_cfg[:, range(9, 11)]

    com_params_dict = vars(IEEE_8023dj).copy()
    com_params_dict.update(cfg_trans(com_cfg1))
    com_params_dict.update(cfg_trans(com_cfg2))

    # Set the Tx tap ranges/steps.
    N_TX_TAPS    = 6  # not including the cursor
    N_TX_TAPS_2  = N_TX_TAPS // 2
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

    for n in range(N_TX_TAPS_2):
        mKey = f"c(-{n + 1})"
        pKey = f"c({n + 1})"
        if mKey in com_params_dict:
            set_tap(N_TX_TAPS_2 - (n + 1), com_params_dict[mKey])
        if pKey in com_params_dict:
            set_tap(N_TX_TAPS_2 + n, com_params_dict[pKey])

    # Set Rx FFE config.
    if "dw" in com_params_dict:
        assert "ffe_post_tap_len" in com_params_dict, ValueError(
            "Either both or neither of: 'ffe_pre_tap_len' & 'ffe_post_tap_len' must be given in the configuration spreadsheet!")  # noqa=E501
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

    # Make sure CTLE d.c. gains are both lists.
    if not isinstance(com_params_dict["g_DC"], (list, np.ndarray)):
        com_params_dict["g_DC"] = [com_params_dict["g_DC"]]
    if not isinstance(com_params_dict["g_DC2"], (list, np.ndarray)):
        com_params_dict["g_DC2"] = [com_params_dict["g_DC2"]]

    # Set Rx DFE min./max.
    dfe_max = dfe_min = array([])  # Default is an empty array and used if we don't find `N_b` in our parse results.
    if "N_b" in com_params_dict:
        N_b = int(com_params_dict["N_b"])
        if N_b > 0:
            dfe_max = zeros(N_b)
            dfe_min = zeros(N_b)
            if "b_max1" in com_params_dict:
                dfe_max[0] = float(com_params_dict["b_max1"])
            if "b_min1" in com_params_dict:
                dfe_min[0] = float(com_params_dict["b_min1"])
            if N_b > 1:
                if "b_maxN" in com_params_dict:
                    dfe_max[1:] = dfe_min[1:] = float(com_params_dict["b_maxN"])

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
    )
    if "z_pB" in com_params_dict:
        rslt.z_pB = com_params_dict["z_pB"]
    return rslt
