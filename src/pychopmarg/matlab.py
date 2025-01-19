"""
General MATLAB utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   January 17, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

from pathlib    import Path
import subprocess
from typing     import Any, TypeAlias

from pychopmarg.common import ChnlSet, ChnlGrpName, ChnlSetName

COM_MATLAB_RESULTS: TypeAlias = dict[str, Any]


def run_com_matlab(
    chnl_sets: list[tuple[ChnlGrpName, ChnlSet]],
    cfg_sheet: Path,
    matlab_exec: Path
) -> dict[ChnlGrpName, dict[ChnlSetName, COM_MATLAB_RESULTS]]:
    """
    Run COM on a list of grouped channel sets, using the MATLAB code.

    Args:
        chnl_sets: List of pairs, each consisting of:

            - ch_grp_name: The group name for this list of channel sets.
            - ch_sets: List of channel sets to run.

        cfg_sheet: Path to MS Excel configuration spreadsheet.
        matlab_exec: Path to MATLAB executable.

    Returns:
        Dictionary, indexed by channel group name, containing dictionaries of MATLAB COM results,
        indexed by channel set name.
    """

    results: dict[ChnlGrpName, dict[ChnlSetName, COM_MATLAB_RESULTS]] = {}
    for grp, ch_set in chnl_sets:
        lbl = ch_set['THRU'][0].stem
        cmd_str = ", ".join([
            f"com_ieee8023_93a('{cfg_sheet}'",
            f"{len(ch_set['FEXT'])}, {len(ch_set['NEXT'])}, '{ch_set['THRU'][0]}'",
            ", ".join(list(map(lambda ch: f"'{str(ch)}'", ch_set['FEXT'] + ch_set['NEXT'])))]) + ");"
        print(f"\nCommand:\n{cmd_str}")
        result = subprocess.run([matlab_exec, "-nodisplay", "-batch", cmd_str], capture_output=True, text=True)
        if grp in results:
            results[grp].update({lbl: {"proc_rslt": result}})
        else:
            results.update({grp: {lbl: {"proc_rslt": result}}})
    return results
