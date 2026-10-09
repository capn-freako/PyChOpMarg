"""
General MATLAB utilities for PyChOpMarg.

Runs the official MATLAB COM code through the MATLAB Engine API for Python (``matlab.engine``), via the small
wrapper ``mfiles/pychopmarg_com.m``, which returns COM's own results as data.

The engine isn't a PyChOpMarg dependency: it must match your MATLAB release (and that release must support your
Python version), so install it yourself, e.g. for MATLAB R2024b: ``pip install "matlabengine==24.2.*"``.

Original author: David Banas <capn.freako@gmail.com>

Original date:   January 17, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

import io
from pathlib    import Path
from typing     import Any, Optional, TypeAlias

import numpy as np

from pychopmarg.common import ChnlSet, ChnlGrpName, ChnlSetName

COM_MATLAB_RESULTS: TypeAlias = dict[str, Any]
MFILES_DIR = Path(__file__).parent / "mfiles"  #: MATLAB code shipped with PyChOpMarg.


def _matlab_engine() -> Any:
    "The ``matlab.engine`` module, or an ``ImportError`` explaining how to install it."
    try:
        import matlab.engine  # type: ignore  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "Running MATLAB COM needs the MATLAB Engine API for Python, matching your MATLAB release, "
            "e.g. for R2024b: `pip install \"matlabengine==24.2.*\"`.") from err
    return matlab.engine


def start_matlab(com_file: Path, work_dir: Optional[Path] = None) -> Any:
    """
    Start a MATLAB session ready to run COM.

    Args:
        com_file: The MATLAB COM code (e.g. ``com_ieee8023_480.m``); its folder is added to the MATLAB path.

    Keyword Args:
        work_dir: MATLAB's working folder, against which COM resolves relative paths (e.g. its ``RESULT_DIR``).
            Default: None (i.e. - Python's current working directory.)

    Returns:
        The MATLAB engine; call its ``quit()`` when done.
    """
    eng = _matlab_engine().start_matlab()
    eng.addpath(str(Path(com_file).resolve().parent), nargout=0)
    eng.addpath(str(MFILES_DIR), nargout=0)
    eng.cd(str(Path(work_dir or Path.cwd()).resolve()), nargout=0)
    return eng


def _to_python(val: Any) -> Any:
    "Convert a value returned by the engine: MATLAB arrays become NumPy arrays; scalars and strings stay as is."
    if isinstance(val, (bool, int, float, complex, str)):
        return val
    return np.asarray(val).squeeze()


def run_com_matlab(
    chnl_sets: list[tuple[ChnlGrpName, ChnlSet]],
    cfg_sheet: Path,
    com_file: Path,
    eng: Optional[Any] = None,
) -> dict[ChnlGrpName, dict[ChnlSetName, COM_MATLAB_RESULTS]]:
    """
    Run COM on a list of grouped channel sets, using the MATLAB code.

    Args:
        chnl_sets: List of pairs, each consisting of:

            - ch_grp_name: The group name for this list of channel sets.
            - ch_sets: List of channel sets to run.

        cfg_sheet: Path to MS Excel configuration spreadsheet.
        com_file: Path to the MATLAB COM code (e.g. ``com_ieee8023_480.m``); its name is the COM function's.

    Keyword Args:
        eng: A MATLAB engine from ``start_matlab()``, to reuse across calls.
            Default: None (i.e. - Start one for this call, and quit it afterwards.)

    Returns:
        Dictionary, indexed by channel group name, containing dictionaries, indexed by channel set name, of:

            - "results": List, one per COM package test case, of dictionaries of COM's results
              (e.g. "channel_operating_margin_dB", "FOM", "sgm_TX", ...); empty if COM failed.
            - "stdout": MATLAB's console output.
            - "stderr": MATLAB's error output.
            - "error": The MATLAB error message, if COM failed; otherwise, ``None``.

    Raises:
        ImportError: If the MATLAB Engine API for Python isn't installed.
    """

    engine = _matlab_engine()
    own_eng = eng is None
    if eng is None:
        eng = start_matlab(com_file)
    results: dict[ChnlGrpName, dict[ChnlSetName, COM_MATLAB_RESULTS]] = {}
    try:
        for grp, ch_set in chnl_sets:
            lbl = ch_set['THRU'][0].stem
            out, err = io.StringIO(), io.StringIO()
            rslt: COM_MATLAB_RESULTS = {"results": [], "error": None}
            try:
                testcases = eng.pychopmarg_com(
                    Path(com_file).stem, str(Path(cfg_sheet).resolve()), str(Path(ch_set['THRU'][0]).resolve()),
                    [str(Path(f).resolve()) for f in ch_set['FEXT']], [str(Path(f).resolve()) for f in ch_set['NEXT']],
                    nargout=1, stdout=out, stderr=err)
                rslt["results"] = [{k: _to_python(v) for k, v in tc.items()} for tc in testcases]
            except engine.MatlabExecutionError as exc:  # COM failed on this channel set; carry on with the rest.
                rslt["error"] = str(exc)
            rslt["stdout"] = out.getvalue()
            rslt["stderr"] = err.getvalue()
            results.setdefault(grp, {})[lbl] = rslt
    finally:
        if own_eng:
            eng.quit()
    return results
