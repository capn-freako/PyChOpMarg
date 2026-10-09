"""Tests for ``pychopmarg.com``."""

import contextlib
import io
from pathlib import Path

import pytest

from pychopmarg.com import COM
from pychopmarg.excel import get_com_params

ROOT = Path(__file__).parent.parent
CHNL = ROOT / "chnl_data"


@pytest.fixture(scope="module")
def example2_com():
    "COM for example2 (THRU + 2 FEXT + 3 NEXT): far too lossy for this 106.25 GBd configuration."
    ch_set = {"THRU": [CHNL / "example2_THRU.s4p"],
              "FEXT": [CHNL / f"example2_FEXT{n}.s4p" for n in (1, 2)],
              "NEXT": [CHNL / f"example2_NEXT{n}.s4p" for n in (1, 2, 3)]}
    with contextlib.redirect_stdout(io.StringIO()):
        com = COM(get_com_params(ROOT / "config" / "config_com_ieee8023dj_PyChOpMarg_vs_MATLAB.xls"), ch_set)
        com()
    return com


def test_com_not_floored(example2_com):
    """Interference far above the signal gives a COM far below 0 dB. The noise PMFs' voltage grid once spanned only
    +/-1.1 * As, capping `Ani` there, i.e. flooring COM at -0.83 dB. (MATLAB COM 4.80 gives -16.85 dB here.)"""
    rslts = example2_com.com_rslts
    assert rslts["Ani"] > 1.1 * rslts["As"]
    assert rslts["COM"] < -10.0


def test_noise_pmfs_cover_their_support(example2_com):
    "The interference PMF is a PMF, and its grid reaches beyond it: no probability is lost at the edges."
    rslts = example2_com.com_rslts
    assert rslts["py"].sum() == pytest.approx(1.0)
    assert rslts["py"][:10].sum() < 1e-12 and rslts["py"][-10:].sum() < 1e-12
