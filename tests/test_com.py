"""Tests for ``pychopmarg.com``."""

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
import pytest

from pychopmarg.com import COM
from pychopmarg.excel import get_com_params

ROOT = Path(__file__).parent.parent
CHNL = ROOT / "chnl_data"
# The tests' own copy of the COM configuration (Tx FFE off, for speed), independent of `config/`'s comparison sheet.
CFG = Path(__file__).parent / "data" / "config_com_ieee8023dj_test.xls"


@pytest.fixture(scope="module")
def example2_com():
    "COM for example2 (THRU + 2 FEXT + 3 NEXT): far too lossy for this 106.25 GBd configuration."
    ch_set = {"THRU": [CHNL / "example2_THRU.s4p"],
              "FEXT": [CHNL / f"example2_FEXT{n}.s4p" for n in (1, 2)],
              "NEXT": [CHNL / f"example2_NEXT{n}.s4p" for n in (1, 2, 3)]}
    with contextlib.redirect_stdout(io.StringIO()):
        com = COM(get_com_params(CFG), ch_set)
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


def test_noise_model_is_the_chosen_eqs(example2_com):
    """COM's Tx noise comes from the noise model of the EQ `opt_eq()` chose, not of the last one it tried (which once
    made it use the last CTLE setting's response: here, -2 dB, vs. the chosen -12 dB)."""
    com = example2_com
    assert com.gDC != com.com_params.g_DC[-1]  # (Else, this test can't tell.)
    assert np.allclose(com.theNoiseCalc.Hctf, com.calc_Hctf(com.gDC, com.gDC2))


def test_tx_noise_variance_per_178a_21(example2_com):
    """The Tx noise variance is 10^(-SNR_TX/10) * sum(h_tn(n)^2), by (178A-21) and Parseval: no sigma_X^2 factor
    (unlike crosstalk's (178A-20)), as in MATLAB COM by default."""
    nc = example2_com.theNoiseCalc
    var = nc.variance(nc.Stn())
    htn = nc.Stn_debug["_htn"]
    # (Within 0.1%: the folded PSD's length rounds; sigma_X^2 would be 44%.)
    assert var == pytest.approx(10**(-nc.snr_tx / 10) * (htn**2).sum(), rel=1e-3)


@pytest.mark.parametrize("phase", [0, 1, 17])
def test_jitter_noise_variance_per_178a_23(example2_com, phase):
    """The jitter noise variance is sigma_X^2 * (A_DD^2 + sigma_Rj^2) * sum(h_J(n)^2), by (178A-23/24) and Parseval,
    with h_J(n) the pulse response's slope (V/UI) at each sample of the cursor's phase. Phase zero (cursor on a
    whole-UI sample index) once paired the slopes of adjacent UIs, roughly halving this variance there."""
    nc = copy.copy(example2_com.theNoiseCalc)
    nspui = nc.nspui
    nc.ts_ix = (example2_com.fom_rslts['cursor_ix'] // nspui) * nspui + phase
    pr = nc.vic_pulse_resp
    ixs = np.arange(nc.ts_ix % nspui, len(pr), nspui)
    ixs = ixs[(ixs >= 1) & (ixs < len(pr) - 1)]
    hJ = (pr[ixs + 1] - pr[ixs - 1]) / 2 * nspui
    assert nc.variance(nc.Sjn) == pytest.approx(nc.varX * (nc.Add**2 + nc.sigma_Rj**2) * (hJ**2).sum(), rel=1e-3)
