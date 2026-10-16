"""Tests for ``pychopmarg.mlsd`` (P802.3dj/D3.3 178A.1.10)."""

import contextlib
import copy
import io

import numpy as np
import pytest
from scipy.stats import norm

from pychopmarg.mlsd import der_mlsd, gauss_pmf, inv_cdf, scale_pmf, solve_g_an

YSTEP = 1e-5
N = 4001  # (Grid: +/-20 mV.)
Y = (np.arange(N) - N // 2) * YSTEP


def moments(p):
    mean = (Y * p).sum()
    return mean, ((Y - mean)**2 * p).sum()


@pytest.mark.parametrize("a", [0.0, 0.3, 0.85, 1.0])
def test_scale_pmf_scales_the_random_variable(a):
    "p(y / a) / |a| (178A-54/55) is the PMF of a * Y: a PMF w/ a^2 times the variance (and a delta, for a = 0)."
    p = gauss_pmf(N, YSTEP, (2e-3)**2)
    q = scale_pmf(p, a)
    assert q.sum() == pytest.approx(1.0)
    assert moments(q)[1] == pytest.approx(a**2 * (2e-3)**2 + (YSTEP**2 / 12 if 0 < a < 1 else 0), rel=1e-2, abs=1e-12)


def test_solve_g_an_gives_the_penalty():
    "The added Rx noise makes 20 * log10(Pan^-1(DER0) / P^-1(DER0)) equal to dCOM_an (178A.1.10.1)."
    p_rest = np.zeros(N)
    p_rest[[N // 2 - 300, N // 2 + 300]] = 0.5  # (A little ISI.)
    var_G, var_rn, DER0, dCOM = (1e-3)**2, (0.5e-3)**2, 2e-4, 3.0
    g_an, p_an = solve_g_an(p_rest, YSTEP, var_G, var_rn, DER0, dCOM)
    p0 = solve_g_an(p_rest, YSTEP, var_G, var_rn, DER0, 1e-9)[1]
    assert g_an > 0
    # (Within the grid's resolution: one 10 uV step at ~4.5 mV is ~0.02 dB.)
    assert 20 * np.log10(inv_cdf(p_an, YSTEP, DER0) / inv_cdf(p0, YSTEP, DER0)) == pytest.approx(dCOM, abs=0.03)


def test_der_mlsd_gaussian_closed_form():
    """With no DFE tap (b = 0), white noise (rho = delta) and Gaussian p_an of std. dev. sigma, the sequence of length
    j has u_j^T u_j = j and p_j of variance j * sigma^2; so (178A-50) reduces to
    sum_j ((L-1)/L)^(j-1) * Phi(-As * sqrt(j) / sigma)."""
    sigma, As, L = 1e-3, 3.5e-3, 4
    p_an = gauss_pmf(N, YSTEP, sigma**2)
    rho = np.zeros(200)
    rho[0] = 1.0
    der, n_terms = der_mlsd(p_an, YSTEP, As, 0.0, rho, L)
    expected = sum(((L - 1) / L)**(j - 1) * norm.cdf(-As * np.sqrt(j) / sigma) for j in range(1, n_terms + 1))
    assert der == pytest.approx(expected, rel=0.02)
    assert 1 < n_terms < 200


def test_mlsd_com(example2_com_mlsd):
    """COM w/ MLSD (178A-49) is COM_DFE plus the MLSD's advantage, and never less than COM_DFE. (DER_MLSD, a union
    bound (178A-50), can exceed 1 on a channel this bad.)"""
    com, com_dfe, com_mlsd = example2_com_mlsd
    rslts = com.com_rslts
    assert com_mlsd == pytest.approx(max(com_dfe, com_dfe + rslts["delta_COM_mlsd"]))
    assert rslts["g_an"] > 0 and rslts["DER_MLSD"] > 0


@pytest.fixture(scope="module")
def example2_com_mlsd():
    "example2 (THRU only), w/ MLSD added to its COM_DFE after the fact."
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    from pychopmarg.com import COM  # pylint: disable=import-outside-toplevel
    from pychopmarg.excel import get_com_params  # pylint: disable=import-outside-toplevel
    root = Path(__file__).parent.parent
    ch_set = {"THRU": [root / "chnl_data" / "example2_THRU.s4p"], "FEXT": [], "NEXT": []}
    with contextlib.redirect_stdout(io.StringIO()):
        com = COM(get_com_params(Path(__file__).parent / "data" / "config_com_ieee8023dj_test.xls"), ch_set)
        com_dfe = com()
    com = copy.deepcopy(com)
    return com, com_dfe, com.calc_mlsd(com_dfe)
