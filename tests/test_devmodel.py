"""
Tests of the P802.3dj Annex 178A transmitter/receiver S-parameter models (`pychopmarg.devmodel`), against a NumPy
port of the MATLAB COM reference code (com_ieee8023_480.m: `make_full_pkg()`, `make_pkg()`, `synth_tline()`,
`combines4p()`, and the terminated transfer function of `s21_pkg()`).
"""

import numpy as np
import pytest

from pychopmarg.config import ieee_8023dj_d3p3 as dj
from pychopmarg.devmodel import (
    DeviceModel, DeviceTermination, gamma_term, s_device_termination, s_rx_model, s_tx_model)

F = np.arange(0, 100.01e9, 50e6)
R0 = dj.R0


# ---- NumPy port of the MATLAB reference ----------------------------------------------------------------------------
def m_combines4p(a, b):
    "Cascade of 2-ports, each given as (s11, s12, s21, s22)."
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    d = 1 - a22 * b11
    return (a11 + a12 * a21 * b11 / d, a12 * b12 / d, a21 * b21 / d, b22 + b21 * b12 * a22 / d)


def m_synth_tline(f, zc, z0, g, tau, d):
    f_ghz = f / 1e9
    with np.errstate(divide="ignore"):
        gamma2 = g[2] * (1 - 2j / np.pi * np.log(f_ghz)) + 2j * np.pi * tau
        gamma = g[0] + g[1] * (1 + 1j) * np.sqrt(f_ghz) + gamma2 * f_ghz
    gamma[f_ghz == 0] = g[0]
    rho = 0 if d == 0 else (zc - 2 * z0) / (zc + 2 * z0)
    e = np.exp(-d * gamma)
    s11 = rho * (1 - e**2) / (1 - rho**2 * e**2)
    s21 = (1 - rho**2) * e / (1 - rho**2 * e**2)
    return s11, s21, s21, s11


def m_shunt(f, c, z0):
    x = 1j * 2 * np.pi * f * c * z0
    return -x / (2 + x), 2 / (2 + x), 2 / (2 + x), -x / (2 + x)


def m_make_pkg(f, length, cpad, cball, zpkg, z0, g, tau, lcomp, cbump):
    f = np.where(f < np.finfo(float).eps, np.finfo(float).eps, f)
    s = m_shunt(f, cpad, z0)
    if lcomp > 0:
        x = 1j * 2 * np.pi * f * lcomp / z0
        s = m_combines4p(s, (x / (2 + x), 2 / (2 + x), 2 / (2 + x), x / (2 + x)))
    if cbump > 0:
        s = m_combines4p(s, m_shunt(f, cbump, z0))
    s = m_combines4p(s, m_synth_tline(f, zpkg, z0, g, tau, length))
    return m_combines4p(s, m_shunt(f, cball, z0))


def m_make_full_pkg(f, cd, ls, cb, cp, lens, zcs, z0, g, tau):
    "MATLAB `make_full_pkg()`, for a die ladder `cd`/`ls` and T-line segments `lens`/`zcs` (SI units)."
    n_ex = len(cd) - 1
    cpad = list(cd) + [0] * (len(lens) - 1)
    lcomp = list(ls) + [0] * (len(lens) - 1)
    cbump = [0] * n_ex + [cb] + [0] * (len(lens) - 1)
    cball = [0] * n_ex + [0] * (len(lens) - 1) + [cp]
    length = [0] * n_ex + list(lens)
    zpkg = [0] * n_ex + list(zcs)
    s = None
    for j in range(len(cpad)):
        blk = m_make_pkg(f, length[j], cpad[j], cball[j], zpkg[j], z0, g, tau, lcomp[j], cbump[j])
        s = blk if s is None else m_combines4p(s, blk)
    return s


def m_device(dev: DeviceModel):
    pkg = dev.package
    st0 = pkg.stages[0]
    return m_make_full_pkg(
        F, [c * 1e-9 for c in dev.termination.Cd], [ls * 1e-9 for ls in dev.termination.Ls],
        dev.termination.Cb * 1e-9, pkg.Cp * 1e-9, [s.zp for s in pkg.stages], [s.zc for s in pkg.stages], R0,
        (st0.gamma0, st0.a1, st0.a2), st0.tau)


def as_tuple(ntwk):
    return ntwk.s[:, 0, 0], ntwk.s[:, 0, 1], ntwk.s[:, 1, 0], ntwk.s[:, 1, 1]


# ---- Tests ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pkg_class,test", sorted(dj.ZP1_KR_C2C))
def test_tx_and_rx_models_match_matlab(pkg_class, test):
    """Tx model == MATLAB `make_full_pkg('TX', ...)`; Rx model == MATLAB's Rx package, port-reversed (as used in
    `s21_pkg()`), for every KR/C2C package class and test case (incl. class B's 4 stages and Tx/Rx lengths)."""
    for is_rx in (False, True):
        dev = dj.kr_c2c_device(pkg_class, test, is_rx)
        m11, m12, m21, m22 = m_device(dev)
        ours = as_tuple((s_rx_model if is_rx else s_tx_model)(F, dev, R0))
        expected = (m22, m21, m12, m11) if is_rx else (m11, m12, m21, m22)  # Rx: port-reversed
        for got, want in zip(ours, expected):
            np.testing.assert_allclose(got[1:], want[1:], rtol=1e-9, atol=1e-12)


def test_terminated_h21_matches_matlab():
    """Terminated transfer function (178A-3 / MATLAB `s21_pkg()`) of Tx model ** channel ** Rx model, for a synthetic
    lossy, mismatched channel and non-ideal terminations, matches the MATLAB formula."""
    import skrf as rf
    from pychopmarg.utility.sparams import sPkgTline
    chnl = rf.Network(s=sPkgTline(F, R0, 1e-3, 3e-4, 6e-3, 1e-3, [(85.0, 150.0)]).s, f=F, z0=2 * R0)
    tx, rx = dj.kr_c2c_device("B", 1, False), dj.kr_c2c_device("A", 2, True)
    total = s_tx_model(F, tx, R0) ** chnl ** s_rx_model(F, rx, R0)
    g1, g2 = gamma_term(50.0, R0), gamma_term(43.0, R0)
    s11, s12, s21, s22 = as_tuple(total)
    ours = s21 * (1 - g1) * (1 + g2) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * (s11 * s22 - s12 * s21))
    # MATLAB: combines4p(Tx, channel), then combines4p(that, reversed Rx package).
    mt = m_device(tx)
    mr = m_device(rx)
    c = as_tuple(chnl)
    a = m_combines4p(mt, c)
    a = m_combines4p(a, (mr[3], mr[2], mr[1], mr[0]))
    m11, m12, m21, m22 = a
    want = m21 * (1 - g1) * (1 + g2) / (1 - m11 * g1 - m22 * g2 - m21 * m12 * g1 * g2 + m11 * m22 * g1 * g2)
    np.testing.assert_allclose(ours[1:], want[1:], rtol=1e-9)


def test_termination_ladder_order():
    """The device termination is Cd(1), Ls(1), ..., Cd(N), Ls(N), then Cb (Figure 178A-4): swapping two stages'
    values changes the network."""
    t = dj.TERMINATION
    swapped = DeviceTermination(Cd=(t.Cd[2], t.Cd[1], t.Cd[0]), Ls=(t.Ls[2], t.Ls[1], t.Ls[0]), Cb=t.Cb)
    a = s_device_termination(F, t, R0).s
    b = s_device_termination(F, swapped, R0).s
    assert not np.allclose(a[:, 0, 0], b[:, 0, 0])


def test_models_are_differential_2r0():
    dev = dj.cr_device("HN")
    for ntwk in (s_tx_model(F, dev, R0), s_rx_model(F, dev, R0)):
        assert ntwk.nports == 2
        np.testing.assert_allclose(ntwk.z0, 2 * R0)


def test_cr_host_classes_and_combinations():
    "Table 179-23 host classes and Table 179-21 valid combinations (CA-A: 8, CA-B: 6, CA-C: 3, CA-D: 1)."
    assert {k: len(v) for k, v in dj.CR_VALID_COMBINATIONS.items()} == {"CA-A": 8, "CA-B": 6, "CA-C": 3, "CA-D": 1}
    for combos in dj.CR_VALID_COMBINATIONS.values():
        assert ("HH", "HH") not in combos
    hl = dj.cr_device("HL")
    assert len(hl.package.stages) == 2 and hl.package.stages[0].zp == 8.0 and hl.host.tline.zp == 22.0
    hh = dj.cr_device("HH", include_host=False)
    assert len(hh.package.stages) == 4 and hh.package.stages[0].zp == 45.0 and hh.host is None


def test_package_insertion_loss_is_plausible():
    "Class A Test 1 package + termination: little loss at low frequency, a few dB (not tens) at Nyquist (53.125 GHz)."
    s21 = s_tx_model(F, dj.kr_c2c_device("A", 1, False), R0).s[:, 1, 0]
    il_nyq = -20 * np.log10(abs(np.interp(53.125e9, F, abs(s21))))
    assert abs(s21[1]) > 0.95  # gamma0 (DC loss) of 5e-4/mm alone costs ~1.7% over 35 mm
    assert 1.0 < il_nyq < 10.0


def test_partial_host_matches_matlab_add_brd():
    """Partial host channel (178A-12) == MATLAB `add_brd()`'s Tx board model (C_0, board T-line, C_1), and the full
    CR Tx model == termination ** package ** partial host. (Non-zero C0/C1 exercise the shunt capacitors.)"""
    from pychopmarg.devmodel import PartialHost, TlineStage, s_partial_host
    tl = TlineStage(zc=92.5, zp=83.0, gamma0=0.0, a1=5.95e-4, a2=2.6e-5, tau=5.79e-3)
    host = PartialHost(C0=20e-6, tline=tl, C1=10e-6)
    ours = as_tuple(s_partial_host(F, host, R0))
    f = np.where(F < np.finfo(float).eps, np.finfo(float).eps, F)
    tline = m_synth_tline(F, 92.5, R0, (0.0, 5.95e-4, 2.6e-5), 5.79e-3, 83.0)
    want = m_combines4p(m_combines4p(m_shunt(f, 20e-15, R0), tline), m_shunt(f, 10e-15, R0))
    for got, exp in zip(ours, want):
        np.testing.assert_allclose(got[1:], exp[1:], rtol=1e-9, atol=1e-12)
    dev = dj.cr_device("HN")
    full = as_tuple(s_tx_model(F, dev, R0))
    no_host = DeviceModel(termination=dev.termination, package=dev.package, host=None, Rd=dev.Rd)
    want = m_combines4p(as_tuple(s_tx_model(F, no_host, R0)), as_tuple(s_partial_host(F, dev.host, R0)))
    for got, exp in zip(full, want):
        np.testing.assert_allclose(got[1:], exp[1:], rtol=1e-9, atol=1e-12)


def test_device_model_resolver():
    "`device_model()` resolves every group's cases, and rejects invalid ones."
    from pychopmarg.config.ieee_8023dj_d3p3 import PKG_CASES, device_model
    for group, cases in PKG_CASES.items():
        for case in cases:
            for is_rx in (False, True):
                assert isinstance(device_model(group, case, is_rx), DeviceModel)
    assert device_model("KR", "B1", True).package.stages[0].zp == 44.0   # class B Test 1, Rx
    assert device_model("CR", "HN", False, include_host=False).host is None
    assert device_model("C2M", "module", False).package is None
    with pytest.raises(ValueError):
        device_model("KR", "HL", False)


def test_terminated_h21_is_formula_178a3():
    from pychopmarg.devmodel import terminated_h21
    import skrf as rf
    ntwk = rf.Network(s=s_tx_model(F, dj.kr_c2c_device("A", 1, False), R0).s, f=F, z0=2 * R0)
    g1, g2 = gamma_term(50.0, R0), gamma_term(40.0, R0)
    s11, s12, s21, s22 = as_tuple(ntwk)
    want = s21 * (1 - g1) * (1 + g2) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * (s11 * s22 - s12 * s21))
    np.testing.assert_allclose(terminated_h21(ntwk, g1, g2), want)
