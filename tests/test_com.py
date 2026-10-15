"""Tests for ``pychopmarg.com``."""

import contextlib
import copy
import dataclasses
import io
import itertools
from pathlib import Path

import numpy as np
import pytest
import skrf as rf

import pychopmarg.optimize as optimize
from pychopmarg.com import COM
from pychopmarg.excel import get_com_params
from pychopmarg.utility import sdd_21

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
    made it use the last CTLE setting's response)."""
    com = example2_com
    cp = com.com_params
    assert (com.gDC, com.gDC2) != (cp.g_DC[-1], cp.g_DC2[-1])  # (Else, this test can't tell.)
    assert np.allclose(com.theNoiseCalc.Hctf, com.calc_Hctf(com.gDC, com.gDC2))


def test_mmse_keeps_every_precursor(example2_com):
    """The MMSE solve sees every pre-cursor sample of the pulse response, unaltered, as in (178A-32) and MATLAB COM.
    It once kept only `dw` of them, and zeroed samples below 0.1% of the peak (in the pulse response itself)."""
    com = example2_com
    r = com.mmse_rslt
    dh, first_samp = divmod(r["cursor_ix"], com.nspui)
    assert r["d"] == dh + com.nRxPreTaps
    assert np.array_equal(r["h"], r["vic_pulse_resp"][first_samp::com.nspui])


def test_com_isi_has_every_precursor(example2_com):
    """COM's ISI is every sample of the post-FFE pulse response at the cursor's phase, pre-cursors included, as in
    (178A-42) and MATLAB COM. It once kept only 5 pre-cursor UIs."""
    com = example2_com
    rslts = com.com_rslts
    curs_uis, curs_ofst = divmod(rslts["cursor_ix"], com.nspui)
    samps = rslts["pulse_resps"][0][curs_ofst::com.nspui]
    assert len(rslts["hISI"]) == len(samps)
    assert np.array_equal(rslts["hISI"][:curs_uis], samps[:curs_uis])
    assert rslts["hISI"][curs_uis] == 0


def test_crosstalk_phase_chosen_at_rx_ffe_input(example2_com):
    """Under MMSE, each aggressor's sampling phase in COM is the one chosen at the Rx FFE input (178A-20), as in
    MATLAB COM, not after it. (It was once re-chosen after the Rx FFE.)"""
    nc = example2_com.theNoiseCalc
    assert example2_com.com_rslts["xt_phases"] == [nc.agg_phase(pr) for pr in nc.agg_pulse_resps]
    # Here, the phases chosen before and after the Rx FFE coincide; so, shift the former, to tell them apart.
    com = copy.deepcopy(example2_com)
    nc = com.theNoiseCalc
    shifted = [(nc.agg_phase(pr) + 7) % com.nspui for pr in nc.agg_pulse_resps]
    nc.agg_phase = lambda pr: shifted[[id(p) for p in nc.agg_pulse_resps].index(id(pr))]
    with contextlib.redirect_stdout(io.StringIO()):
        com.calc_noise(cursor_ix=com.fom_rslts["cursor_ix"])
    assert com.com_rslts["xt_phases"] == shifted


def test_signal_path_includes_tx_rise_time_filter(example2_com):
    """The signal path's transfer function is H_txffe * Ht * H21 * Hr * Hctf (* H_rxffe), as per (178A-14) and MATLAB
    COM (w/ `FORCE_TR`): it includes the Tx rise time filter, Ht (93A-46). (It once didn't, inflating As ~15%.)"""
    com = example2_com
    H21 = com.chnls[0][1]
    Hctf = com.calc_Hctf(com.gDC, com.gDC2)
    expected = com.Htx(com.tx_ix) * com.Ht * H21 * com.Hr * Hctf * com.Hffe_Rx(com.rx_taps)
    assert np.allclose(com.H(H21, com.tx_ix, Hctf=Hctf, rx_taps=com.rx_taps), expected)


def test_tx_noise_variance_per_178a_21(example2_com):
    """The Tx noise variance is 10^(-SNR_TX/10) * sum(h_tn(n)^2), by (178A-21) and Parseval: no sigma_X^2 factor
    (unlike crosstalk's (178A-20)), as in MATLAB COM by default."""
    nc = example2_com.theNoiseCalc
    var = nc.variance(nc.Stn())
    htn = nc.Stn_debug["_htn"]
    # (Within 0.1%: the folded PSD's length rounds; sigma_X^2 would be 44%.)
    assert var == pytest.approx(10**(-nc.snr_tx / 10) * (htn**2).sum(), rel=1e-3)


def test_noise_calc_left_at_chosen_sampling_time(example2_com):
    """After the MMSE timing search, the noise calculator samples at the chosen cursor, which COM's final Tx noise
    (`calc_noise()`, via `Stn()`) relies on. (It was once left at the last offset tried.)"""
    com = example2_com
    assert com.theNoiseCalc.ts_ix == com.fom_rslts["cursor_ix"]


def test_tx_noise_pulse_has_the_signals_phase(example2_com):
    """The Tx noise pulse response (178A-22) is formed as the signal's is, so it's sampled at the signal's phase.
    (It was once formed w/ a causal moving sum, which delayed it ~half a UI.) Here, w/ no Tx FFE, it's the victim's
    pre-Rx-FFE pulse response, scaled and shifted by whole UIs (the FFE frames' fixed delays), which don't matter."""
    nc = copy.deepcopy(example2_com.theNoiseCalc)
    nc.Stn()
    pulse, vic = nc.Stn_debug["htn_pulse"][:len(nc.vic_pulse_resp)], nc.vic_pulse_resp
    shift = int(np.argmax(vic) - np.argmax(pulse))
    assert shift % nc.nspui == 0
    assert np.allclose(np.roll(pulse, shift) / pulse.max(), vic / vic.max(), atol=1e-6)


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


def test_local_search_follows_matlab(monkeypatch):
    """With `local_search` > 0, `opt_eq()` visits exactly the (g_DC, g_DC_HP, Tx FFE) candidates that MATLAB COM's
    `optimize_fom()` loop does, under its `OptFom_Local_Search()` rule, and picks the same one, for a synthetic FOM."""
    cp = dataclasses.replace(get_com_params(CFG), c0_min=0.54, local_search=2)
    com = COM(cp, {"THRU": [CHNL / "example2_THRU.s4p"], "FEXT": [], "NEXT": []})
    target = np.array([0.1, -0.1, 0.2, -0.3, -0.2, 0., 0.])

    def fom(gDC, gDC2, taps):  # A bowl, roughened so that the best so far wanders.
        return (-((gDC + 6) / 3)**2 - (gDC2 + 1)**2 - (((taps - target) / 0.15)**2).sum()
                + 0.3 * np.sin(1e3 * (7.3 * gDC + 3.1 * gDC2 + taps @ [1.1, 2.3, 3.7, 5.9, 7.1, 0, 0])))

    visited = []

    def fake_calc_fom(tx_ix, Hctf, **_):
        taps = com._tx_combs[tx_ix]  # pylint: disable=protected-access
        visited.append((*Hctf, *np.round(taps, 6)))
        com.fom_rslts = {k: np.ones(com.nRxTaps) for k in ('rx_taps', 'pr_samps', 'dfe_tap_weights', 'vic_pulse_resp')}
        com.fom_rslts.update({k: 0 for k in ('cursor_ix', 'vic_peak_loc', 'As', 'varTx', 'varISI', 'varJ', 'varXT', 'varN')})
        return fom(*Hctf, taps)

    monkeypatch.setattr(com, "calc_Hctf", lambda gDC, gDC2: (gDC, gDC2))
    monkeypatch.setattr(com, "calc_fom", fake_calc_fom)
    assert com.opt_eq()

    # MATLAB COM's loops and `OptFom_Local_Search()`, transliterated (1-based indices, as there).
    tap_vals = [np.linspace(lo, hi, int(round((hi - lo) / step)) + 1) if step else np.zeros(1)
                for lo, hi, step in zip(cp.tx_taps_min, cp.tx_taps_max, cp.tx_taps_step)]
    lens = [len(v) for v in tap_vals]
    sweep = sorted([k + 1 for k, n in enumerate(lens) if n > 1], key=lambda kv: -lens[kv - 1])
    grid = [ix for ix in itertools.product(*[range(1, n + 1) for n in lens])
            if 1 - sum(abs(tap_vals[k][i - 1]) for k, i in enumerate(ix)) >= cp.c0_min]
    expected, best = [], None
    for ctle_index, gDC in enumerate(cp.g_DC, start=1):
        for g_LP_index, gDC2 in enumerate(cp.g_DC2, start=1):
            for tx_index in grid:
                if best is not None:
                    skip_it = False
                    for kv in sweep:
                        previous_loop_val = g_LP_index if kv == 1 else tx_index[kv - 2]
                        if previous_loop_val > 1 and abs(tx_index[kv - 1] - best["txffe_index"][kv - 1]) > 2:
                            skip_it = True
                            break
                    if not skip_it and ctle_index > 1 and abs(g_LP_index - best["G_high_pass"]) > 2:
                        skip_it = True
                    if skip_it:
                        continue
                taps = np.array([tap_vals[k][i - 1] for k, i in enumerate(tx_index)])
                expected.append((gDC, gDC2, *np.round(taps, 6)))
                this_fom = fom(gDC, gDC2, taps)
                if best is None or this_fom > best["FOM"]:
                    best = {"FOM": this_fom, "txffe_index": tx_index, "G_high_pass": g_LP_index,
                            "gDC": gDC, "gDC2": gDC2, "taps": taps}

    assert visited == expected
    assert len(visited) < len(cp.g_DC) * len(cp.g_DC2) * len(grid)  # (It did skip some.)
    assert (com.gDC, com.gDC2) == (best["gDC"], best["gDC2"])
    assert np.allclose(com._tx_combs[com.tx_ix], best["taps"])  # pylint: disable=protected-access


@pytest.mark.parametrize("center", [0, 5, -3])
def test_middle_timing_search_follows_matlab(example2_com, monkeypatch, center):
    """With `ts_window` > 0, `mmse()` tries exactly the cursor offsets that MATLAB COM's "middle" search
    (`OptFom_Itick_LocalSearch()`, with `LOCAL_SEARCH` = `ts_window`) would, working outward from `ts_center`."""
    nc = copy.copy(example2_com.theNoiseCalc)
    cp = example2_com.com_params
    args = (nc, example2_com.nRxTaps, example2_com.nRxPreTaps, len(cp.dfe_min), cp.RLM, cp.L,
            cp.dfe_min, cp.dfe_max, cp.rx_taps_min, cp.rx_taps_max)
    peak = int(np.argmax(nc.vic_pulse_resp))
    tried: list[tuple[int, float]] = []
    log10 = optimize.log10
    monkeypatch.setattr(optimize, "log10", lambda x: (tried.append((nc.ts_ix - peak, 20 * log10(x))), log10(x))[1])
    full = optimize.mmse(*args)
    fom_at = dict(tried)
    assert sorted(fom_at) == list(range(-16, 16)) and full["n_ts_evals"] == 32
    tried.clear()
    rslt = optimize.mmse(*args, ts_window=2, ts_center=center)

    # MATLAB COM's "middle" itick loop, transliterated (about `center`, rather than 0).
    expected, pos, neg = [], None, None
    for itick in sorted(range(-16, 16), key=lambda k: abs(k - center)):
        if ((itick >= center and pos is not None and abs(pos[1] - itick) >= 2)
                or (itick <= center and neg is not None and abs(neg[1] - itick) >= 2)):
            continue
        expected.append(itick)
        if itick >= center and (pos is None or fom_at[itick] > pos[0]):
            pos = (fom_at[itick], itick)
        if itick <= center and (neg is None or fom_at[itick] > neg[0]):
            neg = (fom_at[itick], itick)

    assert [ofst for ofst, _ in tried] == expected
    assert rslt["n_ts_evals"] == len(expected) < 32
    assert rslt["cursor_ix"] - peak == max(expected, key=lambda k: (fom_at[k], -expected.index(k)))


def test_fext_uses_its_own_package():
    "Each channel type gets its own end's package lengths, as in MATLAB COM: changing `zp_fext` changes only FEXT."
    ch_set = {"THRU": [CHNL / "example2_THRU.s4p"],
              "FEXT": [CHNL / "example2_FEXT1.s4p"],
              "NEXT": [CHNL / "example2_NEXT1.s4p"]}
    cp = get_com_params(CFG)
    with contextlib.redirect_stdout(io.StringIO()):
        base = {ntype: H for (_, ntype), H in COM(cp, ch_set).chnls}
        alt = {ntype: H for (_, ntype), H in COM(dataclasses.replace(cp, zp_fext=[10.0, 1.8]), ch_set).chnls}
    assert np.array_equal(alt["THRU"], base["THRU"])
    assert np.array_equal(alt["NEXT"], base["NEXT"])
    assert not np.allclose(alt["FEXT"], base["FEXT"])


def test_channel_impulse_response_truncated_as_matlab(example2_com):
    """With `chnl_interp` = "vtf" (default), as MATLAB COM: each channel's end-to-end impulse response (of Ht * H21 * Hr)
    is zero after its last sample above 1e-3 of its peak."""
    com = example2_com
    assert com.chnl_interp == "vtf"
    for (_, ntype), H21 in com.chnls:
        h = np.fft.irfft(com.Ht * H21 * com.Hr)
        last = np.nonzero(abs(h) > 1e-3 * abs(h).max())[0][-1]
        assert last < len(h) - 1, ntype
        assert np.allclose(h[last + 1:], 0, atol=1e-9 * abs(h).max()), ntype


@pytest.mark.parametrize("chnl_interp", ["vtf", "t"])
def test_add_pkg_survives_a_zero_s21(example2_com, monkeypatch, chnl_interp):
    """A channel whose S21 is exactly zero somewhere (e.g., a weakly coupled aggressor, written with too few decimal
    places) can't be interpolated as T parameters, which need 1/S21. The legacy "t" mode of `add_pkg()` falls back
    to S parameters there; the default, "vtf", never needs T parameters."""
    fext = sdd_21(rf.Network(str(CHNL / "example2_FEXT1.s4p")))
    fext.s[-4, 1, 0] = fext.s[-4, 0, 1] = 0  # (As in the CR group's HN_3in_DAC_Z_1p5m_HL_2in_FEXT_TP0_Tx8_to_TP5_Rx7.)
    with pytest.raises(np.linalg.LinAlgError):
        _ = fext.t
    monkeypatch.setattr(example2_com, "chnl_interp", chnl_interp)
    _, H21 = example2_com.add_pkg((fext, "FEXT"))
    assert np.all(np.isfinite(H21))
