# PyChOpMarg deviations from P802.3dj/D3.3 and MATLAB COM 4p17p0

This document tracks where PyChOpMarg computes final or intermediate results differently than:

- **D3.3**: IEEE P802.3dj/D3.3, Annex 178A (the COM reference), and
- **MATLAB**: the IEEE 802.3 MATLAB COM reference code, version 4p17p0.

It came from an audit of the whole calculation path on 2026-10-14 (after v3.2.0, PR #27).
Line numbers refer to `main` at `fd4abd2`.

**Status key:** `[ ]` open · `[~]` in progress · `[x]` resolved · `[-]` won't fix (keep the deviation, with a reason)

When you resolve an item, check it off, and record the commit and any measured effect on the
PyChOpMarg-vs-MATLAB comparison (`notebook/PyChOpMarg_vs_MATLAB.ipynb`).

## Reference channel set

From A5 on, effects are measured on a fixed **reference set** of 40 channels, 10 per group, from the notebook's
200-channel run. They're listed in `notebook/reference_channels.json`; the notebook's `ChnlSel.REFERENCE` mode
runs them and saves MATLAB's results to `notebook/reference_matlab_results.pickle` (not committed). (A1–A4 were
measured on an earlier, overlapping set of 40, against MATLAB results from before the sheet's latest edits.)

Baseline (after A4; MATLAB COM 4p17p0 run 2026-10-15, comparison sheet as committed), PyChOpMarg − MATLAB:

| Quantity | Mean | Std | Range |
|---|---|---|---|
| COM (dB) | +0.33 | 0.24 | +0.04 … +0.98 |
| FOM (dB) | +0.31 | 0.24 | +0.02 … +0.99 |
| As (ratio) | 1.145 | 0.099 | 0.90 … 1.41 |
| Ani (ratio) | 1.102 | 0.087 | 0.81 … 1.29 |
| σ ISI, σ_rjit, σ_XT (ratio) | 1.16, 1.15, 1.12 | | |
| Cursor offset (samples) | +1.18 | rms 4.29 | exact on 1/40 |

Same CTLE as MATLAB on 13/40. The FOM is already high by about as much as COM, so the gap arises in the EQ
optimization rather than the final COM calculation. Signal and noise are both about 10–15% larger than MATLAB's,
a common gain that largely cancels in COM.

---

## A. Differences from both MATLAB and D3.3 that affect today's comparison results

### [x] A1. MMSE solve keeps only `dw` pre-cursor UIs, and zeroes small samples

- **Resolved (2026-10-14):** `mmse()` now uses every pre-cursor sample, unaltered (`d = dh + dw`, as in MATLAB).
  The zeroing also turned out to write into `theNoiseCalc.vic_pulse_resp` (`h` was a strided view of it); that's
  gone too. Test: `test_mmse_keeps_every_precursor`.
- **Measured effect:**
  - CR `HN_3in_DAC_Z_1p5m_HL_2in_thru_TP0_Tx7_to_TP5_Rx7`: same cursor and EQ, FOM −0.009 dB, COM unchanged.
  - `example2`: COM −17.16 → −16.79 dB (MATLAB COM 4.80: −16.85 dB). Its CTLE choice moved (−8/−1 → −2/−3 dB).
  - 40 channels (10 per group), thru only, at MATLAB's CTLE: peak-FOM error vs. MATLAB, max 0.879 → 0.739 dB
    (mean −0.08 dB, unchanged).
  - **It did not explain the cursor-timing bias.** PyChOpMarg's best offset minus MATLAB's: mean +1.45 → +1.48,
    rms 1.70 → 1.70 samples. Without jitter, the bias is about half (mean +0.70, rms 1.36). (A3 turned out not to
    affect the timing search; see there.)

**Original finding:**


- **Where:** `src/pychopmarg/optimize.py:320-329` (`mmse()`)
- **PyChOpMarg:** Truncates `h` to exactly `dw` (5) pre-cursor UIs (`dh = dw`) before forming `HᵀH`.
  It also sets every sample below 0.1% of `max(h)` to zero.
- **MATLAB / D3.3:** MATLAB's `MMSE()` uses *all* pre-cursor samples (`dh` = everything before the cursor).
  D3.3 (178A-32) uses the whole `h`. Neither zeroes small samples.
- **Why it matters:** Early pre-cursor ISI is invisible to PyChOpMarg's MSE, so sampling late costs it nothing.
  This is the leading suspect for the unexplained late cursor-timing bias: PyChOpMarg picks a cursor 0–3 samples
  later than MATLAB (rms 1.7 samples), even with ISI only.
- **Suggested fix:** Keep all pre-cursors and drop the 0.1% zeroing. Then re-run the comparison and check whether
  the cursor-timing scatter tightens.

### [x] A2. Final COM ISI keeps only 5 pre-cursor UIs

- **Resolved (2026-10-15):** `calc_noise()`'s ISI is now every sample of the post-FFE pulse at the cursor's phase
  (as MATLAB's `get_pdf()`), including the post-cursor tail beyond 2048 UI that was also cut. Test:
  `test_com_isi_has_every_precursor`.
- **Measured effect** (same 40 channels and MATLAB results as A3): COM fell 0–0.21 dB per channel (mean 0.06 dB).
  COM − MATLAB's COM went from +0.005 ± 0.27 to −0.057 ± 0.26 dB.
- **Open observation:** PyChOpMarg's `sigma_ISI` is about 1.23× MATLAB's `sgm_isi` (range 0.78–1.54), both before
  and after this change. MATLAB's `sgm_isi` may come from its FOM pass (`OptFom_Calc_Noise`), not from the final
  COM's ISI, so this needs checking before reading it as a discrepancy.

**Original finding:**

- **Where:** `src/pychopmarg/com.py:1105` (`calc_noise()`: `n_pre = min(5, curs_uis)`; the comment cites MATLAB v2.60)
- **MATLAB / D3.3:** 4p17p0 (`get_PSDs` COMPUTE_COM pass, `get_pdf`) and 178A-42 use every pre-cursor sample.

### [x] A3. Jitter slopes taken only from the cursor onward, and magnitude-filtered

- **Resolved (2026-10-15):** `calc_hJ(pulse_resp, cursor_ix, nspui)` now returns one central-difference slope per
  UI at the cursor's phase, pre-cursors included, unfiltered (as `NoiseCalc.Sjn` and MATLAB's `h_J` do). Its
  `As` and `rel_thresh` arguments are gone. The dual-Dirac PMF still drops tiny samples, through `filt_pr_samps`
  (see F1). Tests: `TestCalcHJ` in `tests/test_probability.py`.
- **Not a cause of the cursor-timing bias:** The MMSE timing search's jitter comes from `NoiseCalc.Sjn`, which
  already used every UI. `calc_hJ` only feeds the final COM (`calc_noise()`) and the PRZF FOM.
- **Measured effect** (40 channels, 10 per group, vs. MATLAB results saved 2026-10-11 with the comparison sheet):

  | | old | new |
  |---|---|---|
  | σ_rjit / MATLAB's `sgm_rjit` (mean, range) | 0.91 (0.74–1.09) | 1.07 (0.86–1.20) |
  | σ(p_DD) / MATLAB's `sgm_p_DD` (mean, range) | 0.91 (0.74–1.09) | 1.07 (0.86–1.20) |
  | COM − MATLAB's COM, mean ± std | +0.25 ± 0.30 dB | **+0.005 ± 0.27 dB** |

  COM fell 0.05–0.65 dB per channel (mean 0.24 dB). The remaining 7% excess jitter may come from the late
  cursor timing (steeper slopes there).

**Original finding:**

- **Where:** `src/pychopmarg/utility/probability.py:133-156` (`calc_hJ()`: `range(cursor_ix, ...)`, plus a
  `rel_thresh` = 0.1%·As filter). Used at `src/pychopmarg/com.py:832` and `:1097`.
- **MATLAB / D3.3:** 178A.1.9.3 / 178A-23 and MATLAB (`OptFom_Calc_Noise`, `h_J`) take slopes at *every* UI,
  pre-cursors included, with no magnitude filter.
- **Affects:** `pJ` (DD jitter, `hDD`) and `varJ` (Rj) in `calc_noise()`, and the jitter PSD in the FOM search.

### [x] A4. Sheet `z_p (TX/RX/NEXT/FEXT)` rows are ignored

- **Resolved (2026-10-15):** `COMParams` has new `zp_tx`, `zp_rx`, `zp_next` and `zp_fext` fields: per-segment
  package lengths (mm) for the selected test case. `None` falls back to `[z_p[zp_sel], z_pB]`. `get_com_params()`
  fills them from the sheet's `z_p (…)` rows (one row per segment, one column per test case), at the column
  `z_p select` picks. It warns if more than one case is selected (see B2). `_build_legacy_pkg()` builds a separate
  package for each end, and FEXT aggressors now get the FEXT package. (Before, they got the Tx package.)
  Tests: `test_per_end_package_lengths`, `test_zp_select_picks_test_case`, `test_fext_uses_its_own_package`.
- **Measured effect** (same 40 channels and MATLAB results as A2/A3; sheet: Tx/NEXT/FEXT 34 mm, Rx 32 mm):
  - COM rose 0.33 dB on average (−0.36 to +0.64 dB), and the CTLE choice changed on 11 of 40 channels.
  - COM − MATLAB's COM went from −0.06 ± 0.26 to **+0.28 ± 0.16 dB**: tighter scatter, but now a clear
    optimistic offset.
- **Why so large:** Whether the two packages are the same length matters, not the 1 mm shift. On CR
  `1mOSFPDAC_TP0TP5_25p9dB_PCBHost_4p9dB_THRU`:

  | Tx / Rx (mm) | 33 / 33 | 34 / 34 | 32 / 32 | 34 / 32 | 32 / 34 |
  |---|---|---|---|---|---|
  | COM (dB) | 2.16 | 2.25 | 2.59 | 2.79 | 2.79 |

  Identical packages at the two ends make their reflections add coherently. A 2 mm mismatch breaks that up.
  (33 / 33 reproduces the old code's result exactly.)

**Original finding:**

- **Where:** `src/pychopmarg/excel.py:31-50, 186`
- **PyChOpMarg:** The sheet's `z_p (TX)`, `z_p (RX)`, `z_p (NEXT)`, `z_p (FEXT)` and `z_p select` rows are parsed but
  never used. Package lengths come from the `IEEE_8023dj` default `z_p` = [33, 1.8] mm, the same at both ends.
- **MATLAB:** Uses the sheet's values. The comparison sheet has Tx 34 mm and Rx 32 mm (each with 1.8 mm).
- **Why it matters:** The total length is the same, so the effect is probably small, but it isn't the same model.

### [x] A5. Aggressor sampling phase chosen after the Rx FFE

- **Resolved (2026-10-15):** Under MMSE, `calc_noise()` now samples each aggressor at the phase chosen at the Rx FFE
  input (new `NoiseCalc.agg_phase()`, which `Sxn()` also uses), as in MATLAB's `get_PSDs()`/`get_pdf()`. PRZF
  keeps the post-FFE choice, as MATLAB's non-MMSE path does. The phases are in `com_rslts['xt_phases']`. Test:
  `test_crosstalk_phase_chosen_at_rx_ffe_input`.
- **Measured effect** (reference set): negligible. COM changed −0.016 to +0.008 dB (mean −0.001 dB), and no CTLE
  choice changed. The two phase choices almost always coincide.

**Original finding:**

- **Where:** `src/pychopmarg/com.py:1124` (`calc_noise()`)
- **PyChOpMarg:** Picks each aggressor's phase by maximum energy of its pulse *after* the Rx FFE.
- **MATLAB / D3.3:** Pick it from the pulse *before* the Rx FFE (MATLAB's `iphase` from `get_PSDs`; D3.3's `ts(k)`
  per 178A-20).

### [x] A6. Channel interpolation and extrapolation

- **Resolved (2026-10-15):** New default, `COM.chnl_interp` = `"vtf"`, as MATLAB (`s21_pkg()`,
  `s21_to_impulse_DC()`):
  1. The packages (`_pkg_models()`, now buildable at any frequencies) are cascaded with the raw channel, and the
     voltage transfer function (93A-18) formed (`calc_vtf()`), at the channel file's own frequencies.
  2. Ht · VTF · Hr is interpolated onto the system grid by `interp_vtf()`, a port of MATLAB's `interp_Sparam()`
     with its default policies, `linear_trend_to_DC` and `extrap_cubic_to_dc_linear_to_inf`.
  3. Its impulse response is truncated after its last sample above `COM.ir_trunc_thresh` (1e-3) of its peak.
  4. Ht · Hr are divided back out (`H()` and `NoiseCalc` restore them).

  The legacy modes, `"t"` (T basis, falling back to S where S21 vanishes) and `"s"`, remain selectable. The
  unequalized pulse responses (`gen_pulse_resps(apply_eq=False)`) now include Ht and Hr, as MATLAB's
  `uneq_pulse_response` does. Tests: `tests/test_filter.py`, `test_channel_impulse_response_truncated_as_matlab`,
  `test_add_pkg_survives_a_zero_s21` (both modes).
- **Measured effect** (reference set, all three modes, after A9):

  | vs. `"t"` | Max \|ΔCOM\| | Max \|ΔFOM\| | As ratio | Other |
  |---|---|---|---|---|
  | `"s"` | < 0.00001 dB | < 0.00001 dB | 1.00000 | none |
  | `"vtf"` | 0.016 dB | 0.019 dB | 0.994 … 1.002 | cursor moved on 1 channel; CTLE on 1 |

  COM − MATLAB's COM: −0.156 ± 0.092 dB (`"t"`, `"s"`) vs. −0.150 ± 0.091 dB (`"vtf"`). So the interpolation was
  not a source of disagreement with MATLAB, and the T vs. S basis question is moot. This was the comprehensive
  comparison the S→T ToDo called for. `"vtf"` is the default because it matches MATLAB's processing and never
  needs 1/S21.

**Original finding:**

- **Where:** `src/pychopmarg/com.py:448-476` (`add_pkg()`)
- **PyChOpMarg:** Interpolates the raw channel S-parameters onto the system frequency grid in the T basis (cubic,
  polar). Holds the last value constant above the data's maximum frequency. Then cascades the package.
  (It falls back to the S basis where S21 is exactly zero.)
- **MATLAB:** Interpolates the final end-to-end transfer function (`s21_to_impulse_DC` / `interp_Sparam`).
  Magnitude uses `linear_trend_to_DC` with a high-frequency trend. Phase uses linear interpolation with linear
  group-delay extrapolation. It also truncates the impulse response at 1e-3 of its peak by default. (Causality
  enforcement is off by default.)
- **Note:** Tied to the S→T elimination ToDo in `add_pkg()`'s docstring. Before switching bases, compare S vs. T
  across all channel groups (pulse responses, FOM, COM, chosen EQ, agreement with MATLAB).

### [x] A7. Tx-noise pulse phase

- **Resolved (2026-10-15):** `Stn()` now forms the Tx-noise pulse as the signal's is (`COM.pulse_resp()`):
  `irfft(nspui · sinc(f·Tb) · Htn)`, a zero-phase, 1 UI wide rectangle. Its sampling phase now matches the
  signal's. (On `example2`, the old pulse sat 16 samples, half a UI, off; the new one is off by whole UIs only,
  the FFE frames' fixed delays, which don't affect the PSD.) Test: `test_tx_noise_pulse_has_the_signals_phase`.
- **Measured effect** (reference set, vs. after A6):

  | PyChOpMarg − MATLAB | After A6 | After A7 |
  |---|---|---|
  | COM (dB) | −0.150 ± 0.091 | −0.171 ± 0.094 |
  | Cursor offset (samples) | mean +0.30, rms 3.42 | **mean +0.07**, rms 3.32 |
  | σ_Tx (ratio) | 0.994 | 1.013 |
  | Same CTLE as MATLAB | 14/40 | 15/40 |

  COM changed −0.149 … +0.012 dB (mean −0.022), and the CTLE choice changed on 9 channels: the Tx noise feeds
  the MMSE solve, so its phase error had been nudging the timing search. The cursor-timing bias is now gone in the
  mean, but the per-channel scatter (rms 3.3 samples) remains.

**Original finding:**

- **Where:** `src/pychopmarg/noise.py:247-249` (`NoiseCalc.Stn()`)
- **PyChOpMarg:** Builds the Tx-noise pulse with a causal moving sum (`lfilter(ones(nspui))`), but the signal
  pulse with a zero-phase sinc (`pulse_resp`). The Tx noise is therefore sampled about half a UI (~15.5 samples)
  off the signal's phase.
- **MATLAB / D3.3:** MATLAB uses its causal filter for both (`get_PulseR`, `get_PSDs`). D3.3 (178A-22) samples
  `hnoffe` at `ts0`, with the same pulse shape as the signal.
- **Measured effect:** Negligible (measured earlier).

### [x] A8. Cursor-timing sweep range excludes +16

- **Resolved (2026-10-15):** `mmse()` sweeps −16 … +16 inclusive. The "middle" search order still matches MATLAB's
  stable sort on |offset| (−16 before +16). Test: `test_middle_timing_search_follows_matlab` (now 33 offsets).
- **Measured effect** (reference set, vs. after D2): none. No channel's best offset is +16.
- **Observation:** three CR channels land on the −16 edge in PyChOpMarg. Two are at or near it in MATLAB too
  (`sma_1.0mm_12dB_1500mm`: −16 / −16; `sma_1.85mm_12dB_1500mm`: −16 / −15), so their optimum lies outside the
  ±½ UI search in both tools (see C1). On `li_dj_CR_Design_C_Rev1`, the tools disagree: −16 vs. MATLAB's −2.
- **`li_dj_CR_Design_C_Rev1_THRU` is a sensitivity outlier, not a modeling bug** (investigated 2026-10-15):
  - The difference isn't in the search path. At MATLAB's chosen EQ (CTLE −3/−2.5, offset −2), PyChOpMarg's FOM is
    10.7 dB against MATLAB's 12.24 dB.
  - The channel is extreme. The first pre-FFE pre-cursor is 70% of the peak, and Rx FFE pre-cursor tap 1 is
    clipped at −1.0 in both tools. FOM is a small difference of large terms: PyChOpMarg's swings from 2.6 dB at
    offset 0 to 10.2 dB at −4.
  - `mmse()` matches MATLAB's `MMSE_FOM`: the linear solve, DFE-limit re-solve, tap clipping relative to the
    cursor tap, renormalization and MSE formula.
  - The unequalized pulses agree to within 1% of peak (rms 0.2%; divide MATLAB's `vic_uneq_pulse_response` by
    `A_v` first). The residual sits on the rising edge (−3 UI: 7.16 vs. 6.64 mV; −2 UI: 33.5 vs. 32.3 mV), about
    the size of the difference between MATLAB's causal pulse forming (`filter(ones(M))`) and PyChOpMarg's
    zero-phase sinc. On this channel that plausibly explains the 1.5 dB FOM gap. Not tested.

**Original finding:**

- **Where:** `src/pychopmarg/optimize.py:306` (`range(-ts_sweep_ix, ts_sweep_ix)` → −16…+15)
- **MATLAB:** `sample_adjustment` = [-16 16], which includes +16.

### [x] A9. Signal path omits the Tx rise time filter, Ht

*(Found 2026-10-15, while working on A6; missed by the original audit.)*

- **Where:** `src/pychopmarg/com.py` (`COM.H()`), which computed `Htx * H21 * Hr * Hctf`.
  `Ht` (93A-46) was used only for the Tx noise (`NoiseCalc.Stn()`).
- **MATLAB / D3.3:** D3.3 (178A-14) has H⁽ᵏ⁾(f) = H_txffe · **Ht** · H21⁽ᵏ⁾ · Hr · Hctf. MATLAB, with the sheet's
  `FORCE_TR` = 1 (set since 2024), puts `exp(−2(πf·T_r/1.6832)²)` (plus a pure 3·T_r delay) into every channel's
  `sdd21`, in `s21_pkg()`. With T_r = 4 ps, Ht ≈ 0.73 (−2.7 dB) at Nyquist.
- **Resolved (2026-10-15):** `H()` now includes `self.Ht`, for victim and aggressors alike. Test:
  `test_signal_path_includes_tx_rise_time_filter`.
- **Measured effect** (reference set, vs. after A5): this was the common gain noted in the baseline.

  | PyChOpMarg − MATLAB | After A5 | After A9 |
  |---|---|---|
  | COM (dB) | +0.33 ± 0.24 | **−0.16 ± 0.09** (−0.31 … +0.18) |
  | FOM (dB) | +0.31 ± 0.24 | −0.16 ± 0.08 |
  | As, Ani (ratio) | 1.145, 1.102 | **0.994, 1.012** |
  | σ ISI, σ_rjit, σ_Tx, σ_XT (ratio) | 1.16, 1.15, 1.08, 1.12 | 1.05, 1.02, 0.99, 0.93 |
  | Cursor offset (samples) | +1.18, rms 4.29, exact 1/40 | +0.30, rms 3.41, exact 8/40 |
  | Same CTLE as MATLAB | 13/40 | 14/40 |

  COM fell 0.15–1.07 dB per channel (mean 0.48 dB), and the CTLE choice changed on 14 channels.

---

## B. Missing D3.3 features (no effect on today's comparison, because the sheet doesn't use them)

### [x] B1. Quantization noise

- D3.3 178A-28 (`Sqn`) and 178A-43/46 (`pΔ`, `pqn`). All four D3.3 group tables set `Nqb` = 6 and `Pqc` = 1e-7.
- MATLAB implements it when `N_qb` ≠ 0 (`get_PSDs`, `adjust_Rx_noise_for_quantization`).
- PyChOpMarg had no such term, so its COM was **optimistic** against D3.3.
- The comparison sheet doesn't set `N_qb`, so MATLAB skips it there too.
- **Resolved (2026-10-15), as D3.3:**
  - New `COMParams` fields `N_qb` (default 0, i.e. off) and `P_qc`. The sheet's `N_qb` and `P_qc` rows are read; as in
    MATLAB, `P_qc` defaults to 2 · DER_0 when the sheet sets only `N_qb`. The D3.3 presets
    (`config/ieee_8023dj_d3p3.py`) now set `N_qb` = 6 and `P_qc` = 1e-7.
  - **FOM (MMSE):** `NoiseCalc.Rn()` adds the white `Sqn` = Δ²/12/fb (178A-28) at the Rx FFE input. Δ comes from
    `NoiseCalc.quant_step()` (178A-29/30): the clipping amplitude `Vqc` satisfies Psn(−Vqc) = Pqc/2. Psn is the
    noiseless victim signal at the current sampling time (every UI), convolved with a Gaussian whose variance (178A-31)
    is that of every other noise term, with only the random part of the jitter (`Sjn_rj`), as MATLAB's `get_PSDs()`.
    It's recomputed at every sampling time and EQ setting tried, on a 2001-point voltage grid (Δ within about 0.2%).
  - **COM:** `calc_noise()` recomputes Δ as in 178A.1.9: crosstalk (at each aggressor's phase) and dual-Dirac jitter
    as delta-PMFs at the Rx FFE input, plus a Gaussian with σgn² from `Srn`, `Stn` and `Sjn_rj` (178A-44), on an
    8001-point grid. It then convolves the uniform PMF of width |w(n)|·Δ for every Rx FFE tap (178A-43/46) into the
    final PMF. `com_rslts` gains `quant_step`, `sigma_Q` and `pQ`; `mmse()`'s result gains `quant_step`.
  - PRZF mode (Clause 93A) has no quantization noise.
  - Tests: `test_quantization_noise_*`, `test_quantization_step_per_178a_29` (`tests/test_com.py`), and
    `test_quantization_noise_params` (`tests/test_excel.py`).
- **Measured effect:** none on the comparison, since the sheet doesn't set `N_qb`; with it off, the reference set
  matches the A8 results exactly. With `N_qb` = 6 and `P_qc` = 1e-7 (D3.3) on the reference set, COM − COM(off):

  | Group | COM (dB), mean [range] | FOM (dB), mean | σ_Q / As | CTLE changed |
  |---|---|---|---|---|
  | C2C | −1.12 [−2.05 … −0.66] | −1.13 | 0.08 | 9/10 |
  | C2M | −1.21 [−1.78 … −0.68] | −1.23 | 0.08 | 10/10 |
  | CR | −1.90 [−2.50 … −0.57] | −2.00 | 0.20 | 8/10 |
  | KR | −1.53 [−2.05 … −0.63] | −1.56 | 0.13 | 10/10 |
  | All | −1.44 [−2.50 … −0.57] | −1.48 | 0.12 | 37/40 |

  Δ from the FOM search and from COM agree to 0.1% on average. Run time rose 11% (247 s vs. 222 s).
- **Where MATLAB departs from D3.3 here** (not followed; matters only with `N_qb` ≠ 0):
  - Clipping probability: MATLAB solves Psn(−Vqc) = Pqc (`CDF_inv_ev(P_qc, …)`), not Pqc/2. That clips at
    twice the probability, giving a slightly smaller Δ.
  - COM's Δ: `adjust_Rx_noise_for_quantization()` takes Vqc from the CTLE-output signal convolved with the final
    (post-Rx-FFE) ISI, crosstalk and noise PDF, rather than from D3.3's pre-Rx-FFE noise, pn(y).
  - COM's `pqn`: MATLAB skips the Rx FFE's cursor tap (`if irxffe ~= cursor_tap`). 178A-46 runs over every
    coefficient. The cursor tap is the largest, so MATLAB's quantization noise in COM is smaller. MATLAB
    also drops zero-valued taps before indexing the cursor tap, which would shift that index if a pre-cursor tap were 0.
  - MATLAB's "Fast" `Clip Method` (Vqc = Σ|h|) isn't implemented; its default, "Slow", is the D3.3 method.

### [x] B2. Only the first package test case is evaluated

- PyChOpMarg builds only `zp_sel` = 0 (`_sPkgTx` / `_sPkgRx` are single-element lists).
- D3.3 groups define Test 1 / Test 2 cases, and MATLAB evaluates every case in a multi-case `z_p` list.
- The comparison sheet has one case.
- **Resolved (2026-10-15), as MATLAB:** MATLAB runs the whole COM calculation once per `z_p select` entry and
  reports the lowest COM (`min_COM`).
  - New `COMParams` field `zp_cases`: a list of per-case dicts of `zp_tx`, `zp_rx`, `zp_next` and `zp_fext`, which
    override those fields. `get_com_params()` fills it when the sheet's `z_p select` lists more than one case
    (previously it warned and used the first). `zp_tx`, etc. still hold the first case.
  - `COM.__call__()` evaluates each case in turn: it rebuilds the packages and the channels with them
    (`_set_pkg_case()`), then optimizes EQ and calculates COM (`_calc_com()`). It returns the lowest COM, and the
    instance keeps that case's state (`com_params`, channels, EQ, `com_rslts`, `fom_rslts`, noise calculator).
    `com_rslts["case_coms"]` holds every case's COM; `com_rslts["pkg_case"]`, the (0-based) index of the lowest.
  - The device-model path (`tx_device` / `rx_device`) is unchanged: one device pair, one case.
  - Tests: `test_com_is_lowest_over_package_test_cases` (`tests/test_com.py`) and
    `test_zp_select_lists_every_test_case` (`tests/test_excel.py`).
- **Measured effect:** none on the comparison (one case); the reference set matches the B1 results exactly.
  Each extra case costs one more full EQ optimization.
- **Not modeled:** MATLAB's other per-case parameters, `PKG_Tx_FFE_preset` and `AC_CM_RMS` (see B5).

### [ ] B3. Unsupported D3.3 features

- MLSD (178A.1.10).
- Floating Rx FFE taps (`Nwg`, `Nwf`, `Nmax`; sheet `N-g`, `N_f`, `N-max`). The sheet sets `N-g` = 0.
- Interference noise `Sin` (178A.1.7.5; receiver-test calibration only).

### [ ] B4. Sheet path can't add a host board or partial host channel

- Sheet `Include PCB`, `z_bp (…)`, `board_tl_*` and `board_Z_c` are unused.
- The D3.3 device models in `devmodel.py` can add a partial host channel, but loading settings from a sheet doesn't
  use them.
- The comparison sheet has `Include PCB` = 0.

### [ ] B5. MATLAB-only options PyChOpMarg ignores

- `Raised_Cosine` (now 0 in the comparison sheet). D3.3 calls for Butterworth only, so PyChOpMarg matches D3.3.
- `ts_anchor` modes other than peak.
- `ffe_main_cursor_min`, `ffe_tap_step_size`, `f_min`.
- Per-package-case `PKG_Tx_FFE_preset` and `AC_CM_RMS`.

### [ ] B6. CTLE form can't express `fz2` ≠ `fp3`

- PyChOpMarg's 93A-22 form ties `fz2` = `fp3` = `f_LF`. The D3.3 tables set `fz2` = `fp3` = fb/80, so this is
  equivalent today.

---

## C. Shared with MATLAB, departing from strict D3.3

### [-] C1. Bounded sampling-time search and search speed-ups

- D3.3 says the sampling time and equalizer settings are those that minimize MSE, with no limits.
- Both tools search only ±0.5 UI around the peak.
- PyChOpMarg's Local Search and `middle` / `track` timing search are speed-ups that can miss the optimum.
  Measured: 1.4–2.5× faster, FOM changes ≤ 0.003 dB.

---

## D. Bugs in reported intermediate values (FOM and COM unaffected)

### [x] D1. `fom_rslts['sigma_Tx']` is about 4.4× too large

- **Resolved (2026-10-15):** `mmse()` passes `hasCurs=True`. Its `sigma_Tx` now equals COM's, up to the Rx FFE's
  scaling. Test: `test_fom_and_com_tx_noise_agree`.
- **Note on units:** `fom_rslts`' sigmas aren't comparable with `com_rslts`', and they don't share units with each
  other. ISI and Tx are relative to a unit cursor (`mmse()`'s taps make h0·w = 1; on `example2`, its cursor tap
  is 191 vs. COM's 1). Jitter, crosstalk and thermal noise are pre-FFE volts. FOM itself uses the MSE, which is
  consistent. Only these reported values are affected.

**Original finding:**

- **Where:** `src/pychopmarg/optimize.py:404`
- `mmse()` computes `varTx` with `calc_Hffe(…, w_lim, Nw - dw - 1)` without `hasCurs=True`, so it inserts an extra
  synthesized cursor tap.
- On the CR channel: 0.125 reported vs. 0.0285 correct.
- The notebook plots `com_rslts['sigma_Tx']`, which is correct.
- **Fix:** Pass `hasCurs=True`.

### [x] D2. `theNoiseCalc.ts_ix` left at the last offset tried

- **Resolved (2026-10-15):** `mmse()` now leaves `theNoiseCalc.ts_ix` at the chosen cursor (`rslt["cursor_ix"]`).
  On `example2`, it had been left 30 samples away. Test: `test_noise_calc_left_at_chosen_sampling_time`.
- **Measured effect** (reference set, vs. after A7): negligible. COM changed −0.016 … +0.013 dB (mean −0.001), and
  no CTLE choice changed. COM − MATLAB's COM: −0.172 ± 0.095 dB.

**Original finding:**

- **Where:** `src/pychopmarg/optimize.py:319`
- The timing sweep leaves `ts_ix` at the last offset tried, not the best one, and `calc_noise()`'s `Stn` uses it.
- Measured effect: under 0.1%, even at offsets ±15 / −16.
- **Fix:** Restore `ts_ix` to the winning offset after the sweep.

---

## E. Hazard: sheet rows silently ignored because of name mismatches

### [ ] E1. Sheet names that don't map to PyChOpMarg parameters

- **Where:** `src/pychopmarg/excel.py`
- These sheet rows are ignored, and the `IEEE_8023dj` defaults are used instead:

  | Sheet row | PyChOpMarg field |
  |---|---|
  | `R_LM` | `RLM` |
  | `sigma_RJ` | `sigma_Rj` |
  | `package_tl_tau` | `tau` |
  | `package_tl_gamma0_a1_a2` | `gamma0`, `a1`, `a2` |
  | `package_Z_c` | `z_c` |

- Their values currently equal the defaults, which hides the problem. Editing any of them in the sheet would change
  MATLAB's results but not PyChOpMarg's.
- Related: `T_r` sits in the sheet's I/O column and isn't read (the default 0.004 equals the sheet).
- The stray `com_params.ts_track` row is also ignored; set `ts_track` in the notebook instead.
- **Suggested fix:** Map these names, and warn on any unrecognized sheet row.

---

## F. Small differences (probably not worth changing)

- **F1. Small-sample PMF filtering:** `filt_pr_samps` drops samples below 0.1%·As. MATLAB's
  `get_pdf_from_sampled_signal` drops |x| ≤ the bin size (min(As/1000, 10 µV)). Approximately equal.
- **F2. `varN` in `calc_noise()` skips the DC bin.** Trivial.
- **F3. MMSE `h` length:** PyChOpMarg uses the full pulse length. MATLAB pads or truncates to
  `num_ui_RXFF_noise` (2048 UI). Trivial.
- **F4. PyChOpMarg-only option:** `ts_track`.

---

## Verified as matching (no deviation)

- MMSE solve (178A-33–37), including the `b` / `w` limits and renormalization.
- `clip_taps` (178A-36 scaling by the cursor tap).
- Noise PSDs: `Srn` folding, `Sxn` phase, `Sjn`, and `Stn` (no σX² factor).
- NEXT uses no Tx FFE.
- Signal amplitude `As`; Gaussian σG composition (equivalent form).
- Final COM error-ratio point; Tx FFE grid.
- The cursor index used in `calc_noise()` on the post-FFE pulse.
