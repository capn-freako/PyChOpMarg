# PyChOpMarg deviations from P802.3dj/D3.3 and MATLAB COM 4p17p0

This document tracks where PyChOpMarg computes final or intermediate results differently than:

- **D3.3**: IEEE P802.3dj/D3.3, Annex 178A (the COM reference), and
- **MATLAB**: the IEEE 802.3 MATLAB COM reference code, version 4p17p0.

It came from an audit of the whole calculation path on 2026-10-14 (after v3.2.0, PR #27).
Line numbers refer to `main` at `fd4abd2`.

**Status key:** `[ ]` open · `[~]` in progress · `[x]` resolved · `[-]` won't fix (keep the deviation, with a reason)

When you resolve an item, check it off, and record the commit and any measured effect on the
PyChOpMarg-vs-MATLAB comparison (`notebook/PyChOpMarg_vs_MATLAB.ipynb`).

---

## A. Differences from both MATLAB and D3.3 that affect today's comparison results

### [ ] A1. MMSE solve keeps only `dw` pre-cursor UIs, and zeroes small samples

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

### [ ] A2. Final COM ISI keeps only 5 pre-cursor UIs

- **Where:** `src/pychopmarg/com.py:1105` (`calc_noise()`: `n_pre = min(5, curs_uis)`; the comment cites MATLAB v2.60)
- **MATLAB / D3.3:** 4p17p0 (`get_PSDs` COMPUTE_COM pass, `get_pdf`) and 178A-42 use every pre-cursor sample.

### [ ] A3. Jitter slopes taken only from the cursor onward, and magnitude-filtered

- **Where:** `src/pychopmarg/utility/probability.py:133-156` (`calc_hJ()`: `range(cursor_ix, ...)`, plus a
  `rel_thresh` = 0.1%·As filter). Used at `src/pychopmarg/com.py:832` and `:1097`.
- **MATLAB / D3.3:** 178A.1.9.3 / 178A-23 and MATLAB (`OptFom_Calc_Noise`, `h_J`) take slopes at *every* UI,
  pre-cursors included, with no magnitude filter.
- **Affects:** `pJ` (DD jitter, `hDD`) and `varJ` (Rj) in `calc_noise()`, and the jitter PSD in the FOM search.

### [ ] A4. Sheet `z_p (TX/RX/NEXT/FEXT)` rows are ignored

- **Where:** `src/pychopmarg/excel.py:31-50, 186`
- **PyChOpMarg:** The sheet's `z_p (TX)`, `z_p (RX)`, `z_p (NEXT)`, `z_p (FEXT)` and `z_p select` rows are parsed but
  never used. Package lengths come from the `IEEE_8023dj` default `z_p` = [33, 1.8] mm, the same at both ends.
- **MATLAB:** Uses the sheet's values. The comparison sheet has Tx 34 mm and Rx 32 mm (each with 1.8 mm).
- **Why it matters:** The total length is the same, so the effect is probably small, but it isn't the same model.

### [ ] A5. Aggressor sampling phase chosen after the Rx FFE

- **Where:** `src/pychopmarg/com.py:1124` (`calc_noise()`)
- **PyChOpMarg:** Picks each aggressor's phase by maximum energy of its pulse *after* the Rx FFE.
- **MATLAB / D3.3:** Pick it from the pulse *before* the Rx FFE (MATLAB's `iphase` from `get_PSDs`; D3.3's `ts(k)`
  per 178A-20).

### [ ] A6. Channel interpolation and extrapolation

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

### [ ] A7. Tx-noise pulse phase

- **Where:** `src/pychopmarg/noise.py:247-249` (`NoiseCalc.Stn()`)
- **PyChOpMarg:** Builds the Tx-noise pulse with a causal moving sum (`lfilter(ones(nspui))`), but the signal
  pulse with a zero-phase sinc (`pulse_resp`). The Tx noise is therefore sampled about half a UI (~15.5 samples)
  off the signal's phase.
- **MATLAB / D3.3:** MATLAB uses its causal filter for both (`get_PulseR`, `get_PSDs`). D3.3 (178A-22) samples
  `hnoffe` at `ts0`, with the same pulse shape as the signal.
- **Measured effect:** Negligible (measured earlier).

### [ ] A8. Cursor-timing sweep range excludes +16

- **Where:** `src/pychopmarg/optimize.py:306` (`range(-ts_sweep_ix, ts_sweep_ix)` → −16…+15)
- **MATLAB:** `sample_adjustment` = [-16 16], which includes +16.

---

## B. Missing D3.3 features (no effect on today's comparison, because the sheet doesn't use them)

### [ ] B1. Quantization noise

- D3.3 178A-28 (`Sqn`) and 178A-43/46 (`pΔ`, `pqn`). All four D3.3 group tables set `Nqb` = 6 and `Pqc` = 1e-7.
- MATLAB implements it when `N_qb` ≠ 0 (`get_PSDs`, `adjust_Rx_noise_for_quantization`).
- PyChOpMarg has no such term, so its COM is **optimistic** against D3.3.
- The comparison sheet doesn't set `N_qb`, so MATLAB skips it there too.

### [ ] B2. Only the first package test case is evaluated

- PyChOpMarg builds only `zp_sel` = 0 (`_sPkgTx` / `_sPkgRx` are single-element lists).
- D3.3 groups define Test 1 / Test 2 cases, and MATLAB evaluates every case in a multi-case `z_p` list.
- The comparison sheet has one case.

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

### [ ] D1. `fom_rslts['sigma_Tx']` is about 4.4× too large

- **Where:** `src/pychopmarg/optimize.py:404`
- `mmse()` computes `varTx` with `calc_Hffe(…, w_lim, Nw - dw - 1)` without `hasCurs=True`, so it inserts an extra
  synthesized cursor tap.
- On the CR channel: 0.125 reported vs. 0.0285 correct.
- The notebook plots `com_rslts['sigma_Tx']`, which is correct.
- **Fix:** Pass `hasCurs=True`.

### [ ] D2. `theNoiseCalc.ts_ix` left at the last offset tried

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
