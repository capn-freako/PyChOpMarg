"""
Noise power spectral density (PSD) calculators.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 2, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

import numpy as np


def get_aliased_freq(fs, f):
    """
    Given sampling frequency, fs, return alias of f.
    """
    fn = fs / 2  # Nyquist frequency
    if np.int(f / fn) % 2 == 0:
        return f % fn
    else:
        return fn - (f % fn)


def Srn(eta0: float, hr: Cvec, hctf: Cvec, f: Rvec, fs: Optional[float] = None) -> Rvec:
    """
    Rx noise power spectral density (PSD),
    folded (i.e. - aliased) into the discrete frequency domain.

    Args:
        eta0: Noise power spectral density at Rx AFE input.
        hr: Complex frequency response of Rx AFE.
        hctf: Complex frequency response of Rx CTLE.
        f: Frequencies at which `hr` and `hctf` were sampled.

    Keyword Args:
        fs: The sampling rate (same units as `f`).
            Default: None, which implies: f[-1] = fNyquist.

    Returns:
        Srn: One-sided folded noise PSD at Rx sampler input.

    Raises:
        IndexError: Unless `f`, `hr`, and `hctf` all have the same length.
    """
    assert len(hr) == len(hctf) == len(f), IndexError(
        f"The lengths of `hr` {len(hr)}, `hctf` {len(hctf)}, and `f` {len(f)} must be the same.")
    if fs is not None:
        fN = fs / 2
    else:
        fN = f[-1]
    rslt = eta0 * abs(hr * hctf) ** 2  # "/ 2" in [1] omitted, since we're only considering: m >= 0.
    if f[-1] <= fN:
        return rslt
    else:
        for _f, _v in list(zip(f, rslt))[np.where(f > fN)]:
            targ_ix = argmin(abs(f - get_aliased_freq(fs, _f)))
            rslt[targ_ix] += _v
    return rslt[:np.where(f > fN)[0][0]]


def Sxn(n_levels: int, samps_per_ui: int, fB: float, agg_pulse_resp: Rvec) -> float:
    """
    Crosstalk PSD at Rx FFE input.

    Args:
        n_levels: Number of different modulation voltages. ("L" in spec.)
        samps_per_ui: Number of samples per unit interval.
        fB: Baud., or symbol, rate.
        agg_pulse_resp: Aggressor pulse response.

    Returns:
        Sxn: One-sided crosstalk PSD at Rx FFE input.
    """
    sampled_agg_prs = [agg_pulse_resp[m::samps_per_ui] for m in range(samps_per_ui)]
    best_m = argmax(list(map(lambda pr_samps: sum(np.array(pr_samps)**2), sampled_agg_prs)))
    return ((n_levels**2 - 1) / (3 * (n_levels - 1)**2)) * abs(rfft(sampled_agg_prs[best_m]))**2 / fB


def Stn(Av, Tb, f, snr_tx, Ht, H21, Hr, Hctf):
    """
    """
    Htn = Ht * H21 * Hr * Hctf
    _htn = irfft(Av * Tb * sinc(f * Tb) * Htn)
    htn = _htn[start_ix::nspui]
