"""
Noise power spectral density (PSD) calculator for COM.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 2, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

# pylint: disable=redefined-builtin
from collections.abc        import Callable
from typing                 import Any, Optional

from numpy                  import (abs as np_abs, arange, argmax, array, cumsum, diff, exp, full, ndarray, ones, pad,
                                    sinc, sqrt, sum)
from numpy.fft              import irfft, rfft
from scipy.signal           import fftconvolve

from pychopmarg.common      import Rvec, Cvec, Rmat
from pychopmarg.utility.probability import delta_pmf


class NoiseCalc:  # pylint: disable=too-many-instance-attributes
    """
    Noise calculator for COM

    Notes:
        1. The noise terms independent of the sampling time (``ts_ix``) are calculated once and cached,
        since the MMSE timing search (``optimize.mmse()``) calls ``Rn()`` at each of many sampling times.
        Assigning any other independent variable clears that cache; cached arrays are read-only.
    """

    # Independent variables whose (re)assignment invalidates the cache. (All but `ts_ix`.)
    _CACHE_INPUTS = frozenset({
        "L", "Tb", "t", "vic_pulse_resp", "agg_pulse_resps", "f", "Ht", "H21", "Hr", "Hctf",
        "eta0", "Av", "snr_tx", "Add", "sigma_Rj", "Nqb", "Pqc"})

    # Independent variables, set during instance initialization.
    L               = 4                          # Number of modulation levels.
    Tb              = 9.412e-12                  # UI (s)
    ts_ix           = 0                          # Main pulse sampling index
    t               = array([0], dtype=float)    # System time vector (s)
    vic_pulse_resp  = array([0], dtype=float)    # Victim pulse response (V)
    agg_pulse_resps: list[Rvec]                  # Aggressor pulse responses (V)
    f               = array([0], dtype=float)    # System frequency vector (Hz)
    Ht              = array([0], dtype=complex)  # Transfer function of Tx output driver risetime
    H21             = array([0], dtype=complex)  # Transfer function of terminated interconnect
    Hr              = array([0], dtype=complex)  # Transfer function of Rx AFE
    Hctf            = array([0], dtype=complex)  # Transfer function of Rx CTLE
    eta0            = float(0)                   # Noise density at Rx AFE input (V^2/GHz)
    Av              = 0.6                        # Victim drive level (V)
    snr_tx          = float(25)                  # Tx signal-to-noise ratio (dB)
    Add             = float(0)                   # Dual-Dirac peak amplitude (V)
    sigma_Rj        = float(0)                   # Dual-Dirac random stdev (V)
    Nqb             = 0                          # Noise-equivalent quantization bits (0 = no quantization noise)
    Pqc             = 1e-7                       # Quantization clipping probability

    # Invariant dependent variables, set during instance initialization.
    fN       = 53.125e9                   # Nyquist frequency (Hz)
    nspui    = 32                         # Number of samples per UI
    varX     = float(0)                   # Signal power (V^2)

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self, L: int, Tb: float, ts_ix: int, t: Rvec,
        vic_pulse_resp: Rvec, agg_pulse_resps: list[Rvec],
        f: Rvec, Ht: Cvec, H21: Cvec, Hr: Cvec, Hctf: Cvec,
        eta0: float, Av: float, snr_tx: float, Add: float, sigma_Rj: float,
        eps: float = 0.001, Nqb: int = 0, Pqc: float = 1e-7
    ) -> None:
        """
        ``NoiseCalc`` class initializer.

        Args:
            L: Number of modulation levels.
            Tb: Unit interval (s).
            ts_ix: Main pulse sampling index.
            t: System time vector (for indexing pulse responses) (s).
            vic_pulse_resp: Victim pulse response (V).
            agg_pulse_resps: list of aggressor pulse responses (V).
            f: System frequency vector (for indexing transfer functions) (Hz).
            Ht: Risetime filter transfer function.
            H21: Terminated interconnect transfer function.
            Hr: Rx AFE transfer function.
            Hctf: Rx CTLE transfer function.
            eta0: Noise spectral density at Rx AFE input (V^2/GHz).
            Av: Victim output drive level (V).
            snr_tx: Signal-to-noise ratio of Tx output (dB).
            Add: Dual-Dirac peak deterministic jitter (UI).
            sigma_Rj: Dual-Dirac random jitter standard deviation (UI).

        Keyword Args:
            eps: Margin of error when checking floats for integral multiples.
                Default: 0.001
            Nqb: Noise-equivalent quantization bits (178A.1.7.6).
                Default: 0 (No quantization noise.)
            Pqc: Quantization clipping probability (178A.1.7.6).
                Default: 1e-7

        Returns:
            Initialized ``NoiseCalc`` instance.

        Raises:
            IndexError: Unless `t`, and `vic_pulse_resp` have the same length.
            IndexError: Unless `f`, `Ht`, `H21`, `Hr`, and `Hctf` all have the same length.
            ValueError: Unless `t[0]` == 0.
            ValueError: Unless `Tb` is an integral multiple of `t[1]`.
            ValueError: Unless `t` is uniformly sampled.
            ValueError: Unless `f[0]` == 0.
            ValueError: Unless `f[1]` == 1 / (t[1] * len(t)).
            ValueError: Unless `f[-1]` == 0.5 / t[1].
            ValueError: Unless `f` is uniformly sampled.

        Notes:
            1. The transfer functions are assumed to contain only positive frequency values.

        ToDo:
            1. Consider defining a data class to hold the initialization data.
        """

        assert len(t) == len(vic_pulse_resp), IndexError(
            f"Lengths of `t` ({len(t)}) and `vic_pulse_resp` ({len(vic_pulse_resp)}) must be the same!")
        assert len(f) == len(Ht) == len(H21) == len(Hr) == len(Hctf), IndexError(
            "\n\t".join(
                ["The lengths of the following input vectors must be equal:",
                 f"`f`    ({len(f):7d})",
                 f"`Ht`   ({len(Ht):7d})",
                 f"`H21`  ({len(H21):7d})",
                 f"`Hr`   ({len(Hr):7d})",
                 f"`Hctf` ({len(Hctf):7d})"]))
        assert t[0] == 0, ValueError(
            f"The first element of `t` ({t[0]}) must be zero!")
        assert abs(Tb / t[1] - Tb // t[1]) < eps, ValueError(
            f"`Tb` ({Tb}) must be an integral multiple of `t[1]` ({t[1]})!")
        dts = diff(t)
        assert all((dts - dts[0]) < 1e-15), ValueError(
            "The time vector, `t`, must be uniformly sampled!")
        assert f[0] == 0, ValueError(
            f"The first element of `f` ({f[0]}) must be zero!")
        f0 = 1 / (t[1] * len(t))
        assert f[1] == f0, ValueError(
            f"The second element of `f` ({f[1]}) must be the fundamental frequency implied by the time vector ({f0})!")
        fs = 1 / t[1]
        assert (f[-1] - fs / 2) < eps, ValueError(
            f"The last element of `f` ({f[-1]}) must be half the sampling frequency ({fs})!")
        dfs = diff(f)
        assert all((dfs - dfs[0]) < eps), ValueError(
            "The frequency vector, `f`, must be uniformly sampled!")

        super().__init__()

        # Initialize independent variable values.
        self.L               = L
        self.Tb              = Tb
        self.ts_ix           = ts_ix
        self.t               = t
        self.vic_pulse_resp  = vic_pulse_resp
        self.agg_pulse_resps = agg_pulse_resps
        self.f               = f
        self.Ht              = Ht
        self.H21             = H21
        self.Hr              = Hr
        self.Hctf            = Hctf
        self.eta0            = eta0
        self.Av              = Av
        self.snr_tx          = snr_tx
        self.Add             = Add
        self.sigma_Rj        = sigma_Rj
        self.Nqb             = Nqb
        self.Pqc             = Pqc
        self.quant_step_size = 0.0  # Quantization step size, Delta (V), as of the last call to `Rn()`.

        # Calculate invariant dependent variable values.
        self.fN      = 0.5 / Tb                         # Nyquist frequency
        self.nspui   = int(Tb // t[1])                  # Samples per unit interval
        self.varX    = (L**2 - 1) / (3 * (L - 1)**2)    # Tx output signal power

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._CACHE_INPUTS:
            self.__dict__.pop("_cache", None)
        super().__setattr__(name, value)

    def _cached(self, key: str, calc: Callable[[], Any]) -> Any:
        "``calc()``, calculated once per ``key`` (until an input changes); arrays made read-only."
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            val = calc()
            for arr in val if isinstance(val, tuple) else (val,):
                if isinstance(arr, ndarray):
                    arr.flags.writeable = False
            cache[key] = val
        return cache[key]

    def get_aliased_freq(self, f: float) -> float:
        """
        Return alias of f (Hz).
        """

        fN = self.fN
        if int(f / fN) % 2 == 0:
            return f % fN

        return fN - (f % fN)

    def baud_rate_sample(self, x: Rvec) -> Rvec:
        """
        Resample the input at fBaud., respecting the current sampling phase.

        Args:
            x: Signal to be resampled.

        Returns:
            Resampled vector.
        """
        nspui = self.nspui
        return x[self.ts_ix % nspui::nspui]

    def variance(self, S: Rvec) -> float:
        """
        Variance of the noise whose folded PSD is ``S`` (as returned by ``Srn``, ``Sxn()``, ``Stn()``, or ``Sjn``).

        Notes:
            1. ``S`` holds the [0, PI] half of a two-sided PSD, so its full-band integral is
            ``irfft(S)[0] / Tb`` (i.e. - ``Rn()[0]`` for that term alone), not ``sum(S) * df``,
            which counts only positive frequencies.
        """
        return float(irfft(S)[0] / self.Tb)

    @property
    def Srn(self) -> Rvec:
        """
        Folded (two-sided) Rx noise PSD at Rx FFE input,
        uniformly sampled over [0, PI] (rads./s norm.).

        Notes:
            1. Re: the scaling term: ``2 * self.f[-1]``, when combined w/
            the implicit ``1/N`` of the ``irfft()`` function, this gives ``df``.
            2. As in [1] and the MATLAB COM code, the two-sided density is ``eta0 / 2``
            (``eta0`` being one-sided), so that ``variance(Srn)`` is the Rx noise variance.
        """
        def calc() -> Rvec:
            rslt: Cvec  = self.eta0 / 2 * 1e-9 * abs(self.Hr * self.Hctf) ** 2
            # Rx noise is stationary: fold by sampling its autocorrelation at lags of whole UIs,
            # independent of the cursor sampling phase (unlike `Stn()`/`Sjn`, which are symbol-synchronous).
            return abs(rfft(irfft(rslt)[::self.nspui])) * 2 * self.f[-1] * self.Tb
        return self._cached("Srn", calc)

    def Sxn(self, agg_pulse_resp: Rvec) -> Rvec:
        """
        Crosstalk PSD at Rx FFE input.

        Args:
            agg_pulse_resp: Aggressor pulse response (V).

        Returns:
            Folded (two-sided) crosstalk PSD at Rx FFE input, uniformly sampled over [0, PI] (rads./s norm.).
        """

        return self.varX * abs(rfft(self.sampled_agg_pr(agg_pulse_resp)))**2 * self.Tb  # i.e. - / fB

    def sampled_agg_pr(self, agg_pulse_resp: Rvec) -> Rvec:
        "The given aggressor pulse response, sampled once per UI at its phase (``agg_phase()``)."
        return self._sampled_agg_prs(agg_pulse_resp)[self.agg_phase(agg_pulse_resp)]

    def _sampled_agg_prs(self, agg_pulse_resp: Rvec) -> Rmat:
        "The given aggressor pulse response, sampled once per UI at each phase (one row per phase)."
        nspui = self.nspui
        # Truncate at # of whole UIs in `t`, to avoid +/-1 length variation, due to sampling phase.
        nUI = int(len(self.t) / nspui)
        _agg_pulse_resp = agg_pulse_resp[:nUI * nspui]
        return array([_agg_pulse_resp[m::nspui] for m in range(nspui)])

    def agg_phase(self, agg_pulse_resp: Rvec) -> int:
        """
        Sampling phase of the given aggressor pulse response at Rx FFE input, as per (178A-20):
        the one maximizing the sum of its squared samples (as used by ``Sxn()``).

        Args:
            agg_pulse_resp: Aggressor pulse response (V).

        Returns:
            Sampling phase (vector samples, in [0, nspui)).
        """
        return int(argmax([(pr_samps**2).sum() for pr_samps in self._sampled_agg_prs(agg_pulse_resp)]))

    def Stn(self, Hrx: Optional[Cvec] = None) -> Rvec:
        """
        Folded (two-sided) Tx noise PSD at Rx FFE input,
        uniformly sampled over [0, PI] (rads./s norm.).

        Keyword Args:
            Hrx: Complex voltage transfer function of Rx FFE.
                Default: None (Use flat unity response.)
        """

        f     = self.f

        def calc(Hrx: Cvec) -> tuple[Cvec, Rvec, Rvec]:
            Htn  = self.Ht * self.H21 * self.Hr * self.Hctf * Hrx * self.Av  # "* Av": consistency w/ MATLAB code
            htn  = irfft(Htn)                               # impulse response of complete signal path, but Tx FFE
            # Pulse response (178A-22), formed as the signal's is (`COM.pulse_resp()`): w/ a zero-phase, 1 UI wide
            # rectangle (`nspui * sinc(f * Tb)`), so that sampling at `ts_ix` takes it at the signal's phase.
            return Htn, htn, irfft(self.nspui * sinc(f * self.Tb) * Htn)

        fom = Hrx is None
        if fom:  # (Independent of the sampling time; so, calculated once.)
            Htn, htn, htn_ave_xM = self._cached("Stn", lambda: calc(ones(len(f))))
        else:
            assert Hrx is not None  # (for type checkers)
            Htn, htn, htn_ave_xM = calc(Hrx)
        _htn = self.baud_rate_sample(htn_ave_xM)        # decimated by `nspui`
        _Htn = rfft(_htn)

        # Stash debugging info if FOM'ing.
        if fom:
            self.Stn_debug = {  # pylint: disable=attribute-defined-outside-init
                'Htn':  Htn,
                'htn':  htn,
                'htn_pulse': htn_ave_xM,
                '_Htn': _Htn,
                '_htn': _htn,
            }

        # No `varX` (sigma_X^2) here, unlike `Sxn()`: (178A-21), and MATLAB COM's default (`S_tn_w_AM` = 0).
        return self.Tb * 10**(-self.snr_tx / 10) * abs(_Htn)**2                       # (178A-21)

    def _jitter_psd(self, var_jit: float) -> Rvec:
        """
        Folded (two-sided) PSD of the noise due to jitter of variance ``var_jit`` (UI^2) at Rx FFE input,
        uniformly sampled over [0, PI] (rads./s norm.).
        """

        t              = self.t
        ts_ix          = self.ts_ix
        Tb             = self.Tb
        nspui          = self.nspui
        vic_pulse_resp = self.vic_pulse_resp

        # Central difference around each sample, `ts_ix % nspui + n * nspui`, as per (178A-23).
        # (Zero padding keeps the first sample valid at sampling phase zero, and
        # a whole number of UIs avoids +/-1 length variation, due to sampling phase.)
        nUI = len(vic_pulse_resp) // nspui
        ixs = ts_ix % nspui + nspui * arange(nUI)
        _vic_pulse_resp = pad(vic_pulse_resp, 1)
        hJ = (_vic_pulse_resp[ixs + 2] - _vic_pulse_resp[ixs]) / (2 * t[1])

        return self.varX * var_jit * abs(rfft(hJ) * Tb)**2 * Tb  # i.e. - / fB

    @property
    def Sjn(self) -> Rvec:
        """
        Folded (two-sided) noise PSD due to jitter at Rx FFE input,
        uniformly sampled over [0, PI] (rads./s norm.).
        """
        return self._jitter_psd(self.Add**2 + self.sigma_Rj**2)

    @property
    def Sjn_rj(self) -> Rvec:
        """
        Folded (two-sided) noise PSD due to random jitter alone at Rx FFE input (178A-45),
        uniformly sampled over [0, PI] (rads./s norm.).
        """
        return self._jitter_psd(self.sigma_Rj**2)

    def quant_step(self, var_noise: float, noise_samps: Optional[list[Rvec]] = None, npts: int = 2001) -> float:
        """
        Quantization step size, Delta, as per (178A-29): ``2 * Vqc / (2**Nqb - 1)``, where the clipping amplitude,
        ``Vqc``, satisfies ``Psn(-Vqc) = Pqc / 2`` for the noisy signal at the quantizer (178A-30).

        Args:
            var_noise: Variance of the Gaussian noise at the quantizer (V^2).

        Keyword Args:
            noise_samps: Sample sets (V) whose delta-PMFs (93A-40) are also convolved into the noise
                (e.g. crosstalk and dual-Dirac jitter, as for the final COM calculation, in 178A.1.9).
                Default: None
            npts: Number of points in the voltage grid (odd). The default, used during EQ optimization,
                gives Delta within about 0.2%, at a quarter of the cost of 8001.
                Default: 2001

        Returns:
            Quantization step size (V); 0, if ``Nqb`` is 0.

        Notes:
            1. The noiseless signal is the victim pulse response sampled once per UI at the current
            sampling phase (``ts_ix``), every UI included (178A.1.7.6).
        """

        if not self.Nqb:
            return 0.0
        noise_samps = noise_samps or []
        sig_samps = self.baud_rate_sample(self.vic_pulse_resp)

        # Voltage grid spanning the full support of the noisy signal.
        sigma = sqrt(var_noise)
        support = np_abs(sig_samps).sum() + sum([np_abs(h).sum() for h in noise_samps]) + 10 * sigma
        npts |= 1  # MUST BE ODD!
        ystep = 2 * support / (npts - 1)
        y = (arange(npts) - npts // 2) * ystep

        def pmf(samps: Rvec) -> Rvec:
            return delta_pmf(samps[np_abs(samps) > ystep / 2], L=self.L, y=y)[1]

        psn = pmf(sig_samps)                                                                    # (93A-40)
        for h in noise_samps:
            psn = fftconvolve(psn, pmf(h), mode="same")
        if sigma > 0:
            psn = fftconvolve(psn, exp(-y**2 / (2 * var_noise)), mode="same")                  # (93A-42)
        psn = psn.clip(min=0)
        Psn = cumsum(psn) / psn.sum()                                                           # (178A-30)
        Vqc = -y[argmax(Psn >= self.Pqc / 2)]
        return 2 * Vqc / (2**self.Nqb - 1)                                                      # (178A-29)

    def Sqn(self, delta: float) -> Rvec:
        """
        Folded (two-sided) quantization noise PSD at Rx FFE input (178A-28),
        uniformly sampled over [0, PI] (rads./s norm.).

        Args:
            delta: Quantization step size (V).
        """
        return full(len(self.f) // self.nspui + 1, self.Tb * delta**2 / 12)  # i.e. - / fB

    def Sn(self) -> Rvec:
        """
        Total noise PSD at Rx FFE input (178A-17), at the current sampling time,
        folded (two-sided) and uniformly sampled over [0, PI] (rads./s norm.).

        Notes:
            1. With ``Nqb`` > 0, the quantization noise (178A-28) is included, its step size found from a
            Gaussian approximation of the noise at the quantizer, whose variance (178A-31) is that of
            every other noise term, but with only the random part of the jitter (as MATLAB COM's ``get_PSDs()``).
            The step size is left in ``quant_step_size``.
        """
        Srn = self.Srn
        # (Each aggressor is sampled at its own phase (`agg_phase()`), independent of `ts_ix`.)
        Sxn = self._cached("Sxn", lambda: sum(array(list(map(self.Sxn, self.agg_pulse_resps))), axis=0))
        Stn = self.Stn()
        Sjn = self.Sjn
        if self.agg_pulse_resps:
            min_len = min(len(Srn), len(Sxn), len(Stn), len(Sjn))
            Sn = Srn[:min_len] + Sxn[:min_len] + Stn[:min_len] + Sjn[:min_len]
        else:
            min_len = min(len(Srn), len(Stn), len(Sjn))
            Sn = Srn[:min_len] + Stn[:min_len] + Sjn[:min_len]
        if self.Nqb:
            Sga = Sn - Sjn[:min_len] + self.Sjn_rj[:min_len]
            self.quant_step_size = self.quant_step(self.variance(Sga))                         # (178A-31)
            Sn = Sn + self.Sqn(self.quant_step_size)[:min_len]                                  # (178A-17)
        else:
            self.quant_step_size = 0.0
        return Sn

    def Rn(self) -> Rvec:
        """Noise autocorrelation vector at Rx FFE input (178A-18), at the current sampling time. (See ``Sn()``.)"""
        # i.e. - `* fB`, which when combined w/ the implicit `1/N` of `irfft()` yields `* df`.
        return irfft(self.Sn()) / self.Tb
