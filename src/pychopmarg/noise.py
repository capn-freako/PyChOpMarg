"""
Noise power spectral density (PSD) calculator for COM.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 2, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

# pylint: disable=redefined-builtin
from typing                 import Optional

from numpy                  import argmax, array, concatenate, diff, mean, ones, reshape, sum
from numpy.fft              import irfft, rfft
from scipy.signal           import lfilter

from pychopmarg.common      import Rvec, Cvec, Rmat


class NoiseCalc():  # pylint: disable=too-many-instance-attributes
    "Noise calculator for COM"

    # Independent variables, set during instance initialization.
    L               = int(4)                     # Number of modulation levels.
    Tb              = float(9.412e-12)           # UI (s)
    ts_ix           = int(0)                     # Main pulse sampling index
    t               = array([0], dtype=float)    # System time vector (s)
    vic_pulse_resp  = array([0], dtype=float)    # Victim pulse response (V)
    agg_pulse_resps: list[Rvec] = []             # Aggressor pulse responses (V)
    f               = array([0], dtype=float)    # System frequency vector (Hz)
    Ht              = array([0], dtype=complex)  # Transfer function of Tx output driver risetime
    H21             = array([0], dtype=complex)  # Transfer function of terminated interconnect
    Hr              = array([0], dtype=complex)  # Transfer function of Rx AFE
    Hctf            = array([0], dtype=complex)  # Transfer function of Rx CTLE
    eta0            = float(0)                   # Noise density at Rx AFE input (V^2/GHz)
    Av              = float(0.6)                 # Victim drive level (V)
    snr_tx          = float(25)                  # Tx signal-to-noise ratio (dB)
    Add             = float(0)                   # Dual-Dirac peak amplitude (V)
    sigma_Rj        = float(0)                   # Dual-Dirac random stdev (V)

    # Invariant dependent variables, set during instance initialization.
    fN       = float(53.125e9)            # Nyquist frequency (Hz)
    nspui    = int(32)                    # Number of samples per UI
    varX     = float(0)                   # Signal power (V^2)

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self, L: int, Tb: float, ts_ix: int, t: Rvec,
        vic_pulse_resp: Rvec, agg_pulse_resps: list[Rvec],
        f: Rvec, Ht: Cvec, H21: Cvec, Hr: Cvec, Hctf: Cvec,
        eta0: float, Av: float, snr_tx: float, Add: float, sigma_Rj: float,
        eps: float = 0.001
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
            1. No assumption is made, re: any linkage between `t` and `f`.
            2. The transfer functions are assumed to contain only positive frequency values.
            3. `f` may begin at zero, but is not required to.

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
        assert f[-1] == fs / 2, ValueError(
            f"The last element of `f` ({f[-1]}) must be half the sampling frequency ({fs})!")
        dfs = diff(f)
        assert all(dfs == dfs[0]), ValueError(
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

        # Calculate invariant dependent variable values.
        self.fN      = 0.5 / Tb                         # Nyquist frequency
        self.nspui   = int(Tb // t[1])                  # Samples per unit interval
        self.varX    = (L**2 - 1) / (3 * (L - 1)**2)    # Tx output signal power

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

    @property
    def Srn(self) -> Rvec:
        """
        One-sided folded noise PSD at Rx sampler input,
        uniformly sampled over [0, PI] (rads./s norm.).

        Notes:
            1. Re: the scaling term: ``2 * self.f[-1]``, when combined w/
            the implicit ``1/N`` of the ``irfft()`` function, this gives ``df``.
        """
        # "/ 2" in [1] omitted, since we're only considering: m >= 0.
        rslt: Cvec  = self.eta0 * 1e-9 * abs(self.Hr * self.Hctf) ** 2
        _rslt = abs(rfft(self.baud_rate_sample(irfft(rslt)))) * 2 * self.f[-1] * self.Tb
        return _rslt

    def Sxn(self, agg_pulse_resp: Rvec) -> Rvec:
        """
        Crosstalk PSD at Rx FFE input.

        Args:
            agg_pulse_resp: Aggressor pulse response (V).

        Returns:
            One-sided crosstalk PSD at Rx FFE input, uniformly sampled over [0, PI] (rads./s norm.).
        """

        t     = self.t
        nspui = self.nspui

        # Truncate at # of whole UIs in `t`, to avoid +/-1 length variation, due to sampling phase.
        nUI = int(len(t) / nspui)
        _agg_pulse_resp = agg_pulse_resp[:nUI * nspui]
        sampled_agg_prs: Rmat = array([_agg_pulse_resp[m::nspui] for m in range(nspui)])
        best_m = argmax(list(map(lambda pr_samps: (pr_samps**2).sum(), sampled_agg_prs)))

        return self.varX * abs(rfft(sampled_agg_prs[best_m]))**2 * self.Tb * 2  # i.e. - 2/fB = 1/(fB/2) = 1/fN

    def Stn(self, Hrx: Optional[Cvec] = None) -> Rvec:
        """
        One-sided Tx noise PSD at Rx FFE input,
        uniformly sampled over [0, PI] (rads./s norm.).

        Keyword Args:
            Hrx: Complex voltage transfer function of Rx FFE.
                Default: None (Use flat unity response.)
        """

        f     = self.f

        fom = False
        if Hrx is None:
            Hrx = ones(len(f))
            fom = True

        Htn  = self.Ht * self.H21 * self.Hr * self.Hctf * Hrx * self.Av  # "* Av" is for consistency w/ MATLAB code.
        htn  = irfft(Htn)                               # impulse response of complete signal path, except Tx FFE
        htn_ave_xM = lfilter(ones(self.nspui), 1, htn)  # combination averaging and pre-normalization filter
        _htn = self.baud_rate_sample(htn_ave_xM)        # decimated by `nspui`
        _Htn = rfft(_htn)

        # Stash debugging info if FOM'ing.
        if fom:
            self.Stn_debug = {  # pylint: disable=attribute-defined-outside-init
                'Htn':  Htn,
                'htn':  htn,
                '_Htn': _Htn,
                '_htn': _htn,
            }

        return self.varX * self.Tb * 10**(-self.snr_tx / 10) * abs(_Htn)**2

    @property
    def Sjn(self) -> Rvec:
        """
        One-sided Noise PSD due to jitter at Rx FFE input,
        uniformly sampled over [0, PI] (rads./s norm.).
        """

        t              = self.t
        ts_ix          = self.ts_ix
        Tb             = self.Tb
        nspui          = self.nspui
        varX           = self.varX
        vic_pulse_resp = self.vic_pulse_resp

        dV: Rvec = diff(vic_pulse_resp)
        # Truncate at # of whole UIs in `dV`, to avoid +/-1 length variation, due to sampling phase.
        nUI = int(len(dV) / nspui)
        hJ = mean(reshape(concatenate((dV[(ts_ix - 1) % nspui::nspui][:nUI],
                                       dV[(ts_ix    ) % nspui::nspui][:nUI])),  # noqa=E202
                          shape=(2, nUI)),
                  axis=0) / t[1]

        return varX * (self.Add**2 + self.sigma_Rj**2) * abs(rfft(hJ) * Tb)**2 * Tb  # i.e. - / fB

    def Rn(self) -> Rvec:
        """Noise autocorrelation vector at Rx FFE input."""
        Srn = self.Srn
        Sxn = sum(array(list(map(self.Sxn, self.agg_pulse_resps))), axis=0)
        Stn = self.Stn()
        Sjn = self.Sjn
        min_len = min(len(Srn), len(Sxn), len(Stn), len(Sjn))
        Sn = Srn[:min_len] + Sxn[:min_len] + Stn[:min_len] + Sjn[:min_len]
        # i.e. - `* fB`, which when combined w/ the implicit `1/N` of `irfft()` yields `* df`.
        return irfft(Sn) / self.Tb
