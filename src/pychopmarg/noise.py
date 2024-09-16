"""
Noise power spectral density (PSD) calculator for COM.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 2, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from numpy        import argmax, argmin, array, diff, exp, mean, sinc, sum, where
from numpy.fft    import irfft, rfft
from scipy.linalg import toeplitz
from traits.api   import Array, Float, HasTraits, Int, List, Property, cached_property  # type: ignore

from pychopmarg.common import Rvec, Cvec, Rmat


class NoiseCalc(HasTraits):
    """
    Noise calculator for COM

    Notes:
        1. Subclassing `HasTraits` to support any future desire to have
            this class function as a stand-alone GUI applet.
    """

    Tb     = Float(9.412e-12)      # UI (s)
    fN     = Float(53.125e9)       # Nyquist frequency (Hz)
    Av     = Float(0.6)            # Victim drive level (V)
    ts_ix  = Int(0)                # Main pulse sampling index
    nspui  = Int(32)               # Number of samples per UI
    t      = Array(dtype=float)    # System time vector (s)
    vic_pulse_resp  = Array(dtype=float)  # Victim pulse response (V)
    agg_pulse_resps = List()              # Aggressor pulse responses (V)
    f      = Array(dtype=float)    # System frequency vector (Hz)
    Ht     = Array(dtype=complex)  # Transfer function of Tx output driver risetime
    H21    = Array(dtype=complex)  # Transfer function of terminated interconnect
    Hr     = Array(dtype=complex)  # Transfer function of Rx AFE
    Hctf   = Array(dtype=complex)  # Transfer function of Rx CTLE
    eta0   = Float(0)              # Noise density at Rx AFE input (V^2/Hz)
    varX   = Float(0)              # Signal power (V^2)
    snr_tx = Float(25)             # Tx signal-to-noise ratio (dB)
    Add    = Float(0)              # Dual-Dirac peak amplitude (V)
    sigma_Rj = Float(0)            # Dual-Dirac random stdev (V)

    def __init__(self, L: int, Tb: float, ts_ix: int, t: Rvec,
                 vic_pulse_resp: Rvec, agg_pulse_resps: list[Rvec],
                 f: Rvec, Ht: Cvec, H21: Cvec, Hr: Cvec, Hctf: Cvec,
                 eta0: float, Av: float, snr_tx: float, Add: float, sigma_Rj: float,
                 eps: float = 0.001) -> None:
        """
        ``NoiseCalc`` class initializer.

        Args:
            L: Number of modulation levels.
            Tb: Unit interval (s).
            ts_ix: Main pulse sampling index.
            t: System time vector (for indexing pulse responses) (s).
            vic_pulse_resp: Victim pulse response (V).
            agg_pulse_resps: List of aggressor pulse responses (V).
            f: System frequency vector (for indexing transfer functions) (Hz).
            Ht: Risetime filter transfer function.
            H21: Terminated interconnect transfer function.
            Hr: Rx AFE transfer function.
            Hctf: Rx CTLE transfer function.

        Keyword Args:
            eps: Margin of error when checking floats for integral multiples.

        Returns:
            Initialized ``NoiseCalc`` instance.

        Raises:
            IndexError: Unless `t`, and `vic_pulse_resp` have the same length.
            IndexError: Unless `f`, `Ht`, `H21`, `Hr`, and `Hctf` all have the same length.
            ValueError: Unless `t[0]` == 0.
            ValueError: Unless `Tb` is an integral multiple of `t[1]`.
        """

        assert len(t) == len(vic_pulse_resp), IndexError(
            f"Lengths of `t` ({len(t)}) and `vic_pulse_resp` ({len(vic_pulse_resp)}) must be the same!")
        assert len(f) == len(Ht) == len(H21) == len(Hr) == len(Hctf), IndexError(
            f"Lengths of: `f` ({len(f)}), `Ht` ({len(Ht)}), `H21` ({len(H21)}), `Hr` ({len(Hr)}), and `Hctf` ({len(Hctf)}), must all be the same!")
        assert t[0] == 0, ValueError(
            f"The first element of `t` ({t[0]}) must be zero!")
        assert abs(Tb / t[1] - Tb // t[1]) < eps, ValueError(
            f"`Tb` ({Tb}) must be an integral multiple of `t[1]` ({t[1]})!")

        super(NoiseCalc, self).__init__()

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

        self.fN    = 0.5 / Tb                       # Nyquist frequency
        self.nspui = int(Tb // t[1])                # Samples per unit interval
        self.varX  = (L**2 - 1) / (3 * (L - 1)**2)  # Signal power

    def get_aliased_freq(self, f: float) -> float:
        """
        Return alias of f (Hz).
        """

        fN = self.fN
        if int(f / fN) % 2 == 0:
            return f % fN

        return fN - (f % fN)

    Srn = Property(observe=["fN", "eta0", "Hr", "Hctf", "f"])

    @cached_property
    def _get_Srn(self) -> Rvec:
        """
        One-sided folded noise PSD at Rx sampler input,
        uniformly sampled over [0, PI] (rads./s norm.).
        """

        fN   = self.fN
        f    = self.f

        rslt = self.eta0 * abs(self.Hr * self.Hctf) ** 2  # "/ 2" in [1] omitted, since we're only considering: m >= 0.
        # Do the folding.
        for _f, _v in list(zip(f, rslt))[where(f > fN)]:
            targ_ix = argmin(abs(f - self.get_aliased_freq(_f)))
            rslt[targ_ix] += _v

        return rslt[:where(f > fN)[0][0]]

    def Sxn(self, agg_pulse_resp: Rvec) -> Rvec:
        """
        Crosstalk PSD at Rx FFE input.

        Args:
            agg_pulse_resp: Aggressor pulse response (V).

        Returns:
            Sxn: One-sided crosstalk PSD at Rx FFE input,
                uniformly sampled over [0, PI] (rads./s norm.).
        """

        nspui = self.nspui
        sampled_agg_prs = [agg_pulse_resp[m::nspui] for m in range(nspui)]
        best_m = argmax(list(map(lambda pr_samps: sum(array(pr_samps)**2), sampled_agg_prs)))

        return self.varX * abs(rfft(sampled_agg_prs[best_m]))**2 * self.Tb

    Stn = Property(observe=["Tb", "f", "Ht", "H21", "Hr", "Hctf", "Av", "ts_ix", "nspui", "varX", "snr_tx"])

    @cached_property
    def _get_Stn(self) -> Rvec:
        """
        One-sided Tx noise PSD at Rx FFE input,
        uniformly sampled over [0, PI] (rads./s norm.).
        """

        Tb     = self.Tb
        f      = self.f

        Htn   = self.Ht * self.H21 * self.Hr * self.Hctf
        _htn  = irfft(self.Av * Tb * sinc(f * Tb) * Htn)
        _t    = array([n * 0.5 / f[-1] for n in range(len(_htn))])
        htn   = _htn[self.ts_ix::self.nspui]

        return self.varX * exp(10, -self.snr_tx / 10) * abs(rfft(htn))**2 * Tb


    Sjn = Property(observe=["Tb", "t", "vic_pulse_resp", "ts_ix", "nspui", "varX", "Add", "sigma_Rj"])

    @cached_property
    def _get_Sjn(self) -> Rvec:
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

        dV = diff(vic_pulse_resp)
        dt = t[1]
        nUI = int((len(dV) - ts_ix) / nspui)
        hJ = mean(array([dV[ts_ix - 1::nspui], dV[ts_ix::nspui]]), axis=0) / dt

        return varX * (self.Add**2 + self.sigma_Rj**2) * abs(rfft(hJ))**2 * Tb


    def Rnn(self, agg_pulse_resps: list[Rvec]) -> Rmat:
        """
        Noise autocorrelation matrix at Rx FFE input.

        Args:
            agg_pulse_resps: Aggressor pulse responses (V).

        Returns:
            Rnn: Noise autocorrelation matrix at Rx FFE input.
        """

        Sn = self.Srn() + sum(array(list(map(self.Sxn, self.agg_pulse_resps))), axis=0) + self.Stn + self.Sjn
        Rn = irfft(Sn) / self.Tb

        return toeplitz(Rn)
