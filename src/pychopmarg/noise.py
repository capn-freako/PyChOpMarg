"""
Noise power spectral density (PSD) calculator for COM.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 2, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from numpy        import argmax, argmin, array, diff, exp, mean, sinc, sum, where
from numpy.fft    import irfft, rfft
from scipy.linalg import toeplitz
from traits.api   import HasTraits  # type: ignore


class NoiseCalc(HasTraits):
    """
    Noise calculator for COM

    Notes:
        1. Subclassing `HasTraits` to support any future desire to have
            this class function as a stand-alone GUI applet.
    """

    def __init__(self, L: int, Tb: float, ts_ix: int, t: Rvec,
                 vic_pulse_resp: Rvec, agg_pulse_resps: list[Rvec],
                 f: Rvec, Ht: Cvec, H21: Cvec, Hr: Cvec, Hctf: Cvec,
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
        self.nspui = Tb // t[1]                     # Samples per unit interval
        self.varX  = (L**2 - 1) / (3 * (L - 1)**2)  # Signal power

    def get_aliased_freq(self, f: float) -> float:
        """
        Return alias of f (Hz).
        """

        fN = self.fN
        if int(f / fN) % 2 == 0:
            return f % fN

        return fN - (f % fN)

    def Srn(self, eta0: float) -> Rvec:
        """
        Rx noise power spectral density (PSD),
        folded (i.e. - aliased) into the discrete frequency domain.

        Args:
            eta0: Noise power spectral density at Rx AFE input.

        Returns:
            Srn: One-sided folded noise PSD at Rx sampler input,
                uniformly sampled over [0, PI] (rads./s norm.).
        """

        fN = self.fN
        rslt = eta0 * abs(self.Hr * self.Hctf) ** 2  # "/ 2" in [1] omitted, since we're only considering: m >= 0.
        # Do the folding.
        for _f, _v in list(zip(self.f, rslt))[np.where(f > fN)]:
            targ_ix = argmin(abs(f - self.get_aliased_freq(_f)))
            rslt[targ_ix] += _v

        return rslt[:np.where(f > fN)[0][0]]

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
        best_m = argmax(list(map(lambda pr_samps: sum(np.array(pr_samps)**2), sampled_agg_prs)))

        return self.varX * abs(rfft(sampled_agg_prs[best_m]))**2 * self.Tb

    def Stn(self, Av: float, snr_tx: float) -> Rvec:
        """
        Transmitter noise PSD at Rx FFE input.

        Args:
            Av: Victim pulse amplitude (V).
            snr_tx: Signal-to-noise ratio of Tx output driver.

        Returns:
            Stn: One-sided Tx noise PSD at Rx FFE input,
                uniformly sampled over [0, PI] (rads./s norm.).
        """

        Tb    = self.Tb
        f     = self.f
        ts_ix = self.ts_ix
        nspui = self.nspui
        varX  = self.varX

        Htn   = self.Ht * self.H21 * self.Hr * self.Hctf
        _htn  = irfft(Av * Tb * sinc(f * Tb) * Htn)
        _t    = array([n * 0.5 / f[-1] for n in range(_htn)])
        htn   = _htn[ts_ix::nspui]

        return varX * exp(10, -snr_tx / 10) * abs(rfft(htn))**2 * Tb


    def Sjn(self, aDD: float, sigmaRj: float) -> Rvec:
        """
        Noise PSD due to jitter at Rx FFE input.

        Args:
            aDD: Peak dual-Dirac jitter (s).
            sigmaRj: Random jitter standard deviation (s).

        Returns:
            Sjn: One-sided Noise PSD due to jitter at Rx FFE input,
                uniformly sampled over [0, PI] (rads./s norm.).
        """

        t              = self.t
        ts_ix          = self.ts_ix
        Tb             = self.Tb
        varX           = self.varX
        vic_pulse_resp = self.vic_pulse_resp

        dV = diff(vic_pulse_resp)
        dt = t[1]
        nspui = Tb / dt
        nUI = int((len(dV) - ts_ix) / nspui)
        hJ = mean(array([dV[ts_ix - 1::nspui], dV[ts_ix::nspui]]), axis=0) / dt

        return varX * (aDD**2 + sigmaRj**2) * abs(rfft(hJ))**2 * Tb


    def Rnn(self, agg_pulse_resps: list[Rvec]) -> Rmat:
        """
        Noise autocorrelation matrix at Rx FFE input.

        Args:

        Returns:

        """

        Sn = self.Srn + sum(array(list(map(self.Sxn, self.agg_pulse_resps))), axis=0) + self.Stn + self.Sjn
        Rn = irfft(Sn) / self.Tb

        return toeplitz(Rn)
