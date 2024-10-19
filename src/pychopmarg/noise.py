"""
Noise power spectral density (PSD) calculator for COM.

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 2, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from numpy                  import argmax, argmin, array, concatenate, diff, mean, roll, sinc, sum, where
from numpy.fft              import irfft, rfft, fftshift
from scipy.interpolate      import interp1d
from scipy.linalg           import toeplitz
from traits.api             import Array, Float, HasTraits, Int, List, Property, cached_property  # type: ignore

from pychopmarg.common      import Rvec, Cvec, Rmat


class NoiseCalc(HasTraits):
    """
    Noise calculator for COM

    Notes:
        1. Subclassing `HasTraits` to support any future desire to have
            this class function as a stand-alone GUI applet.
    """

    # Independent variables, set during instance initialization.
    L               = Int(4)                # Number of modulation levels.
    Tb              = Float(9.412e-12)      # UI (s)
    ts_ix           = Int(0)                # Main pulse sampling index
    t               = Array(dtype=float)    # System time vector (s)
    vic_pulse_resp  = Array(dtype=float)    # Victim pulse response (V)
    agg_pulse_resps = List()                # Aggressor pulse responses (V)
    f               = Array(dtype=float)    # System frequency vector (Hz)
    Ht              = Array(dtype=complex)  # Transfer function of Tx output driver risetime
    H21             = Array(dtype=complex)  # Transfer function of terminated interconnect
    Hr              = Array(dtype=complex)  # Transfer function of Rx AFE
    Hctf            = Array(dtype=complex)  # Transfer function of Rx CTLE
    eta0            = Float(0)              # Noise density at Rx AFE input (V^2/GHz)
    Av              = Float(0.6)            # Victim drive level (V)
    snr_tx          = Float(25)             # Tx signal-to-noise ratio (dB)
    Add             = Float(0)              # Dual-Dirac peak amplitude (V)
    sigma_Rj        = Float(0)              # Dual-Dirac random stdev (V)

    # Invariant dependent variables, set during instance initialization.
    fN       = Float(53.125e9)       # Nyquist frequency (Hz)
    nspui    = Int(32)               # Number of samples per UI
    varX     = Float(0)              # Signal power (V^2)
    t_irfft  = Array(dtype=float)    # Time vector for indexing `irfft()` result.

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

        Notes:
            1. No assumption is made, re: any linkage between `t` and `f`.
            2. The transfer functions are assumed to contain only positive frequency values.
            3. `f` may begin at zero, but is not required to.
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
        self.fN      = 0.5 / Tb                                                     # Nyquist frequency
        self.nspui   = int(Tb // t[1])                                              # Samples per unit interval
        self.varX    = (L**2 - 1) / (3 * (L - 1)**2)                                # Signal power
        self.t_irfft = array([n * (0.5 / f[-1]) for n in range(2 * (len(f) - 1))])  # Time indices for `irfft()` output

    def get_aliased_freq(self, f: float) -> float:
        """
        Return alias of f (Hz).
        """

        fN = self.fN
        if int(f / fN) % 2 == 0:
            return f % fN

        return fN - (f % fN)

    def from_irfft(self, x: Rvec) -> Rvec:
        """
        Interpolate `irfft()` output to `t` and subsample at fBaud.

        Args:
            x: `irfft()` results to be interpolated and subsampled.

        Returns:
            y: interpolated and subsampled vector.

        Raises:
            IndexError: If length of input doesn't match length of `t_irfft` vector.

        Notes:
            1. Input vector is shifted, such that its peak occurs at `0.1 * max(t)`, before interpolating.
                This is done to:
                    - ensure that we don't omit any non-causal behavior,
                    which ends up at the end of an IFFT output vector
                    when the peak is very near the beginning, and
                    - to ensure that the majority of our available time span
                    is available for capturing reflections.
            2. The sub-sampling phase is adjusted, so as to ensure that we catch the peak.
        """

        t       = self.t
        t_irfft = self.t_irfft
        nspui   = self.nspui

        assert len(x) == len(t_irfft), IndexError(
            f"Length of input ({len(x)}) must match length of `t_irfft` vector ({len(t_irfft)})!")

        t_pk = 0.1 * t[-1]                      # target peak location time
        targ_ix = where(t_irfft >= t_pk)[0][0]  # target peak vector index, in `x`
        curr_ix = argmax(x)                     # current peak vector index, in `x`
        _x = roll(x, targ_ix - curr_ix)         # `x` with peak repositioned

        krnl = interp1d(t_irfft, _x, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
        y = krnl(t)
        curs_uis, curs_ofst = divmod(argmax(y), nspui)  # Ensure that we capture the peak in the next step.
        return y[curs_ofst::nspui]                      # Sampled at fBaud, w/ peak captured.

    Srn = Property(observe=["fN", "eta0", "Hr", "Hctf", "f"])

    @cached_property
    def _get_Srn(self) -> Rvec:
        """
        One-sided folded noise PSD at Rx sampler input,
        uniformly sampled over [0, PI] (rads./s norm.).

        Notes:
            1. Re: the scaling term: `2 * self.f[-1]`, when combined w/
                the implicit `1/N` of the `irfft()` function, this gives `df`.
        """
        nspui = self.nspui
        rslt  = self.eta0 * 1e-9 * abs(self.Hr * self.Hctf) ** 2  # "/ 2" in [1] omitted, since we're only considering: m >= 0.
        _rslt = abs(rfft(self.from_irfft(irfft(rslt)))) * 2 * self.f[-1] * self.Tb
        return _rslt  # * len(rslt) / len(_rslt)

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

        return self.varX * abs(rfft(sampled_agg_prs[best_m]))**2 * self.Tb  # i.e. - / fB

    Stn = Property(observe=["Tb", "f", "Ht", "H21", "Hr", "Hctf", "Av", "ts_ix", "nspui", "varX", "snr_tx"])

    @cached_property
    def _get_Stn(self) -> Rvec:
        """
        One-sided Tx noise PSD at Rx FFE input,
        uniformly sampled over [0, PI] (rads./s norm.).
        """

        Tb    = self.Tb
        f     = self.f
        nspui = self.nspui

        Htn  = self.Ht * self.H21 * self.Hr * self.Hctf
        # _htn = self.Av * irfft(Tb * sinc(f * Tb) * Htn) * 2 * f[-1]  # See `_get_Srn()`.
        _htn = irfft(sinc(f * Tb) * Htn) * 2  # * f[-1]  # See `_get_Srn()`. But, note that `* df` is not appropriate here.
        # htn  = self.from_irfft(_htn)[self.ts_ix % nspui::nspui]
        htn  = self.from_irfft(_htn)  # ToDo: Do I need to honor `ts_ix`?

        return self.varX * 10**(-self.snr_tx / 10) * abs(rfft(htn))**2 * Tb  # i.e. - / fB

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
        if int(ts_ix % nspui) != 0:
            hJ = mean(concatenate((dV[ts_ix % nspui - 1: -1: nspui], dV[ts_ix % nspui::nspui])).reshape((2, -1)), axis=0) / t[1]
        else:
            hJ = mean(array([dV[nspui - 1::nspui], dV[nspui::nspui]]), axis=0) / t[1]

        # FOR DEBUGGING ONLY!
        self.hJ = hJ
        self.dV = dV

        return varX * (self.Add**2 + self.sigma_Rj**2) * abs(rfft(hJ) * Tb)**2 * Tb  # i.e. - / fB


    def Rn(self) -> Rvec:
        """Noise autocorrelation vector at Rx FFE input."""
        Sn = self.Srn + sum(array(list(map(self.Sxn, self.agg_pulse_resps))), axis=0) + self.Stn + self.Sjn
        return irfft(Sn) / self.Tb  # i.e. - `* fB`, which when combined w/ the implicit `1/N` of `irfft()` yields `* df`.
