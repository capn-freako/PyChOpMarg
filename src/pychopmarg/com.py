"""
The Channel Operating Margin (COM) model, as per IEEE 802.3-22 Annex 93A.

Original author: David Banas <capn.freako@gmail.com>

Original date:   February 29, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

Notes:
    1. Throughout this file, equation numbers refer to Annex 93A of the IEEE 802.3-22 standard.

ToDo:
    1. Provide type hints for imports.
"""

import numpy as np  # type: ignore
import skrf as rf  # type: ignore

from scipy.interpolate import interp1d
from traits.api import HasTraits, Property, Array, Float, cached_property  # type: ignore
from typing import TypeVar, Any, Dict

from pychopmarg.common import Rvec, Cvec, COMParams, PI, TWOPI
from pychopmarg.utility import import_s32p, sdd_21

# Globals used by `calc_Hffe()`, to minimize the size of its cache.
# They are initialized by `COM.__init__()`.
gFreqs: Rvec = None  # type: ignore
gFb: float = None  # type: ignore
gC0min: float = None  # type: ignore
gNtaps: int = None  # type: ignore

T = TypeVar('T', Any, Any)


def all_combs(xss: list[list[T]]) -> list[list[T]]:
    """
    Generate all combinations of input.

    Args:
        xss([[T]]): The lists of candidates for each position in the final output.

    Returns:
        [[T]]: All possible combinations of input lists.
    """
    if not xss:
        return [[]]
    head, *tail = xss
    yss = all_combs(tail)
    return [[x] + ys for x in head for ys in yss]


# @cache
# def calc_Hffe(tap_weights: Rvec) -> Cvec:
def calc_Hffe(tap_weights: list[float]) -> Cvec:
    """
    Calculate the voltage transfer function, H(f), for the Tx FFE,
    according to (93A-21).

    Args:
        tap_weights: The vector of filter tap weights, excluding the cursor tap.

    Returns:
        The complex voltage transfer function, H(f), for the Tx FFE.

    Raises:
        RuntimeError: If the global variables above haven't been initialized.
        ValueError: If the length of the given tap weight vector is incorrect.

    Notes:
        1. This function has been (awkwardly) pulled outside of the
            `COM` class and made to use global variables, strictly for
            performance reasons.
            (Note that `@cached_property` decorated instance functions
            of `HasTraits` subclasses are not actually memoized, like
            `@cache` decorated ordinary functions are.)
            (It is used in the innermost layer of the nested loop structure
            used to find the optimal EQ solution. And its input argument
            is repeated often.)
            (See the `opt_eq()` method of the `COM` class.)
        2. Currently, a single post-cursor tap is assumed.

    ToDo:
        1. Remove the single post-cursor tap assumption.
    """

    assert len(gFreqs) and gFb and gC0min and gNtaps, RuntimeError(
        "Called before global variables were initialized!")
    assert len(tap_weights) == gNtaps, ValueError(
        "Length of given tap weight vector is incorrect!")

    c0 = 1 - sum(list(map(abs, tap_weights)))
    if c0 < gC0min:
        return np.ones(len(gFreqs))
    else:
        cs = tap_weights
        cs.insert(-1, c0)  # Note the assumption of only one post-cursor tap!
        return sum(list(map(lambda n_c: n_c[1] * np.exp(-1j * TWOPI * n_c[0] * gFreqs / gFb),
                            enumerate(cs))))


class COM(HasTraits):
    """
    Encoding of the IEEE 802.3-22 Annex 93A "Channel Operating Margin"
    (COM) specification, as a Python class making use of the Enthought
    Traits/UI machinery, for both calculation efficiency and easy GUI display.
    """

    # Independent variable definitions
    ui = Float(100e-12)  # Unit interval (s).
    freqs = Array(value=np.arange(0, 40_010e6, 10e6))  # System frequencies (Hz).
    gDC = Float(0)  # D.C. gain of Rx CTLE first stage (dB).
    gDC2 = Float(0)  # D.C. gain of Rx CTLE first stage (dB).

    # Dependent variable definitions
    Xsinc = Property(Array, depends_on=["ui", "freqs"])

    @cached_property
    def _get_Xsinc(self):
        """Frequency domain sinc(f) corresponding to Rect(ui)."""
        ui = self.ui
        w = int(ui / self.Ts)
        return w * np.sinc(ui * self.freqs)

    Hr = Property(Array, depends_on=['freqs'])

    @cached_property
    def _get_Hr(self):
        """
        Return the voltage transfer function, H(f), of the Rx AFE,
        according to (93A-20).
        """
        f = self.freqs / (self.params['fr'] * self.fb)
        return 1 / (1 - 3.414214 * f**2 + f**4 + 2.613126j * (f - f**3))

    Hctf = Property(Array, depends_on=['freqs', 'gDC', 'gDC2'])

    @cached_property
    def _get_Hctf(self):
        """
        Return the voltage transfer function, H(f), of the Rx CTLE,
        according to (93A-22).
        """
        f = self.freqs
        g1 = pow(10, self.gDC / 20)
        g2 = pow(10, self.gDC2 / 20)
        fz = self.params['fz'] * 1e9
        fp1 = self.params['fp1'] * 1e9
        fp2 = self.params['fp2'] * 1e9
        fLF = self.params['fLF'] * 1e9
        num = (g1 + 1j * f / fz) * (g2 + 1j * f / fLF)
        den = (1 + 1j * f / fp1) * (1 + 1j * f / fp2) * (1 + 1j * f / fLF)
        return num / den

    # Reserved functions

    def __call__(self, do_opt_eq=True, tx_taps: Rvec = None):
        """
        Calculate the COM value.

        KeywordArgs:
            opt_eq: Perform optimization of linear equalization when True.
                Default: True
            tx_taps: Used when `do_opt_eq` = False.
                Default: None
        """

        assert self.opt_eq(do_opt_eq=do_opt_eq, tx_taps=tx_taps), RuntimeError(
            "EQ optimization failed!")
        self.gDC = self.rslts['gDC']
        self.gDC2 = self.rslts['gDC2']
        self.tx_taps = self.rslts['tx_taps']

        As, Ani, cursor_ix = self.calc_noise()
        com = 20 * np.log10(As / Ani)
        self.com = com
        self.rslts['As'] = As * 1_000  # (mV)
        self.rslts['Ani'] = Ani * 1_000  # (mV)
        self.rslts['com'] = com
        self.cursor_ix = cursor_ix
        return com

    def __init__(self, params: COMParams, chnl_files: list[str], vic_chnl_ix: int,
                 zp_sel: int = 1, num_ui: int = 100, gui: bool = True):
        """
        COM class initializer.

        Args:
            params: COM configuration parameters for desired standard.
                Note: Assumed immutable. ToDo: Can we encode this assumption, using type annotations?
            chnl_files: Touchstone file(s) representing channel, either:
                1. 8 s4p files: [victim, ], or
                2. 1 s32p file, according to VITA 68.2 convention.
            vic_chnl_ix: Victim channel index (from 1).

        KeywordArgs:
            zp_sel: User selection of package T-line length option (from 1).
                Default: 1
            num_ui: Number of unit intervals to include in system time vector.
                Default: 100
            gui: Set to `False` for script/CLI based usage.
                Default: True
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        # Set default parameter values, as necessary.
        if 'zc' not in params:
            params['zc'] = 78.2
        if 'fLF' not in params:
            params['fLF'] = 1

        # Stash function parameters.
        self.params = params
        self.chnl_files = chnl_files
        self.gui = gui

        # Calculate intermediate variable values.
        self.rslts = {}
        self.fom_rslts: Dict[str, Any] = {}
        self.dbg: Dict[str, Any] = {}
        fb = params['fb'] * 1e9
        ui = 1 / fb
        M = self.params['M']
        sample_per = ui / M
        fstep = self.params['fstep']
        trips = list(zip(self.params['tx_min'], self.params['tx_max'], self.params['tx_step']))
        self.tx_combs = all_combs(list(map(
            lambda trip: list(np.arange(trip[0], trip[1] + trip[2], trip[2])),
            trips)))
        self.gamma1 = (self.params['Rd'] - self.params['R0']) / (self.params['Rd'] + self.params['R0'])
        self.gamma2 = self.gamma1

        # Import channel Touchstone file(s).
        if len(chnl_files) == 1:
            _ntwks = import_s32p(chnl_files[0], vic_chnl_ix)
            fmax = _ntwks[0][0].f[-1]
        else:  # ToDo: Do I have the ordering correct here?
            _ntwks = []
            fmax = 1000e9
            n_files = len(chnl_files)
            for n, chnl_file in enumerate(chnl_files):
                ntwk = sdd_21(rf.Network(chnl_file))
                fmax = min(fmax, ntwk.f[-1])
                if n >= n_files // 2:  # This one is a NEXT aggressor.
                    _ntwks.append((ntwk, 'NEXT'))
                elif n > 0:  # This one is a FEXT aggressor.
                    _ntwks.append((ntwk, 'FEXT'))
                else:  # This is the victim.
                    _ntwks.append((ntwk, 'THRU'))

        # Calculate system time/frequency vectors.
        # - We use "decoupled" time/frequency vectors.
        freqs = np.arange(0, fmax + fstep, fstep)  # [0,fN], i.e., one extra element
        times = np.array([n * sample_per for n in range(num_ui * M)])  # independent of `freqs`
        # - That requires a third vector for correctly timing `irfft()` results.
        Ts = 0.5 / fmax  # Sample period satisfying Nyquist criteria.
        t_irfft = np.array([n * Ts for n in range(2 * (len(freqs) - 1))])
        self.freqs = freqs
        self.times = times
        self.Ts = Ts  # Not the same as `sample_per`!
        self.t_irfft = t_irfft

        # Augment w/ packages.
        sTx_NEXT = self.sPkg(zp_opt=1, isTx=True)
        sTx = self.sPkg(zp_opt=zp_sel, isTx=True)
        sRx = self.sPkg(zp_opt=zp_sel, isTx=False)
        ntwks = []
        for ntwk, ntype in _ntwks:
            if ntype == 'NEXT':
                ntwks.append((sTx_NEXT ** ntwk ** sRx, ntype))
            else:
                ntwks.append((sTx ** ntwk ** sRx, ntype))

        # Generate the pulse responses before/after adding the packages, for reference.
        self.ntwks = _ntwks
        self.pulse_resps_nopkg = self.gen_pulse_resps([], apply_eq=False)
        self.ntwks = ntwks
        self.pulse_resps_noeq = self.gen_pulse_resps([], apply_eq=False)
        self.rslts['vic_pulse_pk'] = params['Av'] * max(self.pulse_resps_noeq[0]) * 1_000  # (mV)

        # Store calculated results.
        self.fb = fb
        self.ui = ui
        self.sample_per = sample_per
        self.fmax = fmax
        self.nDFE = len(self.params['dfe_max'])

        # Initialize global variables.
        global gFreqs
        global gFb
        global gC0min
        global gNtaps
        gFreqs = freqs
        gFb = fb
        gC0min = params['c0min']
        gNtaps = len(params['tx_min'])

    # General functions
    def sC(self, c: float) -> rf.Network:
        """
        Return the 2-port network corresponding to a shunt capacitance,
        according to (93A-8).

        Args:
            c: Value of shunt capacitance (F).

        Returns:
            ntwk: 2-port network equivalent to shunt capacitance, calculated at given frequencies.

        Raises:
            None
        """

        r0 = self.params['R0']
        freqs = self.freqs
        w = TWOPI * freqs
        s = 1j * w
        s2p = np.array(
            [1 / (2 + _s * c * r0) * np.array(
                [[-_s * c * r0, 2],
                 [2, -_s * c * r0]])
             for _s in s])
        return rf.Network(s=s2p, f=freqs, z0=[2 * r0, 2 * r0])

    def sZp(self, zp_opt: int = 1) -> rf.Network:
        """
        Return the 2-port network corresponding to a package transmission line,
        according to (93A-9:14).

        KeywordArgs:
            zp_opt: Package TL length option (from 1).
                Default: 1

        Returns:
            ntwk: 2-port network equivalent to package transmission line.

        Raises:
            None
        """

        n_zp_opts = len(self.params['zp'])
        assert zp_opt <= n_zp_opts, ValueError(
            f"Asked for zp option {zp_opt}, but there are only {n_zp_opts}!")
        zc = self.params['zc']
        r0 = self.params['R0']
        zp = self.params['zp'][zp_opt - 1]

        f_GHz  = self.freqs / 1e9               # noqa E221
        a1     = 1.734e-3  # sqrt(ns)/mm        # noqa E221
        a2     = 1.455e-4  # ns/mm              # noqa E221
        tau    = 6.141e-3  # ns/mm              # noqa E221
        rho    = (zc - 2 * r0) / (zc + 2 * r0)  # noqa E221
        gamma0 = 0         # 1/mm
        gamma1 = a1 * (1 + 1j)

        def gamma2(f):
            "f in GHz!"
            return a2 * (1 - 1j * (2 / PI) * np.log(f)) + 1j * TWOPI * tau

        def gamma(f: float) -> complex:
            "Return complex propagation coefficient at frequency f (GHz)."
            if f == 0:
                return gamma0
            else:
                return gamma0 + gamma1 * np.sqrt(f) + gamma2(f) * f

        g = np.array(list(map(gamma, f_GHz)))
        s11 = s22 = rho * (1 - np.exp(-g * 2 * zp)) / (1 - rho**2 * np.exp(-g * 2 * zp))
        s21 = s12 = (1 - rho**2) * np.exp(-g * zp) / (1 - rho**2 * np.exp(-g * 2 * zp))
        s2p = np.array(
            [[[_s11, _s12],
              [_s21, _s22]]
             for _s11, _s12, _s21, _s22 in zip(s11, s12, s21, s22)])
        return rf.Network(s=s2p, f=self.freqs, z0=[2 * r0, 2 * r0])

    def sPkg(self, zp_opt: int = 1, isTx: bool = True) -> rf.Network:
        """
        Return the 2-port network corresponding to a complete package model,
        according to (93A-15:16).

        KeywordArgs:
            zp_opt: Package TL length option (from 1).
                Default: 1
            isTx: Requesting Tx package when True.
                Default: True

        Returns:
            ntwk: 2-port network equivalent to complete package model.

        Raises:
            None
        """

        sd = self.sC(self.params['Cd'] / 1e12)
        sp = self.sC(self.params['Cp'] / 1e12)
        sl = self.sZp(zp_opt)
        if isTx:
            return sd ** sl ** sp
        else:
            return sp ** sl ** sd

    def H21(self, s2p: rf.Network) -> Cvec:
        """
        Return the voltage transfer function, H21(f), of a terminated two
        port network, according to (93A-18).

        Args:
            s2p: Two port network of interest.

        Returns:
            Complex voltage transfer function at given frequencies.

        Raises:
            ValueError: If given network is not two port.

        Notes:
            1. It is at this point in the analysis that the "raw" Touchstone
                data gets interpolated to our system frequency vector.
            2. After this step, the package and R0/Rd mismatch have been
                accounted for, but not the EQ.
        """

        assert s2p.s[0].shape == (2, 2), ValueError("I can only convert 2-port networks.")
        s2p.interpolate_self(self.freqs)
        g1 = self.gamma1
        g2 = self.gamma2
        s11 = s2p.s11.s.flatten()
        s12 = s2p.s12.s.flatten()
        s21 = s2p.s21.s.flatten()
        s22 = s2p.s22.s.flatten()
        dS = s11 * s22 - s12 * s21
        return (s21 * (1 - g1) * (1 + g2)) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * dS)

    def H(self, s2p: rf.Network, tap_weights: Rvec) -> Cvec:
        """
        Return the voltage transfer function, H(f), of a complete COM signal path,
        according to (93A-19).

        Args:
            s2p: Two port network of interest.
            tap_weights: Tx FFE tap weights.

        Returns:
            Complex voltage transfer function of complete path.

        Raises:
            ValueError: If given network is not two port,
                or length of `tap_weights` is incorrect.

        Notes:
            1. Assumes `self.gDC` and `self.gDC2` have been set correctly.
            2. It is in this processing step that linear EQ is first applied.
        """

        assert s2p.s[0, :, :].shape == (2, 2), ValueError(
            f"I can only convert 2-port networks. {s2p}")
        self.H21_temp = self.H21(s2p)
        return calc_Hffe(list(tap_weights)) * self.H21_temp * self.Hr * self.Hctf

    def pulse_resp(self, H: Cvec) -> Rvec:
        """
        Return the unit pulse response, p(t), corresponding to the given
        voltage transfer function, H(f), according to (93A-24).

        Args:
            H: The voltage transfer function, H(f).
                Note: Possitive frequency components only, including fN.

        Returns:
            p: The pulse response corresponding to the given voltage transfer function.

        Raises:
            ValueError: If the length of the given voltage transfer
                function differs from that of the system frequency vector.

        Notes:
            1. It is at this point in the signal processing chain that we change
                time domains.
        """

        assert len(H) == len(self.freqs), ValueError(
            "Length of given H(f) does not match length of f!")

        Xsinc = self.Xsinc
        p = np.fft.irfft(Xsinc * H)
        p_mag = np.abs(p)
        p_beg = np.where(p_mag > 0.01 * max(p_mag))[0][0] - int(5 * self.ui / self.Ts)  # Give it some "front porch".
        spln = interp1d(self.t_irfft, np.roll(p, -p_beg))  # `p` is not yet in our system time domain!
        return spln(self.times)                            # Now, it is.

    def gen_pulse_resps(self, tx_taps: Rvec, apply_eq=True) -> list[Rvec]:
        """
        Generate pulse responses for all networks.

        Args:
            tx_taps: Desired Tx tap weights.

        KeywordArgs:
            apply_eq: Include linear EQ when True; otherwise, exclude it.
                Default: True

        Returns:
            List of pulse responses.

        Raises:
            None

        Notes:
            1. Assumes `self.gDC` and `self.gDC2` have been set correctly.
        """

        pulse_resps = []
        for ntwk, ntype in self.ntwks:
            if apply_eq:
                pr = self.pulse_resp(self.H(ntwk, tx_taps))
            else:
                pr = self.pulse_resp(self.H21(ntwk))

            if ntype == 'THRU':
                pr *= self.params['Av']
            elif ntype == 'NEXT':
                pr *= self.params['Ane']
            else:
                pr *= self.params['Afe']

            pulse_resps.append(pr)

        return pulse_resps

    def filt_pr_samps(self, pr_samps: Rvec, As: float, rel_thresh: float = 0.001) -> Rvec:
        """
        Filter a list of pulse response samples for minimum magnitude.

        Args:
            pr_samps: The pulse response samples to filter.
            As: Signal amplitude, as per 93A.1.6.c.

        KeywordArgs:
            rel_thresh: Filtration threshold (As).
                Default: 0.001 (i.e. - 0.1%, as per Note 2 of 93A.1.7.1)

        Returns:
            filtered_samps: The subset of `pr_samps` passing filtration.
        """

        thresh = As * rel_thresh
        return np.array(list(filter(lambda x: abs(x) >= thresh, pr_samps)))

    def calc_hJ(self, pulse_resp: Rvec, As: float, cursor_ix: int, rel_thresh: float = 0.001) -> Rvec:
        """
        Calculate the set of slopes for valid pulse response samples.

        Args:
            pulse_resp: The pulse response of interest.
            As: Signal amplitude, as per 93A.1.6.c.
            cursor_ix: Cursor index.

        KeywordArgs:
            rel_thresh: Filtration threshold (As).
                Default: 0.001 (i.e. - 0.1%, as per Note 2 of 93A.1.7.1)

        Returns:
            slopes: The calculated slopes around the valid samples.
        """

        M = self.params['M']
        thresh = As * rel_thresh
        valid_pr_samp_ixs = np.array(list(filter(lambda ix: abs(pulse_resp[ix]) >= thresh,
                                                 range(cursor_ix, len(pulse_resp) - 1, M))))
        m1s = pulse_resp[valid_pr_samp_ixs - 1]
        p1s = pulse_resp[valid_pr_samp_ixs + 1]
        return (p1s - m1s) / (2 / M)  # (93A-28)

    def loc_curs(self, pulse_resp: Rvec, max_range: int = 4) -> int:
        """
        Locate the cursor position for the given pulse response,
        according to (93A-25) and (93A-26).

        Args:
            pulse_resp: The pulse response of interest.

        KeywordArgs:
            max_range: The search radius, from the peak.

        Returns:
            The index in the given pulse response vector of the cursor.

        Notes:
            1. As per v3.70 of the COM MATLAB code, we only minimize the
                residual of (93A-25); we don't try to solve it exactly.
        """

        M = self.params['M']
        dfe_max = self.params['dfe_max']
        dfe_min = self.params['dfe_min']

        # Find zero crossings.
        peak_loc = np.argmax(pulse_resp)
        peak_val = pulse_resp[peak_loc]
        search_start = max(0, peak_loc - 4 * M)
        zxi = np.where(np.diff(np.sign(pulse_resp[search_start:peak_loc] - .01 * peak_val)) >= 1)[0] + search_start
        assert zxi, RuntimeError("No zero crossings found!")
        zxi = zxi[-1]

        # Minimize Muller-Mueller criterion within a 2UI range after zero crossing.
        ix_best = zxi
        res_min = 1e6
        for ix in range(zxi, zxi + 2 * M):
            b_1 = min(dfe_max[0],
                      max(dfe_min[0],
                          pulse_resp[ix + M] / pulse_resp[ix]))                          # (93A-26)
            res = abs(pulse_resp[ix - M] - (pulse_resp[ix + M] - b_1 * pulse_resp[ix]))  # (93A-25)
            if res < res_min:
                ix_best = ix
                res_min = res
        return ix_best

    def calc_fom(self, tx_taps: Rvec) -> float:
        """
        Calculate the _figure of merit_ (FOM), given the existing linear EQ settings.

        Args:
            tx_taps: The Tx FFE tap weights, excepting the cursor.
                (The cursor takes whatever is left.)

        Returns:
            FOM: The resultant figure of merit.

        Raises:
            None.

        Notes:
            1. Assumes that `self.gDC` and `self.gDC2` have been set correctly.
            2. See: IEEE 802.3-2022 93A.1.6.
        """

        L = self.params['L']
        M = self.params['M']
        freqs = self.freqs

        # Step a - Pulse response construction.
        pulse_resps = self.gen_pulse_resps(np.array(tx_taps))

        # Step b - Cursor identification.
        vic_pulse_resp = np.array(pulse_resps[0])
        vic_peak_loc = np.argmax(vic_pulse_resp)
        cursor_ix = self.loc_curs(vic_pulse_resp)

        # Step c - As.
        vic_curs_val = vic_pulse_resp[cursor_ix]
        As = self.params['RLM'] * vic_curs_val / (L - 1)

        # Step d - Tx noise.
        varX = (L**2 - 1) / (3 * (L - 1)**2)  # (93A-29)
        varTx = vic_curs_val**2 * pow(10, -self.params['TxSNR'] / 10)  # (93A-30)

        # Step e - ISI.
        nDFE = self.nDFE
        # This is not compliant to the standaard, but is consistent w/ v2.60 of MATLAB code.
        n_pre = cursor_ix // M
        first_pre_ix = cursor_ix - n_pre * M
        vic_pulse_resp_isi_samps = np.concatenate((vic_pulse_resp[first_pre_ix:cursor_ix:M],
                                                   vic_pulse_resp[cursor_ix + M::M]))
        vic_pulse_resp_post_samps = vic_pulse_resp_isi_samps[n_pre:]
        dfe_tap_weights = np.maximum(  # (93A-26)
            self.params['dfe_min'],
            np.minimum(
                self.params['dfe_max'],
                (vic_pulse_resp_post_samps[:nDFE] / vic_curs_val)))
        hISI = vic_pulse_resp_isi_samps \
             - vic_curs_val * np.pad(dfe_tap_weights,  # noqa E127
                                     (n_pre, len(vic_pulse_resp_post_samps) - nDFE),
                                     mode='constant',
                                     constant_values=0)  # (93A-27)
        varISI = varX * sum(hISI**2)  # (93A-31)
        self.dbg['vic_pulse_resp_isi_samps'] = vic_pulse_resp_isi_samps
        self.dbg['vic_pulse_resp_post_samps'] = vic_pulse_resp_post_samps
        self.dbg['hISI'] = hISI

        # Step f - Jitter noise.
        hJ = self.calc_hJ(vic_pulse_resp, As, cursor_ix)
        varJ = (self.params['Add']**2 + self.params['sigma_Rj']**2) * varX * sum(hJ**2)  # (93A-32)

        # Step g - Crosstalk.
        varXT = 0
        for pulse_resp in pulse_resps[1:]:  # (93A-34)
            varXT += max([sum(np.array(self.filt_pr_samps(pulse_resp[m::M], As))**2) for m in range(M)])  # (93A-33)
        varXT *= varX

        # Step h - Spectral noise.
        df = freqs[1]
        varN = self.params['eta0'] * sum(abs(self.Hr * self.Hctf)**2) * (df / 1e9)  # (93A-35)

        # Step i - FOM calculation.
        fom = 10 * np.log10(As**2 / (varTx + varISI + varJ + varXT + varN))  # (93A-36)

        # Stash our calculation results.
        self.fom_rslts['pulse_resps'] = pulse_resps
        self.fom_rslts['vic_pulse_resp'] = vic_pulse_resp
        self.fom_rslts['vic_peak_loc'] = vic_peak_loc
        self.fom_rslts['cursor_ix'] = cursor_ix
        self.fom_rslts['As'] = As
        self.fom_rslts['varTx'] = varTx
        self.fom_rslts['dfe_tap_weights'] = dfe_tap_weights
        self.fom_rslts['varISI'] = varISI
        self.fom_rslts['varJ'] = varJ
        self.fom_rslts['varXT'] = varXT
        self.fom_rslts['varN'] = varN

        return fom

    def opt_eq(self, do_opt_eq: bool = True, tx_taps: Rvec = None) -> bool:
        """
        Find the optimum values for the linear equalization parameters:
        c(-2), c(-1), c(1), gDC, and gDC2, as per IEEE 802.3-22 93A.1.6.

        KeywordArgs:
            do_opt_eq: Perform optimization of linear EQ when True.
                Default: True
            tx_taps: Used when `do_opt_eq` = False.
                Default: None

        Returns:
            success: True if no errors encountered; False otherwise.
        """

        if do_opt_eq:
            # Run the nested optimization loops.
            def check_taps(tx_taps: Rvec) -> bool:
                if (1 - sum(abs(np.array(tx_taps)))) < self.params['c0min']:
                    return False
                else:
                    return True

            fom_max = -100.0
            fom_max_changed = False
            foms = []
            for gDC2 in self.params['gDC2']:
                self.gDC2 = gDC2
                for gDC in self.params['gDC']:
                    self.gDC = gDC
                    for n, tx_taps in enumerate(self.tx_combs):
                        if not check_taps(np.array(tx_taps)):
                            continue
                        fom = self.calc_fom(tx_taps)
                        foms.append(fom)
                        if fom > fom_max:
                            fom_max_changed = True
                            fom_max = fom
                            gDC2_best = gDC2
                            gDC_best = gDC
                            tx_taps_best = tx_taps
                            dfe_tap_weights_best = self.fom_rslts['dfe_tap_weights']
                            cursor_ix_best = self.fom_rslts['cursor_ix']
                            As_best = self.fom_rslts['As']
                            varTx_best = self.fom_rslts['varTx']
                            varISI_best = self.fom_rslts['varISI']
                            varJ_best = self.fom_rslts['varJ']
                            varXT_best = self.fom_rslts['varXT']
                            varN_best = self.fom_rslts['varN']
        else:
            assert tx_taps, RuntimeError("You must define `tx_taps` when setting `do_opt_eq` False!")
            fom = self.calc_fom(tx_taps)
            foms = [fom]
            fom_max = fom
            fom_max_changed = True
            gDC2_best = self.gDC2
            gDC_best = self.gDC
            tx_taps_best = tx_taps
            dfe_tap_weights_best = self.fom_rslts['dfe_tap_weights']
            cursor_ix_best = self.fom_rslts['cursor_ix']
            As_best = self.fom_rslts['As']
            varTx_best = self.fom_rslts['varTx']
            varISI_best = self.fom_rslts['varISI']
            varJ_best = self.fom_rslts['varJ']
            varXT_best = self.fom_rslts['varXT']
            varN_best = self.fom_rslts['varN']

        # Check for error and save the best results.
        if not fom_max_changed:
            return False
        self.rslts['fom'] = fom_max
        self.rslts['gDC2'] = gDC2_best
        self.rslts['gDC'] = gDC_best
        self.rslts['tx_taps'] = tx_taps_best
        self.rslts['dfe_tap_weights'] = dfe_tap_weights_best
        self.rslts['cursor_ix'] = cursor_ix_best
        self.rslts['As'] = As_best * 1_000  # (mV)
        self.rslts['sigma_ISI'] = np.sqrt(varISI_best) * 1_000  # (mV)
        self.rslts['sigma_J'] = np.sqrt(varJ_best) * 1_000  # (mV)
        self.rslts['sigma_XT'] = np.sqrt(varXT_best) * 1_000  # (mV)
        # These two are also calculated by `calc_noise()`; so, add "_best".
        self.rslts['sigma_Tx_best'] = np.sqrt(varTx_best) * 1_000  # (mV)
        self.rslts['sigma_N_best'] = np.sqrt(varN_best) * 1_000  # (mV)
        self.rslts['foms'] = foms
        return True

    def calc_noise(self) -> tuple[float, float, int]:
        """
        Calculate the interference and noise for COM.

        KeywordArgs:
            npts: Number of vector points.
                Default: 2001

        Returns:
            (As, Ani, cursor_ix): Triple containing:
                - signal amplitude,
                - noise + interference amplitude (V), and
                - cursor location within victim pulse response vector.

        Raises:
            None

        Warns:
            1. If `2*As/npts` rises above 10 uV, against standard's recommendation.

        Notes:
            1. Assumes the following instance variables have been set:
                - gDC
                - gDC2
                - tx_taps
        """

        L = self.params['L']
        M = self.params['M']
        freqs = self.freqs
        nDFE = self.nDFE

        pulse_resps = self.gen_pulse_resps(np.array(self.tx_taps))
        vic_pulse_resp = pulse_resps[0]
        cursor_ix = self.loc_curs(vic_pulse_resp)
        vic_curs_val = vic_pulse_resp[cursor_ix]
        As = self.params['RLM'] * vic_curs_val / (L - 1)
        npts = 2 * max(int(As / 0.001), 1_000) + 1  # Note 1 of 93A.1.7.1; MUST BE ODD!
        y = np.linspace(-As, As, npts)
        ystep = 2 * As / (npts - 1)

        delta = np.zeros(npts)
        delta[npts // 2] = 1
        varX = (L**2 - 1) / (3 * (L - 1)**2)  # (93A-29)
        df = freqs[1]

        def pn(hn: float) -> Rvec:
            """
            (93A-39)
            """
            return 1 / L * sum([np.roll(delta, int((2 * el / (L - 1) - 1) * hn / ystep))
                                for el in range(L)])

        def p(h_samps: Rvec) -> Rvec:
            """
            Calculate the "delta-pmf" for a set of pulse response samples,
            as per (93A-40).

            Args:
                h_samps: Vector of pulse response samples.
                As: Signal amplitude, as per 93A.1.6.c.

            Returns:
                Vector of "deltas" giving amplitude probability distribution.

            Raises:
                None

            Notes:
                1. The input set of pulse response samples is filtered,
                    as per Note 2 of 93A.1.7.1.
            """

            pns = []
            rslts = []
            rslt = delta
            rslts.append(rslt)
            for hn in h_samps:
                _pn = pn(hn)
                pns.append(_pn)
                rslt = np.convolve(rslt, _pn, mode='same')
                rslts.append(rslt)
            rslt /= sum(rslt)  # Enforce a PMF. Commenting out didn't make a difference.
            self.dbg['pns'] = pns
            self.dbg['rslts'] = rslts
            return rslt

        # Sec. 93A.1.7.2
        varN = self.params['eta0'] * sum(abs(self.Hr * self.Hctf)**2) * (df / 1e9)  # (93A-35)
        varTx = vic_curs_val**2 * pow(10, -self.params['TxSNR'] / 10)               # (93A-30)
        hJ = self.calc_hJ(vic_pulse_resp, As, cursor_ix)
        pJ = p(self.params['Add'] * hJ)
        self.dbg['pJ'] = pJ
        self.dbg['hJ'] = hJ
        varG = varTx + self.params['sigma_Rj']**2 * varX * sum(hJ**2) + varN  # (93A-41)
        pG = np.exp(-y**2 / (2 * varG)) / np.sqrt(TWOPI * varG)               # (93A-42)
        pN = np.convolve(pG, pJ, mode='same')                                 # (93A-43)

        # Sec. 93A.1.7.3
        # - ISI (Inconsistent w/ IEEE 802.3-22, but consistent w/ v2.60 of MATLAB code.)
        n_pre = cursor_ix // M
        first_pre_ix = cursor_ix - n_pre * M
        vic_pulse_resp_isi_samps = np.concatenate((vic_pulse_resp[first_pre_ix:cursor_ix:M],
                                                   vic_pulse_resp[cursor_ix + M::M]))
        vic_pulse_resp_post_samps = vic_pulse_resp_isi_samps[n_pre:]
        dfe_tap_weights = np.maximum(  # (93A-26)
            self.params['dfe_min'],
            np.minimum(
                self.params['dfe_max'],
                (vic_pulse_resp_post_samps[:nDFE] / vic_curs_val)))
        hISI = vic_pulse_resp_isi_samps \
             - vic_curs_val * np.pad(dfe_tap_weights,  # noqa E127
                                     (n_pre, len(vic_pulse_resp_post_samps) - nDFE),
                                     mode='constant',
                                     constant_values=0)  # (93A-27)
        hISI = self.filt_pr_samps(hISI, As)
        py = p(hISI)  # `hISI` from (93A-27); `p(y)` as per (93A-40)

        # - Crosstalk
        self.rslts['py0'] = py.copy()  # For debugging.
        xt_samps = []
        pks = []  # For debugging.
        for pulse_resp in pulse_resps[1:]:  # (93A-44)
            i = np.argmax([sum(np.array(self.filt_pr_samps(pulse_resp[m::M], As))**2) for m in range(M)])  # (93A-33)
            samps = self.filt_pr_samps(pulse_resp[i::M], As)
            xt_samps.append(samps)
            pk = p(samps)  # For debugging.
            pks.append(pk)
            py = np.convolve(py, pk, mode='same')
        self.rslts['py1'] = py.copy()  # For debugging.
        self.xt_samps = xt_samps
        self.pks = pks
        py = np.convolve(py, pN, mode='same')  # (93A-45)

        # Final calculation
        Py = np.cumsum(py)
        Py /= Py[-1]  # Enforce cumulative probability distribution.

        # Store some results.
        self.pulse_resps = pulse_resps
        self.cursor_ix = cursor_ix
        self.rslts['sigma_Tx'] = np.sqrt(varTx) * 1_000  # (mV)
        self.rslts['sigma_G'] = np.sqrt(varG) * 1_000  # (mV)
        self.rslts['sigma_N'] = np.sqrt(varN) * 1_000  # (mV)
        self.rslts['pG'] = pG
        self.rslts['pN'] = pN
        self.rslts['py'] = py
        self.rslts['Py'] = Py
        self.rslts['y'] = y
        self.rslts['dfe_taps'] = dfe_tap_weights

        return (As,
                abs(np.where(Py >= self.params['DER'])[0][0] - npts // 2) * ystep,
                cursor_ix)
