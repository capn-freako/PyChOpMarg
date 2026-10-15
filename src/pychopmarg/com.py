# pylint: disable=too-many-lines

"""
The Channel Operating Margin (COM) model, as per IEEE 802.3-22 Annex 93A/178A.

Original author: David Banas <capn.freako@gmail.com>

Original date:   February 29, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

Notes:
    1. Throughout this file, equation numbers refer to Annex 93A of the IEEE 802.3-22 standard.
    2. Throughout this file, reference may be made to the following, via "[n]" syntax:
        [1] - Healey, A., Hegde, R., _Reference receiver framework for 200G/lane electrical interfaces and PHYs_,
            IEEE P802.3dj Task Force, January 2024 (r4).

ToDo:
    1. Provide type hints for imports.
"""

from pathlib import Path
from typing  import Any, Optional, cast

import numpy as np  # type: ignore
import skrf  as rf  # type: ignore
from numpy             import array, arange
from scipy.signal       import fftconvolve  # type: ignore

from pychopmarg.common import (
    PI, TWOPI, Cvec, Rvec, Cmat, OptMode,
    COMChnl, COMNtwk, ChnlSet, ChnlGrpName, ChnlSetName)
from pychopmarg.config.template    import COMParams
from pychopmarg.devmodel import DeviceModel, gamma_term, renormalize_to, s_rx_model, s_tx_model
from pychopmarg.noise    import NoiseCalc
from pychopmarg.optimize import NormMode, mmse, przf
from pychopmarg.utility  import (
    import_s32p, sdd_21, sDieLadderSegment, sPkgTline, sCshunt, filt_pr_samps,
    delta_pmf, mk_combs, calc_Hffe, calc_Hctle, calc_H21, calc_hJ, loc_curs)


def _conv_pmfs(p1: Rvec, p2: Rvec) -> Rvec:
    """
    Convolve two PMFs on the same, zero-centered voltage grid, returning a PMF on that grid.

    (Via FFT, since COM's grids can be long; any FFT round-off below zero is clipped.)
    """
    p = np.maximum(fftconvolve(p1, p2, mode='same'), 0.0)
    return p / p.sum()


class COM:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    Encoding of the IEEE 802.3-22 Annex 93A/178A 'Channel Operating Margin' (COM) specification.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
        self,
        com_params: COMParams,
        channels: Path | dict[str, list[Path]],
        vic_chnl_ix: int = 1,
        debug: bool = False,
        do_init: bool = True,
        tx_device: Optional[DeviceModel] = None,
        rx_device: Optional[DeviceModel] = None,
        next_device: Optional[DeviceModel] = None,
    ) -> None:
        """
        Args:
            com_params: The COM parameters for this instance.
            channels: Channel file path(s), either:

                - A single "*.s32p" containing a 32-port Touchstone model, or

                - A dictionary containing the following key/value pairs:

                    - "THRU": A singleton list containing the path to the "*.s4p" of the thru-channel.
                    - "FEXT": A list of "*.s4p"s containing the Touchstone models for all far-end aggressors.
                    - "NEXT": A list of "*.s4p"s containing the Touchstone models for all near-end aggressors.

                  Any list element may instead be a differential 2-port ``skrf.Network``, used as is.

        Keyword Args:
            vic_chnl_ix: Index (1-based) of victim path in s32p.
                Default: 1
            debug: Gather/report certain debugging information when ``True``.
                Default: ``False``
            do_init: Skip normal initialization behavior when ``False``.
                Default: ``True``
            tx_device: Transmitter device termination/package/partial host model (P802.3dj 178A.1.4).
                When given along with ``rx_device``, it replaces the package model built from ``com_params``;
                terminations then come from each device's ``Rd``, and channels are renormalized to ``2 * R_0``.
                Default: ``None``
            rx_device: Receiver device model (P802.3dj 178A.1.5). (See ``tx_device``.)
                Default: ``None``
            next_device: Near-end aggressor transmitter device model.
                Default: ``None`` (use ``tx_device``)
        """

        # General
        self.com_params: COMParams = com_params
        self.debug: bool = debug
        self.do_init: bool = do_init
        self.status_str: str = "Ready"
        self.opt_mode: OptMode = OptMode(OptMode.MMSE)
        self.norm_mode: NormMode = NormMode(NormMode.P8023dj)
        self.unit_amp: bool = True

        # Linear EQ
        self.tx_ix: int = 0             #: Index into list of all possible combinations of Tx FFE tap weights.
        self.rx_taps: Rvec = array([1.0])
        self.dfe_taps: Rvec = array([])
        self.nRxTaps: int = 0
        self.nRxPreTaps: int = 0        #: `dw` from `com_params`
        self.gDC: float = 0.0           #: Choices are in `com_params.g_DC`.
        self.gDC2: float = 0.0          #: Choices are in `com_params.g_DC2`.
        self.rx_ffe_phase_matrix: Cmat = array([])
        self._Htx_cache: dict[int, Cvec] = {}  # Tx FFE frequency responses, by `tx_taps_ix` (see `Htx()`).

        # Channel data
        self.chnls: list[COMChnl] = []
        self.chnls_noPkg: list[COMChnl] = []
        self.pulse_resps_nopkg: list[Rvec] = []
        self.pulse_resps_noeq: list[Rvec] = []
        self.cursor_ix: int = 0

        # Package
        self.zp_sel: int = 0            #: Package length selector.
        self._sPkgTx: list[rf.Network] = []
        self._sPkgRx: list[rf.Network] = []
        self._sPkgNext: list[rf.Network] = []
        self._sPkgFext: list[rf.Network] = []

        if not do_init:
            return

        # Process the given channel file names.
        ntwks: list[COMNtwk] = []
        if isinstance(channels, str):  # Should be a "*.s32p".
            assert channels.endswith(".s32p"), ValueError(
                "When `channels` is a string it must contain a '*.s32p' value!")
            if channels.exists() and channels.is_file():
                ntwks = import_s32p(channels, vic_chnl_ix)
            else:
                raise RuntimeError(
                    f"Unable to import '{channels}'!")
        elif isinstance(channels, dict):  # Using s4p files.
            chnl_types = ["THRU", "FEXT", "NEXT"]
            assert all(k in channels and isinstance(channels[k], list)
                       for k in chnl_types), ValueError(
                f"When `channels` is a dictionary it must contain: {chnl_types} keys, which must all refer to lists!")
            assert len(channels["THRU"]) == 1, ValueError(
                f"Length of `channels['THRU']` must be 1, not {len(channels['THRU'])}!")
            named_chnls = [(chnl_name, chnl_type) for chnl_type in chnl_types for chnl_name in channels[chnl_type]]
            for fname, chtype in named_chnls:
                if isinstance(fname, rf.Network) and fname.nports == 2:  # Already differential.
                    ntwks.append((fname, chtype))
                else:
                    ntwks.append((sdd_21(rf.Network(str(fname))), chtype))
        else:
            raise TypeError(f"`channels` must be of type 'str' or 'dict', not '{type(channels)}'!")

        # Create the system time & frequency vectors.
        fb = com_params.fb * 1e9
        fstep = com_params.fstep * 1e9
        # fstep = 50e6               # Dramatically improves performance.
        # - time
        tmax = 1 / fstep           # Just enough to cover one full cycle of the fundamental.
        ui = 1 / fb
        tstep = ui / com_params.M  # Obeying requested samps. per UI.
        t = arange(0, tmax, tstep)
        # - freq.
        fmax = 0.5 / t[1]  # Nyquist freq.
        f = arange(0, fmax + fstep, fstep)  # "+ fstep", to include `fmax`.
        self.t: Rvec = t
        self.f: Rvec = f

        # Pre-calculate constant responses.
        self._Xsinc = int(ui / t[1]) * np.sinc(ui * f)
        self._Ht = np.exp(-2 * (PI * (f / 1e9) * com_params.T_r / 1.6832)**2)  # 93A-46 calls for f in GHz.
        _f = f / (com_params.f_r * fb)
        H_Butterworth   = 1 / (1 - 3.414214 * _f**2 + _f**4 + 2.613126j * (_f - _f**3))
        self._Hr = H_Butterworth
        Rd = com_params.R_d
        R0 = com_params.R_0
        self._gamma1: Rvec = (Rd - R0) / (Rd + R0)
        self.use_devices = tx_device is not None and rx_device is not None
        if self.use_devices:
            assert tx_device is not None and rx_device is not None  # (for type checkers)
            self._gamma1 = array([gamma_term(tx_device.Rd, R0), gamma_term(rx_device.Rd, R0)])
            self._sPkgTx = [s_tx_model(self.freqs, tx_device, R0)]
            self._sPkgRx = [s_rx_model(self.freqs, rx_device, R0)]
            self._sPkgNext = [s_tx_model(self.freqs, next_device or tx_device, R0)]
            self._sPkgFext = self._sPkgTx
            self.chnls = list(map(self.add_pkg, ntwks))
        else:
            self._build_legacy_pkg(ntwks)
        if debug:
            self.chnls_noPkg = [(ntwks[0], calc_H21(f, ntwks[0][0], self.gamma1_Tx, self.gamma2_Rx))]
            self.pulse_resps_nopkg = self.gen_pulse_resps(chnls=[self.chnls_noPkg[0]], apply_eq=False)
            self.pulse_resps_noeq  = self.gen_pulse_resps(chnls=[self.chnls[0]],       apply_eq=False)

        # Generate all possible combinations of Tx FFE tap weights, in grid order (first tap varying slowest),
        # along with each one's per-tap grid indices (used by the local search in ``opt_eq()``).
        c0_min = com_params.c0_min
        trips = list(zip(com_params.tx_taps_min,
                         com_params.tx_taps_max,
                         com_params.tx_taps_step))
        tx_grid_lens = [len(mk_combs([trip])) for trip in trips]
        _tx_grid = [(comb, ixs) for comb, ixs in zip(mk_combs(trips), np.ndindex(*tx_grid_lens))
                    if (1 - abs(comb).sum()) >= c0_min]
        # (Index 0 is always "no Tx EQ", which NEXT aggressors use.)
        self._tx_combs = [np.zeros(len(com_params.tx_taps_min)), *[comb for comb, _ in _tx_grid]]
        self._tx_grid_ixs: list[Optional[tuple[int, ...]]] = [None, *[ixs for _, ixs in _tx_grid]]
        self._tx_grid_lens = tx_grid_lens
        self._num_tx_combs = len(self._tx_combs)

        # Set Rx FFE quantities.
        self.nRxTaps = len(com_params.rx_taps_max)
        self.nRxPreTaps = com_params.dw
        self.rx_ffe_phase_matrix = np.exp(np.outer(np.arange(self.nRxTaps), -1j * TWOPI * ui * f))
        self.null_rx_ffe: Rvec = array([0] * self.nRxPreTaps + [1.0] + [0] * (self.nRxTaps - self.nRxPreTaps - 1))
        self.empty_array: Rvec = array([])

        # Misc.
        self.com_rslts: dict[str, Any] = {}
        self.fom_rslts: dict[str, Any] = {}
        self.dbg_dict: Optional[dict[str, Any]] = None

        self.set_status("Ready")

    def __call__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        opt_mode:  Optional[OptMode]  = None,
        norm_mode: Optional[NormMode] = None,
        unit_amp:  Optional[bool]     = None,
        dbg_dict:  Optional[dict[str, Any]] = None
    ) -> float:
        """
        Calculate the COM value.

        Keyword Args:
            opt_mode: Optimization mode.
                Default: None (i.e. - Use `self.opt_mode`.)
            norm_mode: The tap weight normalization mode to use.
                Default: None (i.e. - Use `self.norm_mode`.)
            unit_amp: Enforce unit pulse response amplitude when True.
                (For comparing `przf()` results to `mmse()` results.)
                Default: None (i.e. - Use `self.unit_amp`.)
            dbg_dict: Optional dictionary into which debugging values may be stashed,
                for later analysis.
                Default: None

        Returns:
            COM: The calculated COM value (dB).

        Raises:
            RuntimeError if throughput channel is missing.
            RuntimeError if EQ optimization fails.
        """

        # Honor any mode overrides.
        if opt_mode:
            self.opt_mode  = opt_mode
        if norm_mode:
            self.norm_mode = norm_mode
        if unit_amp:
            self.unit_amp  = unit_amp
        if dbg_dict is not None:
            self.dbg_dict  = dbg_dict

        self.set_status("Optimizing EQ...")
        assert self.opt_eq(), RuntimeError("EQ optimization failed!")
        self.set_status("Calculating noise...")
        As, Ani, self.cursor_ix = self.calc_noise(
            cursor_ix=self.fom_rslts['cursor_ix'])
        com = 20 * np.log10(As / Ani)
        self.com_rslts["COM"] = com
        self.set_status(f"Ready; COM = {com: 5.1f} dB")
        return com

    # Instance Properties
    def _build_legacy_pkg(self, ntwks: list[COMNtwk]) -> None:
        "Build the (pre-178A) package model from `com_params`, and add it to the channels."
        com_params = self.com_params

        def tline(z_p: Optional[list[float]]) -> rf.Network:
            "Package transmission line, w/ the given segment lengths (default: ``[z_p[zp_sel], z_pB]``)."
            if z_p is None:
                z_p = [com_params.z_p[self.zp_sel], com_params.z_pB]
            return sPkgTline(self.freqs, com_params.R_0, com_params.a1, com_params.a2,
                             com_params.tau, com_params.gamma0, list(zip(com_params.z_c, z_p)))

        def diff(sPkg_SE: rf.Network) -> rf.Network:
            "Differential version of the given single-ended package model."
            return sdd_21(rf.network.concat_ports([sPkg_SE, sPkg_SE], port_order='first'))

        def tx_pkg(z_p: Optional[list[float]]) -> rf.Network:
            "Transmitter-end package model."
            return diff(rf.network.cascade_list([
                self.sDie(False), tline(z_p), self.sC(com_params.C_p[0] / 1e9)]))

        cp_rx_ix = min(1, len(com_params.C_p) - 1)
        self._sPkgTx   = [tx_pkg(com_params.zp_tx)]
        self._sPkgNext = [tx_pkg(com_params.zp_next)]
        self._sPkgFext = [tx_pkg(com_params.zp_fext)]
        self._sPkgRx   = [diff(rf.network.cascade_list([
            self.sC(com_params.C_p[cp_rx_ix] / 1e9), tline(com_params.zp_rx), self.sDie(True)]))]

        self.chnls = list(map(self.add_pkg, ntwks))

    @property
    def fb(self) -> float:
        "Baud. rate (Hz)"
        return self.com_params.fb * 1e9

    @property
    def ui(self) -> float:
        "Unit interval (s)"
        return 1 / self.fb

    @property
    def nspui(self) -> int:
        "Samples per UI"
        return self.com_params.M

    @property
    def times(self) -> Rvec:
        "System time vector (s); decoupled from system frequency vector!"
        return self.t

    @property
    def fstep(self) -> float:
        "Frequency step (Hz)"
        return self.f[1]

    @property
    def fmax(self) -> float:
        "Maximum frequency (Hz)"
        return self.f[-1]

    @property
    def freqs(self) -> Rvec:
        "System frequency vector (Hz); decoupled from system time vector!"
        return self.f

    @property
    def Xsinc(self) -> Rvec:
        """Frequency domain sinc(f) corresponding to Rect(ui) in time domain."""
        return self._Xsinc

    @property
    def Ht(self) -> Cvec:
        """
        Return the voltage transfer function, H(f), associated w/ the Tx risetime,
        according to (93A-46).
        """
        return self._Ht

    @property
    def num_tx_combs(self) -> int:
        "Return the number of available Tx FFE tap weight combinations."
        return self._num_tx_combs

    @property
    def Hr(self) -> Cvec:
        """
        Return the voltage transfer function, H(f), of the Rx AFE,
        according to (93A-20).
        """
        return self._Hr

    @property
    def Hctf(self) -> Cvec:
        """
        Return the voltage transfer function, H(f), of the Rx CTLE,
        according to (93A-22).
        """
        return self.calc_Hctf(self.gDC, self.gDC2)

    def calc_Hctf(self, gDC: Optional[float] = None, gDC2: Optional[float] = None) -> Cvec:
        """
        Return the voltage transfer function, H(f), of the Rx CTLE,
        according to (93A-22).

        Keyword Args:
            gDC: CTLE first stage d.c. gain (dB).
                Default: None
            gDC2: CTLE second stage d.c. gain (dB).
                Default: None

        Notes:
            1. The instance's current value(s) for ``gDC`` and ``gDC2`` are used if not provided.
            (Necessary, to accommodate sweeping when optimizing EQ.)
        """
        if gDC is None:
            gDC = self.gDC
        if gDC2 is None:
            gDC2 = self.gDC2
        return calc_Hctle(self.freqs, self.com_params.f_z * 1e9, self.com_params.f_p1 * 1e9,
                          self.com_params.f_p2 * 1e9, self.com_params.f_LF * 1e9, gDC, gDC2)

    @property
    def gamma1_Tx(self) -> float:
        "Reflection coefficient looking out of the left end of the channel."
        return self._gamma1[0]

    @property
    def gamma2_Tx(self) -> float:
        "Reflection coefficient looking out of the right end of the channel."
        return self._gamma1[0]

    @property
    def gamma1_Rx(self) -> float:
        "Reflection coefficient looking out of the left end of the channel."
        return self._gamma1[min(1, len(self._gamma1) - 1)]

    @property
    def gamma2_Rx(self) -> float:
        "Reflection coefficient looking out of the right end of the channel."
        return self._gamma1[min(1, len(self._gamma1) - 1)]

    def sDie(self, isRx: bool) -> rf.Network:
        "On-die parasitic capacitance/inductance ladder network, including bump."
        ix = 1 if isRx else 0
        # C_d[ix], L_s[ix], C_b[ix] may be scalars (single-segment die model);
        # use atleast_1d so zip/len work regardless of whether the config
        # supplies one value or a list of per-segment values.
        # L_s and C_b configs may have fewer than 2 elements — fall back to [0].
        ls_ix = min(ix, len(self.com_params.L_s) - 1)
        cb_ix = min(ix, len(self.com_params.C_b) - 1)
        Cd = np.atleast_1d(np.array(self.com_params.C_d[ix], dtype=float) / 1e9)
        Ls = np.atleast_1d(np.array(self.com_params.L_s[ls_ix], dtype=float) / 1e9)
        R0 = [self.com_params.R_0] * len(Cd)  # type: ignore
        rslt = rf.network.cascade_list(
            [sDieLadderSegment(self.freqs, trip) for trip in zip(R0, Cd, Ls)])  # type: ignore
        rslt = rslt ** self.sC(self.com_params.C_b[cb_ix] / 1e9)
        if isRx:
            rslt.flip()
        return rslt

    @property
    def sPkgRx(self) -> rf.Network:
        "Rx package response."
        return self._sPkgRx[self.zp_sel]

    @property
    def sPkgTx(self) -> rf.Network:
        "Tx package response."
        return self._sPkgTx[self.zp_sel]

    @property
    def sPkgNEXT(self) -> rf.Network:
        "NEXT package response."
        return self._sPkgNext[0]

    @property
    def sPkgFEXT(self) -> rf.Network:
        "FEXT package response."
        return self._sPkgFext[0]

    # Package modeling
    def add_pkg(self, ntwk: COMNtwk) -> COMChnl:
        """
        Add package response to raw channel and pre-calculate H21.

        ToDo:
            1. (Important) Eliminate the S->T translation, i.e. - interpolate as S parameters always.
            Its 1/S21 fails wherever S21 vanishes (hence the fallback below), and it's needlessly
            ill-conditioned for weakly coupled aggressors. But first, comprehensively compare the
            results of the two bases (pulse responses, FOM, COM, chosen EQ), across all channel groups,
            and against MATLAB COM, before committing to that path.
        """

        # Pre-interpolate channel model to system frequency vector, to prevent noisy non-causal artifacts.
        freqs = self.freqs
        sChnl = ntwk[0]
        sChnlDC = sChnl.extrapolate_to_dc()
        # T parameters need 1/S21, so fall back to S parameters if S21 vanishes anywhere. (Weakly coupled aggressors'
        # S21 can, when written with too few decimal places: e.g., -240 dB near 100 GHz, to 1e-12, cancels to zero.)
        basis = 't' if np.all(sChnlDC.s[:, 1, 0]) else 's'
        # (`interpolate()` is typed `Network | ndarray`; it returns the latter only when given `return_array=True`.)
        sChnlInterp = cast(rf.Network, sChnlDC.interpolate(
            freqs[freqs <= sChnl.f[-1]], kind='cubic', coords='polar', basis=basis, assume_sorted=True))
        new_s = sChnlInterp.s.tolist()  # I couldn't make NumPy array padding work correctly here.
        pad_len = len(freqs) - len(sChnlInterp.f)
        last_s = [[sChnlInterp.s11.s[-1, 0, 0], sChnlInterp.s12.s[-1, 0, 0]],
                  [sChnlInterp.s21.s[-1, 0, 0], sChnlInterp.s22.s[-1, 0, 0]]]
        new_s.extend([last_s] * pad_len)
        new_s = np.array(new_s)
        sChnlInterp = rf.Network(s=new_s, z0=sChnlInterp.z0[0], f=freqs)
        if self.use_devices:  # Transform to the reference impedance of the device models (178A-4).
            sChnlInterp = renormalize_to(sChnlInterp, 2 * self.com_params.R_0)

        ntype = ntwk[1]
        if ntype == 'NEXT':
            _ntwk = self.sPkgNEXT ** sChnlInterp ** self.sPkgRx
        elif ntype == 'FEXT':
            _ntwk = self.sPkgFEXT ** sChnlInterp ** self.sPkgRx
        else:
            _ntwk = self.sPkgTx   ** sChnlInterp ** self.sPkgRx
        return ((_ntwk, ntype), calc_H21(self.freqs, _ntwk, self.gamma1_Tx, self.gamma2_Rx))

    # Logging / Debugging
    def set_status(self, status: str) -> None:
        "Set the GUI status string and print it if we're debugging."
        self.status_str = status
        if self.debug:
            # print(status, flush=True)
            pass

    # General functions
    def sC(self, c: float) -> rf.Network:
        """
        Return the 2-port network corresponding to a shunt capacitance,
        according to (93A-8).

        Args:
            c: Value of shunt capacitance (F).

        Returns:
            2-port network equivalent to shunt capacitance, calculated at given frequencies.
        """
        return sCshunt(self.freqs, c, self.com_params.R_0)

    def Htx(self, tx_taps_ix: int) -> Cvec:
        """
        Return the complex frequency response of the Tx deemphasis filter.

        Args:
            tx_taps_ix: Index into the Tx tap weight combinations list.

        Returns:
            Complex frequency response of Tx deemphasis filter.
        """
        assert 0 <= tx_taps_ix < self.num_tx_combs, ValueError(
            f"tx_taps_ix ({tx_taps_ix}) must be an integer in: [0, {self.num_tx_combs})!")
        # Cached per instance: `functools.cache` on a method would keep every `COM` alive, via its `self` keys.
        if tx_taps_ix not in self._Htx_cache:
            self._Htx_cache[tx_taps_ix] = calc_Hffe(self.freqs, 1 / self.fb, array(self._tx_combs[tx_taps_ix]), 3)
        return self._Htx_cache[tx_taps_ix]

    def Hffe_Rx(self, taps: Optional[Rvec] = None) -> Cvec:
        """
        Return the complex frequency response of the Rx FFE.

        Keyword Args:
            taps: Tap weights to use for calculation.
                Default: None (Means "use self.rx_taps".)

        Returns:
            Complex frequency response of Rx FFE.

        Raises:
            ValueError: If ``taps`` is supplied and the length is wrong.
        """
        if taps is None:
            taps = self.rx_taps
        else:
            assert len(taps) == self.nRxTaps, ValueError(
                f"If `taps` is given then its length ({len(taps)}) must equal `nRxTaps` ({self.nRxTaps})!")

        return taps @ self.rx_ffe_phase_matrix

    def H(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
        self, H21: Cvec, tx_ix: int,
        Hctf: Optional[Cvec] = None,
        rx_taps: Optional[Rvec] = None, dfe_taps: Optional[Rvec] = None,
        passive_RxFFE: bool = False
    ) -> Cvec:
        """
        Return the voltage transfer function, H(f), of a complete COM signal path,
        according to (93A-19).

        Args:
            H21: Voltage transfer function of channel + package.
            tx_ix: Tx FFE tap weights index.

        Keyword Args:
            Hctf: Complex voltage transfer function of CTLE.
                Default: None (i.e. - Calculate, using ``self.gDC`` & ``self.gDC2``.)
            rx_taps: Rx FFE tap weights.
                Default: None (i.e. - Use ``self.rx_taps``.)
            dfe_taps: Rx DFE tap weights.
                Default: None (i.e. - Use ``self.dfe_taps``.)
            passive_RxFFE: Enforce passivity of Rx FFE when True.
                Default: True

        Returns:
            Complex voltage transfer function of complete path.

        Raises:
            ValueError: If the length of the given voltage transfer function
            differs from the length of the system frequency vector.

        Notes:
            1. It is in this processing step that linear EQ is first applied.

            2. Any unprovided EQ values are taken from the ``COM`` instance.
            If you really want to omit a particular EQ component then call with:

                - ``tx_taps``: []
                - ``rx_taps``: [1.0]
                - ``Hctle``: ones(len(self.freqs))
        """

        freqs = self.freqs
        if rx_taps is None:
            rx_taps = self.rx_taps
        if dfe_taps is None:
            dfe_taps = self.dfe_taps

        assert len(H21) == len(freqs), ValueError(
            f"Length of `H21` ({len(H21)}) must match length of frequency vector ({len(freqs)})!")

        Htx  = self.Htx(tx_ix)
        Hr   = self.Hr
        if Hctf is None:
            Hctf = self.calc_Hctf(self.gDC, self.gDC2)
        rslt = Htx * self.Ht * H21 * Hr * Hctf  # (178A-14): includes Ht, the Tx rise time filter (93A-46).
        nRxTaps = len(rx_taps)
        if nRxTaps:
            Hrx = self.Hffe_Rx(rx_taps)
            if passive_RxFFE:
                Hrx /= max(np.abs(Hrx))
            rslt *= Hrx

        return rslt

    def pulse_resp(self, H: Cvec) -> Rvec:
        """
        Return the unit pulse response, p(t), corresponding to the given
        voltage transfer function, H(f), according to (93A-24).

        Args:
            H: The voltage transfer function, H(f).
                Note: Possitive frequency components only, including fN.

        Returns:
            The pulse response corresponding to the given voltage transfer function.

        Raises:
            ValueError: If the length of the given voltage transfer
                function differs from that of the system frequency vector.

        Notes:
            1. The pulse response is trimmed to match the length of the system time vector.
        """

        assert len(H) == len(self.freqs), ValueError(
            f"Length of given H(f) {len(H)} does not match length of f {len(self.freqs)}!")

        p = np.fft.irfft(self.Xsinc * H)
        return p[:len(self.times)]

    def gen_pulse_resps(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        chnls: Optional[list[COMChnl]] = None,
        Hctf: Optional[Cvec] = None,
        tx_ix: Optional[int] = None, rx_taps: Optional[Rvec] = None,
        dfe_taps: Optional[Rvec] = None, apply_eq: bool = True
    ) -> list[Rvec]:
        """
        Generate pulse responses for all networks.

        Keyword Args:
            chnls: The list of networks to generate pulse responses for.
                Default: None (i.e. - Use ``self.chnls``.)
            Hctf: Complex voltage transfer function of CTLE.
                Default: None (i.e. - Calculate, using ``self.gDC`` & ``self.gDC2``.)
            tx_ix: Desired Tx tap weights index.
                Default: None (i.e. - Use ``self.tx_ix``.)
            rx_taps: Desired Rx FFE tap weights.
                Default: None (i.e. - Use ``self.rx_taps``.)
            dfe_taps: Desired Rx DFE tap weights.
                Default: None (i.e. - Use ``self.dfe_taps``.)
            apply_eq: Include linear EQ when True; otherwise, exclude it.
                (Allows for pulse response generation of terminated, but unequalized, channel.)
                Default: True

        Returns:
            list of pulse responses.

        Notes:
            1. Assumes ``self.gDC``, ``self.gDC2``, ``self.tx_taps``, ``self.rx_taps``, and ``self.dfe_taps``
            have been set correctly, if the equivalent function parameters have not been provided.

            2. To generate pulse responses that include all linear EQ except the Rx FFE/DFE
            (i.e. - pulse responses suitable for Rx FFE/DFE tap weight optimization,
            via either ``optimize.przf()`` or ``optimize.mmse()``),
            set ``rx_taps`` equal to: ``[1.0]`` and ``dfe_taps`` equal to: ``[]``.
        """

        if chnls is None:
            chnls = self.chnls
        if tx_ix is None:
            tx_ix = self.tx_ix
        if rx_taps is None:
            rx_taps = self.rx_taps
        if dfe_taps is None:
            dfe_taps = self.dfe_taps
        if Hctf is None:
            Hctf = self.calc_Hctf(self.gDC, self.gDC2)

        pulse_resps = []
        for (_, ntype), H21 in chnls:
            if apply_eq:
                if ntype == 'NEXT':
                    tx_ix = 0
                pr = self.pulse_resp(self.H(H21, tx_ix, Hctf=Hctf, rx_taps=rx_taps, dfe_taps=dfe_taps))
            else:
                pr = self.pulse_resp(H21)

            if ntype == 'THRU':
                pr *= self.com_params.A_v
            elif ntype == 'NEXT':
                pr *= self.com_params.A_ne
            else:
                pr *= self.com_params.A_fe

            pulse_resps.append(pr)

        return pulse_resps

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
    def calc_fom(
        self,
        tx_ix: int,
        Hctf: Cvec,
        rx_taps: Optional[Rvec] = None,
        opt_mode: Optional[OptMode] = None,
        norm_mode: Optional[NormMode] = None,
        unit_amp: Optional[bool] = None,
        n_pre: Optional[int] = None,
        ts_window: int = 0,
        ts_center: int = 0
    ) -> float:
        """
        Calculate the **figure of merit** (FOM), given the existing linear EQ settings.
        Optimize Rx FFE taps if they aren't specified by caller.

        Args:
            tx_ix: Index into the list of Tx tap weight combinations.
            Hctf: Complex voltage transfer function of CTLE.

        Keyword Args:
            rx_taps: Rx FFE tap weight overrides.
                Default: None (i.e. - Optimize Rx FFE tap weights.)
            opt_mode: Optimization mode.
                Default: None (i.e. - Use ``self.opt_mode``.)
            norm_mode: The tap weight normalization mode to use.
                Default: None (i.e. - Use ``self.norm_mode``.)
            unit_amp: Enforce unit pulse response amplitude when True.
                (For comparing ``optimize.przf()`` results to ``optimize.mmse()`` results.)
                Default: None (i.e. - Use ``self.unit_amp``.)
            n_pre: Number of pre-cursor taps to use in calculating ISI.
                Default: None (i.e. - Use ``self.nRxPreTaps``.)
            ts_window: MMSE cursor timing search window (vector samples); 0 = full search. (See ``optimize.mmse()``.)
                Default: 0
            ts_center: MMSE cursor timing search starting point, relative to the pulse peak (vector samples).
                Default: 0

        Returns:
            The resultant figure of merit.

        Notes:
            1. See: IEEE 802.3-2022 93A.1.6.

            2. Unlike other member functions of the ``COM`` class,
               this function *optimizes* the Rx FFE tap weights when they are not provided.
        """

        # Honor any mode overrides.
        opt_mode  = opt_mode  or self.opt_mode
        norm_mode = norm_mode or self.norm_mode
        if unit_amp is None:
            unit_amp = self.unit_amp

        # Copy instance variables.
        L = self.com_params.L
        M = self.nspui
        times = self.times
        freqs = self.freqs
        nDFE = len(self.com_params.dfe_min)
        nRxTaps = self.nRxTaps
        nRxPreTaps = self.nRxPreTaps
        rx_taps_min = self.com_params.rx_taps_min
        rx_taps_max = self.com_params.rx_taps_max
        bmin = self.com_params.dfe_min
        bmax = self.com_params.dfe_max
        tb = 1 / self.fb

        # Resolve remaining optionals.
        n_pre = n_pre or nRxPreTaps

        pulse_resps_preFFE = self.gen_pulse_resps(  # Assumes no Rx FFE/DFE.
            tx_ix=tx_ix, Hctf=Hctf,
            rx_taps=self.null_rx_ffe, dfe_taps=self.empty_array)
        match opt_mode:
            case OptMode.PRZF:
                # Step a - Pulse response construction.
                pulse_resps = pulse_resps_preFFE
                if nRxTaps:  # If we have an Rx FFE...
                    if rx_taps is None:        # If we received no explicit override of the Rx FFE
                        rx_taps, _, _ = przf(  # tap weight values,then optimize them.
                            pulse_resps_preFFE[0], M, nRxTaps, nRxPreTaps, nDFE,
                            rx_taps_min, rx_taps_max, bmin, bmax,
                            norm_mode=norm_mode, unit_amp=unit_amp)
                    pulse_resps = self.gen_pulse_resps(           # Regenerate the pulse responses,
                        tx_ix=tx_ix, Hctf=Hctf, rx_taps=rx_taps,  # including the Rx FFE.
                        dfe_taps=self.empty_array)
                else:        # Otherwise, pass the signal through unaltered.
                    rx_taps = self.null_rx_ffe

                # Step b - Cursor identification.
                vic_pulse_resp = pulse_resps[0]  # Note: Includes any Rx FFE, but not DFE.
                vic_peak_loc = np.argmax(vic_pulse_resp)
                cursor_ix = loc_curs(vic_pulse_resp, M,
                                     self.com_params.dfe_max, self.com_params.dfe_min)
                curs_uis, curs_ofst = divmod(cursor_ix, M)
                pr_samps = vic_pulse_resp[curs_ofst::M]
                if n_pre > curs_uis:
                    pr_samps = np.pad(pr_samps, (n_pre - curs_uis, 0))
                else:
                    pr_samps = pr_samps[curs_uis - n_pre:]
                # At this point, `pr_samps` contains one sample per UI w/ exactly `n_pre` pre-cursor samples.

                # Step c - As.
                vic_curs_val = pr_samps[n_pre]
                As = self.com_params.RLM * vic_curs_val / (L - 1)

                # Step d - Tx noise.
                varX = (L**2 - 1) / (3 * (L - 1)**2)                                                    # (93A-29)
                varTx = vic_curs_val**2 * pow(10, -self.com_params.SNR_TX / 10)                         # (93A-30)

                # Step e - ISI.
                hISI = pr_samps.copy()
                hISI[n_pre] = 0  # No ISI at cursor.
                dfe_slice = slice(n_pre + 1, n_pre + 1 + nDFE)
                dfe_tap_weights = np.maximum(                                                           # (93A-26)
                    self.com_params.dfe_min,
                    np.minimum(
                        self.com_params.dfe_max,
                        (hISI[dfe_slice] / vic_curs_val)))
                hISI[dfe_slice] -= dfe_tap_weights * vic_curs_val                                       # (93A-27)
                varISI = varX * (hISI**2).sum()                                                         # (93A-31)

                # Step f - Jitter noise.
                hJ = calc_hJ(vic_pulse_resp, cursor_ix, M)
                varJ = (self.com_params.A_DD**2 + self.com_params.sigma_Rj**2) * varX * (hJ**2).sum()   # (93A-32)

                # Step g - Crosstalk.
                varXT = 0.
                for pulse_resp in pulse_resps[1:]:                                                      # (93A-34)
                    varXT += max((filt_pr_samps(pulse_resp[m::M], As)**2).sum() for m in range(M))  # (93A-33)
                varXT *= varX

                # Step h - Spectral noise.
                df = freqs[1]
                varN = (self.com_params.eta_0 / 1e9) * (abs(self.Hr * Hctf)**2).sum() * df              # (93A-35)

                # Step i - FOM calculation.
                fom = 10 * np.log10(As**2 / (varTx + varISI + varJ + varXT + varN))                     # (93A-36)

            case OptMode.MMSE:
                theNoiseCalc = NoiseCalc(
                    L, tb, 0, times, pulse_resps_preFFE[0], pulse_resps_preFFE[1:],
                    freqs, self.Ht, self.chnls[0][1], self.Hr, Hctf,
                    self.com_params.eta_0, self.com_params.A_v, self.com_params.SNR_TX,
                    self.com_params.A_DD, self.com_params.sigma_Rj)
                rslt = mmse(theNoiseCalc, nRxTaps, nRxPreTaps, len(self.com_params.dfe_min), self.com_params.RLM,
                            self.com_params.L, bmin, bmax, rx_taps_min, rx_taps_max,
                            norm_mode=norm_mode, ts_window=ts_window, ts_center=ts_center)
                fom = rslt["fom"]
                rx_taps = rslt["rx_taps"]
                dfe_tap_weights = rslt["dfe_tap_weights"]
                pr_samps = rslt["h"]
                vic_pulse_resp = rslt["vic_pulse_resp"]  # Note: Does not include Rx FFE/DFE!
                vic_peak_loc = np.argmax(vic_pulse_resp)
                cursor_ix = rslt["cursor_ix"]
                As = vic_pulse_resp[cursor_ix]
                varTx = rslt["varTx"]
                varISI = rslt["varISI"]
                varJ = rslt["varJ"]
                varXT = rslt["varXT"]
                varN = rslt["varN"]
                self.fom_rslts['mse'] = rslt.get('mse')
                self.theNoiseCalc = theNoiseCalc  # pylint: disable=attribute-defined-outside-init
                self.mmse_rslt = rslt             # pylint: disable=attribute-defined-outside-init
            case _:
                raise ValueError(f"Unrecognized optimization mode: {opt_mode}, requested!")

        # Stash our calculation results.
        self.fom_rslts['pulse_resps'] = pulse_resps_preFFE
        self.fom_rslts['vic_pulse_resp'] = vic_pulse_resp
        self.fom_rslts['vic_peak_loc'] = vic_peak_loc
        self.fom_rslts['cursor_ix'] = cursor_ix
        self.fom_rslts['As'] = As
        self.fom_rslts['varTx'] = varTx
        self.fom_rslts['varISI'] = varISI
        self.fom_rslts['varJ'] = varJ
        self.fom_rslts['varXT'] = varXT
        self.fom_rslts['varN'] = varN
        self.fom_rslts['rx_taps'] = rx_taps
        self.fom_rslts['dfe_tap_weights'] = dfe_tap_weights
        self.fom_rslts['pr_samps'] = pr_samps

        return fom

    def opt_eq(  # pylint: disable=too-many-locals,too-many-branches
        self,
        opt_mode: Optional[OptMode] = None,
        norm_mode: Optional[NormMode] = None,
        unit_amp: Optional[bool] = None
    ) -> bool:
        """
        Find and set the optimum values for the linear equalization parameters:
        ``c[n]``, ``gDC``, ``gDC2``, and ``w[n]`` as per IEEE 802.3-22 93A.1.6
        (or, [1] slide 11 if MMSE has been chosen).

        Keyword Args:
            opt_mode: Optimization mode.
                Default: None (i.e. - Use ``self.opt_mode``.)
            norm_mode: The tap weight normalization mode to use.
                Default: None (i.e. - Use ``self.norm_mode``.)
            unit_amp: Enforce unit pulse response amplitude when True.
                (For comparing ``przf()`` results to ``mmse()`` results.)
                Default: None (i.e. - Use ``self.unit_amp``.)

        Returns:
            True if no errors encountered; False otherwise.

        Notes:
            1. The found optimum equalization values are set for the instance.

            2. The search visits candidates in MATLAB COM's order: ``gDC`` (outermost), ``gDC2``,
            then the Tx FFE grid (first tap varying slowest).

            3. With ``com_params.local_search`` (MATLAB COM's "Local Search") > 0, candidates
            too far from the best so far are skipped, by MATLAB COM's ``OptFom_Local_Search()`` rule:
            a swept Tx tap whose grid index is more than ``local_search`` steps from the best's,
            once the loop just outside it (the preceding tap, or ``gDC2`` for the first) is past its first value;
            and a ``gDC2`` more than ``local_search`` steps from the best's, once ``gDC`` is past its first value.
            This is a speed-up heuristic, which can miss the global optimum.

            4. With ``com_params.ts_srch_mode`` == "middle", each candidate's cursor timing search
            works outward from the pulse peak, as MATLAB COM's (see ``optimize.mmse()``);
            with ``com_params.ts_track`` also, from the best cursor offset so far, after a full search for the first.
        """

        # Honor any mode overrides.
        opt_mode  = opt_mode  or self.opt_mode
        norm_mode = norm_mode or self.norm_mode
        if unit_amp is None:  # Don't try the more Pythonic syntax above; it doesn't work for Boolean values.
            unit_amp = self.unit_amp

        cp = self.com_params
        local_search = cp.local_search
        middle = cp.ts_srch_mode == "middle"

        # Tx FFE candidates, in grid order. Index 0 ("no Tx EQ") is tried on its own only if it's not in the grid.
        tx_ixs = list(range(1, self.num_tx_combs))
        if not any(np.allclose(self._tx_combs[ix], 0) for ix in tx_ixs):
            tx_ixs.insert(0, 0)
        # Swept taps, longest grid first (the order in which MATLAB checks them; it doesn't change the outcome).
        swept_taps = sorted((k for k, n in enumerate(self._tx_grid_lens) if n > 1),
                            key=lambda k: -self._tx_grid_lens[k])

        def skip(i_gDC: int, i_gDC2: int, tx_ix: int, best: Optional[dict[str, Any]]) -> bool:
            "MATLAB COM's ``OptFom_Local_Search()`` rule."
            if not local_search or best is None:
                return False
            ixs, best_ixs = self._tx_grid_ixs[tx_ix], best['tx_grid_ixs']
            if ixs is not None and best_ixs is not None:
                for k in swept_taps:
                    outer_ix = i_gDC2 if k == 0 else ixs[k - 1]  # index of the loop just outside tap `k`'s
                    if outer_ix > 0 and abs(ixs[k] - best_ixs[k]) > local_search:
                        return True
            return i_gDC > 0 and abs(i_gDC2 - best['i_gDC2']) > local_search

        # Run the nested optimization loops, keeping the best candidate's results.
        fom_max = -1000.0
        best: Optional[dict[str, Any]] = None
        self.n_fom_evals = 0  # pylint: disable=attribute-defined-outside-init
        for i_gDC, _gDC in enumerate(cp.g_DC):
            for i_gDC2, _gDC2 in enumerate(cp.g_DC2):
                _Hctle: Optional[Cvec] = None
                for _tx_ix in tx_ixs:
                    if skip(i_gDC, i_gDC2, _tx_ix, best):
                        continue
                    if _Hctle is None:
                        _Hctle = self.calc_Hctf(_gDC, _gDC2)
                    ts_window = local_search if middle else 0
                    ts_center = 0
                    if middle and cp.ts_track:
                        if best is None:
                            ts_window = 0
                        else:
                            ts_center = int(best['fom_rslts']['cursor_ix'] - best['fom_rslts']['vic_peak_loc'])
                    fom = self.calc_fom(
                        _tx_ix, _Hctle,
                        opt_mode=opt_mode, norm_mode=norm_mode, unit_amp=unit_amp,
                        ts_window=ts_window, ts_center=ts_center)
                    self.n_fom_evals += 1
                    if fom > fom_max:
                        fom_max = fom
                        rslts = self.fom_rslts
                        best = {
                            'gDC2': _gDC2, 'gDC': _gDC, 'tx_ix': _tx_ix,
                            'i_gDC2': i_gDC2, 'tx_grid_ixs': self._tx_grid_ixs[_tx_ix],
                            **{k: rslts[k] for k in ('rx_taps', 'pr_samps', 'dfe_tap_weights', 'cursor_ix', 'As',
                                                     'varTx', 'varISI', 'varJ', 'varXT', 'varN', 'vic_pulse_resp')},
                            'mse': rslts.get('mse', 0),
                            # Everything `calc_fom()` leaves behind for this candidate, overwritten by later ones;
                            # notably its MMSE noise model, which `calc_noise()` takes the Tx noise from.
                            'fom_rslts': dict(rslts),
                            'theNoiseCalc': getattr(self, 'theNoiseCalc', None),
                            'mmse_rslt': getattr(self, 'mmse_rslt', None),
                        }

        # Check for error and save the best results.
        if best is None:
            return False  # Flags the caller that the following settings have NOT been made.
        # Note the normalization of the Rx FFE tap weights, to produce a unit amplitude cursor tap.
        # This is not dictated by the spec., but is what the MATLAB code does.
        self.gDC2     = best['gDC2']
        self.gDC      = best['gDC']
        self.tx_ix    = best['tx_ix']
        self.rx_taps  = best['rx_taps'] / best['rx_taps'][self.nRxPreTaps]
        self.dfe_taps = best['dfe_tap_weights']
        self.fom_rslts = best['fom_rslts']
        if best['theNoiseCalc'] is not None:
            self.theNoiseCalc = best['theNoiseCalc']  # pylint: disable=attribute-defined-outside-init
            self.mmse_rslt = best['mmse_rslt']        # pylint: disable=attribute-defined-outside-init
        self.fom_rslts['FOM']            = fom_max
        self.fom_rslts['mse']            = best['mse']
        self.fom_rslts['pr_samps']       = best['pr_samps']
        self.fom_rslts['cursor_ix']      = best['cursor_ix']
        self.fom_rslts['As']             = best['As']
        self.fom_rslts['sigma_ISI']      = np.sqrt(best['varISI'])
        self.fom_rslts['sigma_J']        = np.sqrt(best['varJ'])
        self.fom_rslts['sigma_XT']       = np.sqrt(best['varXT'])
        self.fom_rslts['sigma_Tx']       = np.sqrt(best['varTx'])
        self.fom_rslts['sigma_N']        = np.sqrt(best['varN'])
        self.fom_rslts['vic_pulse_resp'] = best['vic_pulse_resp']
        return True

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
    def calc_noise(
        self,
        cursor_ix: Optional[int] = None,
    ) -> tuple[float, float, int]:
        """
        Calculate the interference and noise for COM.

        Keyword Args:
            cursor_ix: An optional predetermined cursor index,
                to be used instead of our own estimate.
                (In support of MMSE.)
                Default: None

        Returns:
            A triplet containing

                - signal amplitude (V)
                - noise + interference amplitude (V)
                - cursor location within victim pulse response vector

        Notes:
            1. Assumes the following instance variables have been set optimally:

                - ``gDC``
                - ``gDC2``
                - ``tx_ix``
                - ``rx_taps``

                (This assumption is embedded into the ``gen_pulse_resps()`` function.)

            2. Fills in the ``com_results`` dictionary w/ various useful values for debugging.
        """

        # Copy instance variables.
        L = self.com_params.L
        M = self.nspui
        RLM = self.com_params.RLM
        freqs = self.freqs
        nDFE = len(self.com_params.dfe_min)
        nRxTaps = self.nRxTaps

        self.set_status("Calculating COM...")
        pulse_resps = self.gen_pulse_resps(dfe_taps=self.empty_array)  # DFE taps are included explicitly, below.
        vic_pulse_resp = pulse_resps[0]
        if cursor_ix is None:
            cursor_ix = loc_curs(vic_pulse_resp, self.nspui, self.com_params.dfe_max, self.com_params.dfe_min)
        curs_uis, curs_ofst = divmod(cursor_ix, M)
        vic_curs_val = vic_pulse_resp[cursor_ix]
        # Missing `2*` is also missing from `Ani` definition; so, they cancel in COM calc.
        As = RLM * vic_curs_val / (L - 1)

        # Sec. 93A.1.7.2
        varX = (L**2 - 1) / (3 * (L - 1)**2)                                        # (93A-29)
        df = freqs[1] - freqs[0]
        if nRxTaps:
            Hrx = self.Hffe_Rx()
        else:
            Hrx = np.ones(len(freqs))
        varN = self.com_params.eta_0 * (
            abs(self.Hr[1:] * self.Hctf[1:] * Hrx[1:])**2).sum() * (df / 1e9)       # (93A-35) + Hffe
        if self.opt_mode == OptMode.PRZF:
            varTx = vic_curs_val**2 * pow(10, -self.com_params.SNR_TX / 10)         # (93A-30)
        else:
            Stx = self.theNoiseCalc.Stn(Hrx=Hrx)
            varTx = self.theNoiseCalc.variance(Stx)                                 # (178A-17)
        hJ = calc_hJ(vic_pulse_resp, cursor_ix, self.nspui)
        hDD = filt_pr_samps(self.com_params.A_DD * hJ, As)
        varJ = self.com_params.sigma_Rj**2 * varX * (hJ**2).sum()                   # (93A-31)
        varG = varTx + varJ + varN                                                  # (93A-41)

        # Sec. 93A.1.7.3
        self.set_status("Sec. 93A.1.7.3")
        # - ISI: every sample at the cursor's phase, pre-cursors included, as per (178A-42) and MATLAB COM's `get_pdf()`.
        n_pre = curs_uis
        isi_sample_slice = slice(curs_ofst, len(vic_pulse_resp), M)
        tISI = self.times[isi_sample_slice]
        hISI = vic_pulse_resp[isi_sample_slice].copy()
        hISI[n_pre] = 0  # No ISI at cursor.
        dfe_slice = slice(n_pre + 1, n_pre + 1 + nDFE)
        dfe_tap_weights = np.maximum(                                               # (93A-26)
            self.com_params.dfe_min,
            np.minimum(
                self.com_params.dfe_max,
                (hISI[dfe_slice] / vic_curs_val)))
        hISI[dfe_slice] -= dfe_tap_weights * vic_curs_val                                       # (93A-27)
        varISI = varX * (hISI**2).sum()                                                         # (93A-31)
        hISI_filt = filt_pr_samps(hISI, As)
        # - Crosstalk: each aggressor's samples at its own worst phase. Under MMSE, that's the phase chosen at the
        #   Rx FFE input (178A-20), as in MATLAB COM. (The Rx FFE's taps are whole UIs apart; so, it keeps phases.)
        hXTs = []
        xt_phases = []
        for k, pulse_resp in enumerate(pulse_resps[1:]):                                        # (93A-44)
            if self.opt_mode == OptMode.MMSE and nRxTaps:
                i = self.theNoiseCalc.agg_phase(self.theNoiseCalc.agg_pulse_resps[k])           # (178A-20)
            else:
                i = np.argmax([(pulse_resp[m::M]**2).sum() for m in range(M)])                  # (93A-33)
            xt_phases.append(int(i))
            hXTs.append(filt_pr_samps(pulse_resp[i::M], As))

        # Voltage grid, as MATLAB COM: bin size As/1000, but no larger than 10 uV (its default "PDF bin size"),
        # spanning the full support of every distribution below (a sum of |samples| for each delta-PMF, plus
        # Gaussian tails), so none is clipped and, so, `Ani` isn't capped (e.g. at 1.1 * As, i.e. COM >= -0.83 dB).
        ystep = min(As / 1000, 1e-5)
        support = abs(hISI_filt).sum() + sum(abs(h).sum() for h in hXTs) + abs(hDD).sum()
        ymax = max(1.1 * As, support + 10 * np.sqrt(varG) + 2 * ystep)
        npts = 2 * int(np.ceil(ymax / ystep)) + 1  # MUST BE ODD!
        y = (np.arange(npts) - npts // 2) * ystep
        pDelta = np.zeros(npts)
        pDelta[npts // 2] = 1.0

        _, pJ = delta_pmf(hDD, L=L, y=y)                                                       # (93A-40)
        pG = np.exp(-y**2 / (2 * varG)) / np.sqrt(TWOPI * varG) * ystep             # (93A-42), but converted to PMF.
        pN = _conv_pmfs(pG, pJ)                                                     # (93A-43)
        _, pISI = delta_pmf(hISI_filt, L=L, y=y)                                                # (93A-40)
        pXT = pDelta
        for hXT in hXTs:
            _, pk = delta_pmf(hXT, L=L, y=y)
            pXT = _conv_pmfs(pXT, pk)
        varXT = sum(_y**2 * p for (_y, p) in zip(y, pXT))

        # Final calculation
        py = _conv_pmfs(_conv_pmfs(pISI, pN), pXT)                                                # (93A-45)
        Py = np.cumsum(py)
        Py /= Py[-1]  # Enforce cumulative probability distribution.
        Ani = -y[np.where(Py >= self.com_params.DER_0)[0][0]]

        # Store some results.
        self.com_rslts['As']          = As
        self.com_rslts['Ani']         = Ani
        self.com_rslts['pulse_resps'] = pulse_resps
        self.com_rslts['cursor_ix']   = cursor_ix
        self.com_rslts['sigma_Tx']    = np.sqrt(varTx)
        self.com_rslts['sigma_G']     = np.sqrt(varG)
        self.com_rslts['sigma_N']     = np.sqrt(varN)
        self.com_rslts['sigma_J']     = np.sqrt(varJ)
        self.com_rslts['sigma_ISI']   = np.sqrt(varISI)
        self.com_rslts['sigma_XT']    = np.sqrt(varXT)
        self.com_rslts['tISI']        = tISI
        self.com_rslts['hISI']        = hISI
        self.com_rslts['pG']          = pG
        self.com_rslts['pN']          = pN
        self.com_rslts['pJ']          = pJ
        self.com_rslts['pISI']        = pISI
        self.com_rslts['pXT']         = pXT
        self.com_rslts['py']          = py
        self.com_rslts['Py']          = Py
        self.com_rslts['y']           = y
        self.com_rslts['dfe_taps']    = dfe_tap_weights
        self.com_rslts['xt_phases']   = xt_phases

        return (As, Ani, cursor_ix)

    about_str = """
      <H2><em>PyChOpMarg</em> - A Python implementation of COM, as per IEEE 802.3-22 Annex 93A/178A.</H2>\n
      <strong>By:</strong> David Banas <capn.freako@gmail.com><p>\n
      <strong>On:</strong> January 20, 2025<p>\n
      <strong>At:</strong> v3.1.0\n
      <H3>Useful Links</H3>\n
      (You'll probably need to: right click, select <em>Copy link address</em>, and paste into your browser.)
        <UL>\n
          <LI><a href="https://github.com/capn-freako/PyChOpMarg"><em>GitHub</em> Home</a>
          <LI><a href="https://pypi.org/project/PyChOpMarg/"><em>PyPi</em> Home</a>
          <LI><a href="https://readthedocs.org/projects/pychopmarg/"><em>Read the Docs</em> Home</a>
        </UL>
    """


def run_com(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    chnl_sets: list[tuple[ChnlGrpName, ChnlSet]],
    com_params: COMParams,
    opt_mode: OptMode = OptMode.MMSE,
    norm_mode: NormMode = NormMode.P8023dj,
    unit_amp: bool = True,
    dbg_dict: Optional[dict[str, Any]] = None
) -> dict[ChnlGrpName, dict[ChnlSetName, COM]]:
    """
    Run COM on a list of grouped channel sets.

    Args:
        chnl_sets: List of pairs, each consisting of:
            - ch_grp_name: The group name for this list of channel sets.
            - ch_sets: List of channel sets to run.
        params: The COM configuration parameters to use.

    Keyword Args:
        opt_mode: The optimization mode desired.
            Default: OptMode.MMSE
        norm_mode: The normalization mode desired for Rx FFE tap weights.
            Default: NormMode.P8023dj
        dbg_dict: Optional dictionary into which debugging values may be stashed,
            for later analysis.
            Default: None

    Returns:
        2D dictionary indexed by channel group name, then by channel set name,
        containing the completed COM objects.
    """

    theCOMs: dict[ChnlGrpName, dict[ChnlSetName, COM]] = {}
    for grp, ch_set in chnl_sets:
        lbl = ch_set['THRU'][0].stem
        print(f"{grp} : {lbl}")
        if dbg_dict is not None:
            theCOM = COM(com_params, ch_set, debug=True)
        else:
            theCOM = COM(com_params, ch_set)

        # Calling the object calculates the COM value, as well as many other intermediate results.
        theCOM(opt_mode=opt_mode, norm_mode=norm_mode, unit_amp=unit_amp, dbg_dict=dbg_dict)

        if grp in theCOMs:
            theCOMs[grp].update({ch_set['THRU'][0].stem: theCOM})
        else:
            theCOMs.update({grp: {ch_set['THRU'][0].stem: theCOM}})

    return theCOMs


if __name__ == "__main__":
    # from pychopmarg.cli import cli
    # cli()
    print("Sorry, the PyChOpMarg package is currently only usable as a library.")
    print("It's GUI is currently broken.")
