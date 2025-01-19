#!/usr/bin/env python

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

from functools import cache
from pathlib import Path
from typing  import Any, Dict, Optional

import numpy as np  # type: ignore
import skrf  as rf  # type: ignore
from numpy             import array, arange

from pychopmarg.common import (
    PI, TWOPI, Cvec, Rvec, Cmat, OptMode,
    COMChnl, COMNtwk, ChnlSet, ChnlGrpName, ChnlSetName)
from pychopmarg.config.ieee_8023dj import IEEE_8023dj
from pychopmarg.config.template    import COMParams
from pychopmarg.noise    import NoiseCalc
from pychopmarg.optimize import NormMode, mmse, przf
from pychopmarg.utility  import (
    import_s32p, sdd_21, sDieLadderSegment, sPkgTline, sCshunt, filt_pr_samps,
    delta_pmf, mk_combs, calc_Hffe, calc_Hctle, calc_H21, calc_hJ, loc_curs)


class COM():  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    Encoding of the IEEE 802.3-22 Annex 93A/178A 'Channel Operating Margin' (COM) specification.
    """

    # General
    status_str = str("Ready")
    debug = bool(False)
    opt_mode = OptMode(OptMode.MMSE)
    norm_mode = NormMode(NormMode.P8023dj)
    unit_amp = bool(True)
    com_params = IEEE_8023dj

    # Linear EQ
    tx_ix: int = 0          # Index into list of all possible combinations of Tx FFE tap weights.
    rx_taps: Rvec = array([1.0])
    dfe_taps: Rvec = array([])
    nRxTaps: int = 0
    nRxPreTaps: int = 0     # `dw` from `com_params`
    gDC = 0.0               # Choices are in `com_params.g_DC`.
    gDC2 = 0.0              # Choices are in `com_params.g_DC2`.
    rx_ffe_phase_matrix: Cmat = array([])

    # Channel data
    vic_chnl_ix = int(1)  # Used with s32p file.
    chnls: list[COMChnl] = []
    chnls_noPkg: list[COMChnl] = []
    pulse_resps_nopkg: list[Rvec] = []
    pulse_resps_noeq:  list[Rvec] = []
    cursor_ix: int = 0

    # Package
    zp_sel = 0  # package length selector
    _sPkgTx: list[rf.Network] = []
    _sPkgRx: list[rf.Network] = []

    def __init__(  # noqa=E501 pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
        self,
        com_params: COMParams,
        channels: Path | dict[str, list[Path]],
        vic_chnl_ix: int = 1,
        debug: bool = False,
        do_init: bool = True
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

        Keyword Args:
            vic_chnl_ix: Index (1-based) of victim path in s32p.
                Default: 1
            debug: Gather/report certain debugging information when ``True``.
                Default: ``False``
            do_init: Skip normal initialization behavior when ``False``.
                Default: ``True``
        """

        self.com_params = com_params
        self.debug = debug
        self.do_init = do_init

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
            for fname, chtype in [(chnl_name, chnl_type) for chnl_type in chnl_types
                                                         for chnl_name in channels[chnl_type]]:  # noqa=E127
                ntwks.append((sdd_21(rf.Network(fname)), chtype))
        else:
            raise ValueError(f"`channels` must be of type 'str' or 'dict', not '{type(channels)}'!")

        # Create the system time & frequency vectors.
        fb = com_params.fb * 1e9
        # fstep = com_params.fstep * 1e9
        fstep = 50e6               # Dramatically improves performance.
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
        z_pairs = list(zip(com_params.z_c, [com_params.z_p[self.zp_sel], com_params.z_pB]))
        sPkgTx_SE = rf.network.cascade_list([
            self.sDie(False),
            sPkgTline(self.freqs, self.com_params.R_0, self.com_params.a1, self.com_params.a2,
                      self.com_params.tau, self.com_params.gamma0, z_pairs),
            self.sC(self.com_params.C_p[0] / 1e9)])
        self._sPkgTx = [sdd_21(rf.network.concat_ports([sPkgTx_SE, sPkgTx_SE], port_order='first'))]
        sPkgRx_SE = rf.network.cascade_list([
            self.sC(self.com_params.C_p[1] / 1e9),
            sPkgTline(self.freqs, self.com_params.R_0, self.com_params.a1, self.com_params.a2,
                      self.com_params.tau, self.com_params.gamma0, z_pairs),
            self.sDie(True)])
        self._sPkgRx = [sdd_21(rf.network.concat_ports([sPkgRx_SE, sPkgRx_SE], port_order='first'))]

        self.chnls = list(map(self.add_pkg, ntwks))
        if debug:
            self.chnls_noPkg = list(
                map(lambda ntwk: (ntwk, calc_H21(f, ntwk[0], self.gamma1_Tx, self.gamma2_Rx)),
                    [ntwks[0]]))
            self.pulse_resps_nopkg = self.gen_pulse_resps(chnls=[self.chnls_noPkg[0]], apply_eq=False)
            self.pulse_resps_noeq  = self.gen_pulse_resps(chnls=[self.chnls[0]],       apply_eq=False)

        # Generate all possible combinations of Tx FFE tap weights.
        c0_min = com_params.c0_min
        trips = list(zip(com_params.tx_taps_min,
                         com_params.tx_taps_max,
                         com_params.tx_taps_step))
        _tx_combs: list[Rvec] = list(filter(
            lambda v: (1 - abs(v).sum()) >= c0_min,
            mk_combs(trips)))
        self._tx_combs = [np.zeros(len(com_params.tx_taps_min)), *_tx_combs]
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
        do_opt_eq: bool = True,
        tx_taps:   Optional[Rvec]     = None,
        opt_mode:  Optional[OptMode]  = None,
        norm_mode: Optional[NormMode] = None,
        unit_amp:  Optional[bool]     = None,
        dbg_dict:  Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate the COM value.

        Keyword Args:
            do_opt_eq: Perform optimization of linear equalization when True.
                Default: True
            tx_taps: Used when `do_opt_eq` = False.
                Default: None
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
        assert self.opt_eq(do_opt_eq=do_opt_eq, tx_taps=tx_taps), RuntimeError(
            "EQ optimization failed!")
        self.set_status("Calculating noise...")
        As, Ani, self.cursor_ix = self.calc_noise(
            cursor_ix=self.fom_rslts['cursor_ix'])
        com = 20 * np.log10(As / Ani)
        self.com_rslts["COM"] = com
        self.set_status(f"Ready; COM = {com: 5.1f} dB")
        return com

    # Instance Properties
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
        return self._gamma1[1]

    @property
    def gamma2_Rx(self) -> float:
        "Reflection coefficient looking out of the right end of the channel."
        return self._gamma1[1]

    def sDie(self, isRx: bool) -> rf.Network:
        "On-die parasitic capacitance/inductance ladder network, including bump."
        if isRx:
            ix = 1
        else:
            ix = 0
        Cd = self.com_params.C_d[ix] / 1e9
        Ls = self.com_params.L_s[ix] / 1e9
        R0 = [self.com_params.R_0] * len(Cd)  # type: ignore
        rslt = rf.network.cascade_list(
            list(map(lambda trip: sDieLadderSegment(self.freqs, trip),
                     zip(R0, Cd, Ls))))  # type: ignore
        rslt = rslt ** self.sC(self.com_params.C_b[ix] / 1e9)
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
        return self._sPkgTx[0]

    # Package modeling
    def add_pkg(self, ntwk: COMNtwk) -> COMChnl:
        """Add package response to raw channel and pre-calculate H21."""

        # Pre-interpolate channel model to system frequency vector, to prevent noisy non-causal artifacts.
        freqs = self.freqs
        sChnl = ntwk[0]
        sChnlInterp = sChnl.extrapolate_to_dc().interpolate(
            freqs[freqs <= sChnl.f[-1]], kind='cubic', coords='polar', basis='t', assume_sorted=True)
        new_s = sChnlInterp.s.tolist()  # I couldn't make NumPy array padding work correctly here.
        pad_len = len(freqs) - len(sChnlInterp.f)
        new_s.extend(
            [[[sChnlInterp.s11.s[-1, 0, 0], sChnlInterp.s12.s[-1, 0, 0]],
              [sChnlInterp.s21.s[-1, 0, 0], sChnlInterp.s22.s[-1, 0, 0]]]
            ] * pad_len)  # noqa=E124
        new_s = np.array(new_s)
        sChnlInterp = rf.Network(s=new_s, z0=sChnlInterp.z0[0], f=freqs)

        ntype = ntwk[1]
        if ntype == 'NEXT':
            _ntwk = self.sPkgNEXT ** sChnlInterp ** self.sPkgRx
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

    @cache  # pylint: disable=method-cache-max-size-none
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
        return calc_Hffe(self.freqs, 1 / self.fb, array(self._tx_combs[tx_taps_ix]), 3)

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
        rslt = Htx * H21 * Hr * Hctf
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
        n_pre: Optional[int] = None
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
                hJ = calc_hJ(vic_pulse_resp, As, cursor_ix, M)
                varJ = (self.com_params.A_DD**2 + self.com_params.sigma_Rj**2) * varX * (hJ**2).sum()   # (93A-32)

                # Step g - Crosstalk.
                varXT = 0.
                for pulse_resp in pulse_resps[1:]:                                                      # (93A-34)
                    varXT += max((filt_pr_samps(pulse_resp[m::M], As)**2).sum()
                                  for m in range(M))                                                    # noqa=E127,E501 (93A-33)
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
                            norm_mode=norm_mode)
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
                self.fom_rslts['mse'] = rslt['mse'] if 'mse' in rslt else None
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

    def opt_eq(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
        self,
        do_opt_eq: bool = True,
        tx_taps: Optional[Rvec] = None,
        opt_mode: Optional[OptMode] = None,
        norm_mode: Optional[NormMode] = None,
        unit_amp: Optional[bool] = None
    ) -> bool:
        """
        Find and set the optimum values for the linear equalization parameters:
        ``c[n]``, ``gDC``, ``gDC2``, and ``w[n]`` as per IEEE 802.3-22 93A.1.6
        (or, [1] slide 11 if MMSE has been chosen).

        Keyword Args:
            do_opt_eq: Perform optimization of linear EQ when True.
                Default: True
            tx_taps: Used when ``do_opt_eq`` = False.
                Default: None
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
        """

        if do_opt_eq:
            # Honor any mode overrides.
            opt_mode  = opt_mode  or self.opt_mode
            norm_mode = norm_mode or self.norm_mode
            if unit_amp is None:  # Don't try the more Pythonic syntax above; it doesn't work for Boolean values.
                unit_amp = self.unit_amp

            # Run the nested optimization loops.
            fom_max = -1000.0
            fom_max_changed = False
            for _gDC2 in self.com_params.g_DC2:
                for _gDC in self.com_params.g_DC:
                    _Hctle = self.calc_Hctf(_gDC, _gDC2)
                    for _tx_ix in range(self.num_tx_combs):
                        fom = self.calc_fom(
                            _tx_ix, _Hctle,
                            opt_mode=opt_mode, norm_mode=norm_mode, unit_amp=unit_amp)
                        if fom > fom_max:
                            fom_max_changed = True
                            fom_max = fom
                            gDC2_best = _gDC2
                            gDC_best = _gDC
                            tx_ix_best = _tx_ix
                            rx_taps_best = self.fom_rslts['rx_taps']
                            pr_samps_best = self.fom_rslts['pr_samps']
                            dfe_tap_weights_best = self.fom_rslts['dfe_tap_weights']
                            cursor_ix_best = self.fom_rslts['cursor_ix']
                            As_best = self.fom_rslts['As']
                            varTx_best = self.fom_rslts['varTx']
                            varISI_best = self.fom_rslts['varISI']
                            varJ_best = self.fom_rslts['varJ']
                            varXT_best = self.fom_rslts['varXT']
                            varN_best = self.fom_rslts['varN']
                            vic_pulse_resp = self.fom_rslts['vic_pulse_resp']
                            mse_best = self.fom_rslts['mse'] if 'mse' in self.fom_rslts else 0
        else:
            assert tx_taps, RuntimeError("You must define `tx_taps` when setting `do_opt_eq` False!")
            fom = self.calc_fom(0, self.calc_Hctf(self.gDC, self.gDC2))
            fom_max = fom
            fom_max_changed = True
            gDC2_best = self.gDC2
            gDC_best = self.gDC
            tx_ix_best = 0
            rx_taps_best = self.rx_taps
            dfe_tap_weights_best = self.fom_rslts['dfe_tap_weights']
            cursor_ix_best = self.fom_rslts['cursor_ix']
            As_best = self.fom_rslts['As']
            varTx_best = self.fom_rslts['varTx']
            varISI_best = self.fom_rslts['varISI']
            varJ_best = self.fom_rslts['varJ']
            varXT_best = self.fom_rslts['varXT']
            varN_best = self.fom_rslts['varN']
            vic_pulse_resp = self.fom_rslts['vic_pulse_resp']
            mse_best = 0

        # Check for error and save the best results.
        if not fom_max_changed:
            return False  # Flags the caller that the following settings have NOT been made.
        # Note the normalization of the Rx FFE tap weights, to produce a unit amplitude cursor tap.
        # This is not dictated by the spec., but is what the MATLAB code does.
        self.gDC2     = gDC2_best                                   # pylint: disable=possibly-used-before-assignment
        self.gDC      = gDC_best                                    # pylint: disable=possibly-used-before-assignment
        self.tx_ix    = tx_ix_best                                  # pylint: disable=possibly-used-before-assignment
        self.rx_taps  = rx_taps_best / rx_taps_best[self.nRxPreTaps]  # pylint: disable=possibly-used-before-assignment
        self.dfe_taps = dfe_tap_weights_best                        # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['FOM']            = fom_max                  # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['mse']            = mse_best                 # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['pr_samps']       = pr_samps_best            # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['cursor_ix']      = cursor_ix_best           # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['As']             = As_best                  # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['sigma_ISI']      = np.sqrt(varISI_best)     # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['sigma_J']        = np.sqrt(varJ_best)       # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['sigma_XT']       = np.sqrt(varXT_best)      # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['sigma_Tx']       = np.sqrt(varTx_best)      # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['sigma_N']        = np.sqrt(varN_best)       # pylint: disable=possibly-used-before-assignment
        self.fom_rslts['vic_pulse_resp'] = vic_pulse_resp           # pylint: disable=possibly-used-before-assignment
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
        ymax = 1.1 * As
        npts = 2 * min(int(ymax / 0.00001), 1_000) + 1  # Note 1 of 93A.1.7.1; MUST BE ODD!
        y = np.linspace(-ymax, ymax, npts)
        ystep = 2 * ymax / (npts - 1)
        pDelta = np.zeros(npts)
        pDelta[npts // 2] = 1.0

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
            varTx = sum(Stx) * df                                                   # (178A-17)
        hJ = calc_hJ(vic_pulse_resp, As, cursor_ix, self.nspui)
        _, pJ = delta_pmf(filt_pr_samps(self.com_params.A_DD * hJ, ymax), L=L, y=y)  # (93A-40)
        varJ = self.com_params.sigma_Rj**2 * varX * (hJ**2).sum()                   # (93A-31)
        varG = varTx + varJ + varN                                                  # (93A-41)
        pG = np.exp(-y**2 / (2 * varG)) / np.sqrt(TWOPI * varG) * ystep             # (93A-42), but converted to PMF.
        pN = np.convolve(pG, pJ, mode='same')                                       # (93A-43)
        pN /= pN.sum()  # Enforce a PMF.

        # Sec. 93A.1.7.3
        self.set_status("Sec. 93A.1.7.3")
        # - ISI (Inconsistent w/ IEEE 802.3-22, but consistent w/ v2.60 of MATLAB code.)
        n_pre = min(5, curs_uis)
        # Sample every M points, such that we include our identified cursor sample.
        isi_sample_slice = slice(curs_ofst, len(vic_pulse_resp), M)
        isi_select_slice = slice(curs_uis - n_pre, curs_uis + 2048)
        tISI = self.times[isi_sample_slice][isi_select_slice]
        hISI = vic_pulse_resp[isi_sample_slice][isi_select_slice].copy()
        hISI[n_pre] = 0  # No ISI at cursor.
        dfe_slice = slice(n_pre + 1, n_pre + 1 + nDFE)
        dfe_tap_weights = np.maximum(                                               # (93A-26)
            self.com_params.dfe_min,
            np.minimum(
                self.com_params.dfe_max,
                (hISI[dfe_slice] / vic_curs_val)))
        hISI[dfe_slice] -= dfe_tap_weights * vic_curs_val                                       # (93A-27)
        varISI = varX * (hISI**2).sum()                                                         # (93A-31)
        _, pISI = delta_pmf(filt_pr_samps(hISI, ymax), L=L, y=y)                                # (93A-40)

        # - Crosstalk
        pXT = pDelta
        for pulse_resp in pulse_resps[1:]:                                                      # (93A-44)
            i = np.argmax([(pulse_resp[m::M]**2).sum() for m in range(M)])                      # (93A-33)
            samps = pulse_resp[i::M]  # [isi_select_slice]
            _, pk = delta_pmf(filt_pr_samps(samps, ymax), L=L, y=y)  # , dbg_dict=dbg_dict)
            pXT = np.convolve(pXT, pk, mode='same')
        pXT /= pXT.sum()  # Enforce a PMF.
        varXT = sum(_y**2 * p for (_y, p) in zip(y, pXT))

        # Final calculation
        py = np.convolve(np.convolve(pISI, pN, mode='same'), pXT, mode='same')                  # (93A-45)
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
    dbg_dict: Optional[Dict[str, Any]] = None
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
