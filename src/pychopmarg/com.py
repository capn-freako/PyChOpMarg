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
    2. Straighten out the new 2-segment package model.
"""

from enum    import Enum
from functools import cache
from pathlib import Path
from typing  import Any, Dict, Optional, TypeVar

import numpy as np  # type: ignore
import skrf  as rf  # type: ignore
from numpy             import array, arange
from numpy.typing      import NDArray
from scipy.interpolate import interp1d

from pychopmarg.common   import Rvec, Cvec, PI, TWOPI, COMChnl, COMNtwk
from pychopmarg.config.ieee_8023by import IEEE_8023by
from pychopmarg.config.template import COMParams
from pychopmarg.noise    import NoiseCalc
from pychopmarg.optimize import NormMode, mmse, przf
from pychopmarg.utility  import (
    import_s32p, sdd_21, sDieLadderSegment, sPkgTline, sCshunt, filt_pr_samps,
    delta_pmf, mk_combs, calc_Hffe, calc_Hctle, calc_H21, calc_hJ, loc_curs)


class OptMode(Enum):
    "Linear equalization optimization mode."
    PRZF = 1
    MMSE = 2


class COM():  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    Encoding of the IEEE 802.3-22 Annex 93A/178A 'Channel Operating Margin' (COM) specification.

    ToDo:
        1. Clean up unused attributes.
    """

    # General
    status_str = str("Ready")
    debug = bool(False)
    opt_mode = OptMode(OptMode.MMSE)
    norm_mode = NormMode(NormMode.P8023dj)
    unit_amp = bool(True)
    com_params = IEEE_8023by

    # Linear EQ
    tx_taps: Rvec = array([])
    rx_taps: Rvec = array([1.0])
    dfe_taps: Rvec = array([])
    nRxTaps: int = 0
    nRxPreTaps: int = 0     # `dw` from `com_params`
    gDC = 0.0               # Choices are in `com_params.g_DC`.
    gDC2 = 0.0              # Choices are in `com_params.g_DC2`.

    # Channel data
    vic_chnl_ix = int(1)  # Used with s32p file.
    chnls: list[COMChnl] = []
    chnls_noPkg: list[COMChnl] = []
    pulse_resps_nopkg: list[Rvec] = []
    pulse_resps_noeq:  list[Rvec] = []
    cursor_ix: int = 0

    # Package
    zp_sel = 0  # package length selector

    def __init__(
        self,
        com_params: COMParams,
        channels: Path | dict[str, list[Path]],
        vic_chnl_ix: int = 1,
        debug: bool = False
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
        """

        # Process the given channel file names.
        ntwks: list[COMNtwk] = []
        if isinstance(channels, str):  # Should be a "*.s32p".
            assert channels.endswith(".s32p"), ValueError(
                f"When `channels` is a string it must contain a '*.s32p' value!")
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
                                                         for chnl_name in channels[chnl_type]]:
                ntwks.append((sdd_21(rf.Network(fname)), chtype))
        else:
            raise ValueError(f"`channels` must be of type 'str' or 'dict', not '{type(channels)}'!")

        # Create the system time & frequency vectors.
        fb = com_params.fb * 1e9
        ui = 1 / fb
        fmax = min(map(lambda ch: ch[0].f[-1], ntwks))
        fstep = com_params.fstep * 1e9
        tmax = 1 / fstep           # Just enough to cover one full cycle of the fundamental.
        tstep = ui / com_params.M  # Obeying requested samps. per UI.
        f = arange(0, fmax + fstep, fstep)  # "+ fstep", to include `fmax`.
        t_irfft = array([n * 0.5 / fmax for n in range(2 * (len(f) - 1))])
        _t = arange(0, tmax, tstep)
        t = _t[_t < t_irfft[-1]]  # to avoid interpolation bounds errors
        self.t: Rvec = t
        self.f: Rvec = f
        self._t_irfft: Rvec = t_irfft

        # Pre-calculate constant responses.
        self._Xsinc = int(ui / t_irfft[1]) * np.sinc(ui * f)
        self._Ht = np.exp(-2 * (PI * (f / 1e9) * com_params.T_r / 1.6832)**2)  # 93A-46 calls for f in GHz.
        _f = f / (com_params.f_r * fb)
        self._Hr = 1 / (1 - 3.414214 * _f**2 + _f**4 + 2.613126j * (_f - _f**3))

        self.chnls_noPkg = list(
            map(lambda ntwk: (ntwk, calc_H21(f, ntwk[0], self.gamma1[0], self.gamma2[0])),
                ntwks))
        self.chnls = list(map(self.add_pkg, ntwks))
        self.pulse_resps_nopkg = self.gen_pulse_resps(chnls=self.chnls_noPkg, apply_eq=False)
        self.pulse_resps_noeq = self.gen_pulse_resps(chnls=self.chnls, apply_eq=False)

        # Generate all possible combinations of Tx FFE tap weights.
        c0_min = com_params.c0_min
        trips = list(zip(com_params.tx_taps_min,
                         com_params.tx_taps_max,
                         com_params.tx_taps_step))
        self._tx_combs: list[Rvec] = list(filter(
            lambda v: (1 - abs(v).sum()) >= c0_min,
            mk_combs(trips)))
        self._num_tx_combs = len(self._tx_combs)

        @cache
        def _Htx(tx_combs_ix: int) -> Cvec:
            return calc_Hffe(self.freqs, 1 / self.fb, array(self._tx_combs[tx_combs_ix]), 3)
        self._Htx = _Htx

        # Rx linear EQ
        _f = f / (com_params.f_r * fb)
        self._Hr = 1 / (1 - 3.414214 * _f**2 + _f**4 + 2.613126j * (_f - _f**3))

        # Set Rx FFE quantities.
        self.nRxTaps = len(com_params.rx_taps_max)
        self.nRxPreTaps = com_params.dw

        # Misc.
        self.com_rslts: dict[str, Any] = {}
        self.fom_rslts: dict[str, Any] = {}
        self.dbg_dict: Optional[dict[str, Any]] = None

        self.com_params = com_params
        self.debug = debug

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
        opt_mode  = opt_mode  or self.opt_mode
        norm_mode = norm_mode or self.norm_mode
        if unit_amp is None:
            unit_amp = self.unit_amp
        self.opt_mode  = opt_mode
        self.norm_mode = norm_mode
        self.unit_amp  = unit_amp
        if dbg_dict is not None:
            self.dbg_dict = dbg_dict

        self.set_status("Optimizing EQ...")
        assert self.opt_eq(do_opt_eq=do_opt_eq, tx_taps=tx_taps), RuntimeError(
            "EQ optimization failed!")
        self.set_status("Calculating noise...")
        As, Ani, self.cursor_ix = self.calc_noise(dbg_dict=dbg_dict)
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
    def t_irfft(self) -> Rvec:
        "`irfft()` result time index (s) (i.e. - time vector coupled to frequency vector)."
        return self._t_irfft

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
        gDC = gDC or self.gDC
        gDC2 = gDC2 or self.gDC2
        return calc_Hctle(self.freqs, self.com_params.f_z, self.com_params.f_p1,
                          self.com_params.f_p2, self.com_params.f_LF, gDC, gDC2)

    @property
    def gamma1(self) -> NDArray:
        "Reflection coefficient looking out of the left end of the channel."
        Rd = self.com_params.R_d
        R0 = self.com_params.R_0
        return (Rd - R0) / (Rd + R0)

    @property
    def gamma2(self) -> NDArray:
        "Reflection coefficient looking out of the right end of the channel."
        return self.gamma1

    @property
    def sDieLadder(self) -> rf.Network:
        "On-die parasitic capacitance/inductance ladder network."
        Cd = list(map(lambda x: x / 1e12, self.com_params.C_d))
        Ls = list(map(lambda x: x / 1e9, self.com_params.L_s))
        # Cd = list(map(lambda x: x / 1e12, self.com_params.C_d))[0]
        # Ls = list(map(lambda x: x / 1e9, self.com_params.L_s))[0]
        R0 = [self.com_params.R_0] * len(Cd)  # type: ignore
        rslt = rf.network.cascade_list(
            list(map(lambda trip: sDieLadderSegment(self.freqs, trip),
                     zip(R0, Cd, Ls))))  # type: ignore
        return rslt

    @property
    def sPkgRx(self) -> rf.Network:
        "Rx package response."
        # return self.sC(self.com_params.C_p / 1e12) ** self.sZp ** self.sDieLadder
        return self.sC(self.com_params.C_p[0] / 1e12) ** self.sZp ** self.sDieLadder  # type: ignore

    @property
    def sPkgTx(self) -> rf.Network:
        "Tx package response."
        # return self.sDieLadder ** self.sZp ** self.sC(self.com_params.C_p / 1e12)
        return self.sDieLadder ** self.sZp ** self.sC(self.com_params.C_p[0] / 1e12)  # type: ignore

    @property
    def sPkgNEXT(self) -> rf.Network:
        "NEXT package response."
        # return self.sDieLadder ** self.sZpNEXT ** self.sC(self.com_params.C_p / 1e12)
        return self.sDieLadder ** self.sZpNEXT ** self.sC(self.com_params.C_p[0] / 1e12)  # type: ignore

    @property
    def sZp(self) -> rf.Network:
        "THRU/FEXT package transmission line."
        return self.calc_sZp()

    @property
    def sZpNEXT(self) -> rf.Network:
        "NEXT package transmission line."
        return self.calc_sZp(NEXT=True)

    def calc_sZp(self, NEXT: bool = False) -> rf.Network:
        """
        Return the 2-port network corresponding to a package transmission line,
        according to (93A-9:14).

        Keyword Args:
            NEXT: Use first package T-line length option when True.
                Default: False

        Returns:
            2-port network equivalent to package transmission line.
        """

        zc = self.com_params.z_c
        assert len(zc) in [1, 2], ValueError(
            f"Length of `zc` ({len(zc)}) must be 1 or 2!")

        if NEXT:
            zp = self.com_params.z_p[0]
        else:
            zp = self.com_params.z_p[self.zp_sel]
        if len(zc) == 1:
            zps = [zp]
        else:
            zps = [zp, self.com_params.z_pB]

        return sPkgTline(self.freqs, self.com_params.R_0, self.com_params.a1, self.com_params.a2,
                         self.com_params.tau, self.com_params.gamma0, list(zip(zc, zps)))

    # Package modeling
    def add_pkg(self, ntwk: tuple[rf.Network, str]) -> COMChnl:
        """Add package response to raw channel."""
        ntype = ntwk[1]
        if ntype == 'NEXT':
            _ntwk = self.sPkgNEXT ** ntwk[0] ** self.sPkgRx
        _ntwk = self.sPkgTx ** ntwk[0] ** self.sPkgRx
        return ((_ntwk, ntype), calc_H21(self.freqs, _ntwk, self.gamma1[0], self.gamma2[0]))

    # Logging / Debugging
    def set_status(self, status: str) -> None:
        "Set the GUI status string and print it if we're debugging."
        self.status_str = status
        if self.debug:
            print(status, flush=True)

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
        return self._Htx(tx_taps_ix)

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
            1. It is at this point in the analysis that the "raw" Touchstone data
            gets interpolated to our system frequency vector.

            2. After this step, the package and R0/Rd mismatch have been accounted for, but not the EQ.
        """
        # return calc_H21(self.freqs, s2p, self.gamma1, self.gamma2)
        return calc_H21(self.freqs, s2p, self.gamma1[0][0], self.gamma2[0][0])

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

        Keyword Args:
            tx_ix: Tx FFE tap weights index.
                Default: None (i.e. - Use ``self.tx_taps``.)
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
        tb    = 1 / self.fb
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
        try:
            nRxTaps = len(rx_taps)
        except:
            print(f"self.rx_taps: {self.rx_taps}")
            raise
        if nRxTaps:
            Hrx  = calc_Hffe(freqs, tb, array(rx_taps).flatten(), nRxTaps - self.nRxPreTaps - 1, hasCurs=True)
            if passive_RxFFE:
                Hrx /= max(abs(Hrx))
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
            1. It is at this point in the signal processing chain that we change
            time domains.
        """

        assert len(H) == len(self.freqs), ValueError(
            f"Length of given H(f) {len(H)} does not match length of f {len(self.freqs)}!")

        p = np.fft.irfft(self.Xsinc * H)
        spln = interp1d(self.t_irfft, p)  # `p` is not yet in our system time domain!
        try:
            rslt = spln(self.times)       # Now, it is.
        except:
            print(f"max(self.times): {max(self.times)}")
            print(f"max(self.t_irfft): {max(self.t_irfft)}")
            raise

        return rslt

    def gen_pulse_resps(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, chnls: Optional[list[COMChnl]] = None,
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
                Default: None (i.e. - Use ``self.tx_taps``.)
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
            tx_ix = 0
        if rx_taps is None:
            rx_taps = self.rx_taps
        if dfe_taps is None:
            dfe_taps = self.dfe_taps
        if Hctf is None:
            Hctf = self.calc_Hctf(self.gDC, self.gDC2)

        # tx_taps = array(tx_taps)
        rx_taps = array(rx_taps)
        dfe_taps = array(dfe_taps)

        pulse_resps = []
        for (ntwk, ntype), H21 in chnls:
            if apply_eq:
                if ntype == 'NEXT':
                    pr = self.pulse_resp(self.H(
                        # ntwk, np.zeros(tx_taps.shape), gDC=gDC, gDC2=gDC2, rx_taps=rx_taps, dfe_taps=dfe_taps))
                        H21, 0, Hctf=Hctf, rx_taps=rx_taps, dfe_taps=dfe_taps))
                else:
                    pr = self.pulse_resp(self.H(
                        H21, tx_ix, Hctf=Hctf, rx_taps=rx_taps, dfe_taps=dfe_taps))
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
        unit_amp: Optional[bool] = None
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
        if rx_taps is None:
            rx_taps=array([1.0])  # Keeps `self.rx_taps` from getting set to None.

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
        nspui = self.nspui

        pulse_resps_preFFE = self.gen_pulse_resps(  # Assumes no Rx FFE/DFE.
            tx_ix=tx_ix, Hctf=Hctf, rx_taps=array([1.0]), dfe_taps=array([]))
        match opt_mode:
            case OptMode.PRZF:
                # Step a - Pulse response construction.
                pulse_resps = pulse_resps_preFFE
                pr_samps = None
                if nRxTaps:              # If we have an Rx FFE...
                    if rx_taps is None:  # If we received no explicit override of the Rx FFE tap weight values,
                        rx_taps, _, pr_samps = przf(  # then optimize them.
                            pulse_resps_preFFE[0], M, nRxTaps, nRxPreTaps, nDFE,
                            array(rx_taps_min), array(rx_taps_max), array(bmin), array(bmax),
                            norm_mode=norm_mode, unit_amp=unit_amp)
                    pulse_resps = self.gen_pulse_resps(
                        tx_ix=tx_ix, Hctf=Hctf, rx_taps=array(rx_taps), dfe_taps=array([]))

                # Step b - Cursor identification.
                vic_pulse_resp = array(pulse_resps[0])  # Note: Includes any Rx FFE, but not DFE.
                vic_peak_loc = np.argmax(vic_pulse_resp)
                cursor_ix = loc_curs(vic_pulse_resp, self.nspui,
                                     array(self.com_params.dfe_max), array(self.com_params.dfe_min))
                if pr_samps is None:
                    pr_samps = vic_pulse_resp[cursor_ix % nspui::nspui]

                # Step c - As.
                vic_curs_val = vic_pulse_resp[cursor_ix]
                As = self.com_params.RLM * vic_curs_val / (L - 1)

                # Step d - Tx noise.
                varX = (L**2 - 1) / (3 * (L - 1)**2)  # (93A-29)
                varTx = vic_curs_val**2 * pow(10, -self.com_params.SNR_TX / 10)  # (93A-30)

                # Step e - ISI.
                # This is not compliant to the standaard, but is consistent w/ v2.60 of MATLAB code.
                n_pre = cursor_ix // M
                first_pre_ix = cursor_ix - n_pre * M
                vic_pulse_resp_isi_samps = np.concatenate((vic_pulse_resp[first_pre_ix:cursor_ix:M],
                                                           vic_pulse_resp[cursor_ix + M::M]))
                vic_pulse_resp_post_samps = vic_pulse_resp_isi_samps[n_pre:]
                dfe_tap_weights = np.maximum(  # (93A-26)
                    self.com_params.dfe_min,
                    np.minimum(
                        self.com_params.dfe_max,
                        (vic_pulse_resp_post_samps[:nDFE] / vic_curs_val)))
                hISI = vic_pulse_resp_isi_samps \
                     - vic_curs_val * np.pad(dfe_tap_weights,  # noqa E127
                                             (n_pre, len(vic_pulse_resp_post_samps) - nDFE),
                                             mode='constant',
                                             constant_values=0)  # (93A-27)
                # varISI = varX * sum(hISI**2)  # (93A-31)
                varISI = varX * (hISI**2).sum()  # (93A-31)

                # Step f - Jitter noise.
                hJ = calc_hJ(vic_pulse_resp, As, cursor_ix, self.nspui)
                # varJ = (self.com_params.A_DD**2 + self.com_params.sigma_Rj**2) * varX * sum(hJ**2)  # (93A-32)
                varJ = (self.com_params.A_DD**2 + self.com_params.sigma_Rj**2) * varX * (hJ**2).sum()  # (93A-32)

                # Step g - Crosstalk.
                varXT = 0.
                for pulse_resp in pulse_resps[1:]:  # (93A-34)
                    # pylint: disable=consider-using-generator
                    # varXT += max([sum(array(filt_pr_samps(pulse_resp[m::M], As))**2)
                    varXT += max([(filt_pr_samps(pulse_resp[m::M], As)**2).sum()
                                  for m in range(M)])  # (93A-33)
                varXT *= varX

                # Step h - Spectral noise.
                df = freqs[1]
                varN = (self.com_params.eta_0 / 1e9) * (abs(self.Hr * Hctf)**2).sum() * df  # (93A-35)

                # Step i - FOM calculation.
                fom = 10 * np.log10(As**2 / (varTx + varISI + varJ + varXT + varN))  # (93A-36)

            case OptMode.MMSE:
                theNoiseCalc = NoiseCalc(
                    L, tb, 0, times, pulse_resps_preFFE[0], pulse_resps_preFFE[1:],
                    freqs, self.Ht, self.chnls[0][1], self.Hr, Hctf,
                    self.com_params.eta_0, self.com_params.A_v, self.com_params.SNR_TX,
                    self.com_params.A_DD, self.com_params.sigma_Rj)
                rslt = mmse(theNoiseCalc, nRxTaps, nRxPreTaps, len(self.com_params.dfe_min), self.com_params.RLM,
                            self.com_params.L, array(bmin), array(bmax), array(rx_taps_min), array(rx_taps_max),
                            norm_mode=norm_mode)
                fom = rslt["fom"]
                rx_taps = rslt["rx_taps"]
                dfe_tap_weights = rslt["dfe_tap_weights"]
                pr_samps = rslt["h"]
                vic_pulse_resp = rslt["vic_pulse_resp"]  # Note: Does not include Rx FFE/DFE!
                vic_peak_loc = np.argmax(vic_pulse_resp)
                cursor_ix = rslt["cursor_ix"]
                As = 1.0
                varTx = rslt["varTx"]
                varISI = rslt["varISI"]
                varJ = rslt["varJ"]
                varXT = rslt["varXT"]
                varN = rslt["varN"]
                self.fom_rslts['mse'] = rslt['mse'] if 'mse' in rslt else None
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
            if unit_amp is None:
                unit_amp = self.unit_amp

            # Run the nested optimization loops.
            fom_max = -1000.0
            fom_max_changed = False
            for _gDC2 in self.com_params.g_DC2:
                for _gDC in self.com_params.g_DC:
                # for _gDC in self.com_params.g_DC[0]:  # type: ignore
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
                            tx_taps_best = self._tx_combs[_tx_ix]
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
            tx_taps_best = tx_taps
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
            return False  # Flags the caller that the next 5 settings have NOT been made.
        self.gDC2     = gDC2_best                                   # pylint: disable=possibly-used-before-assignment
        self.gDC      = gDC_best                                    # pylint: disable=possibly-used-before-assignment
        self.tx_taps  = tx_taps_best                                # pylint: disable=possibly-used-before-assignment
        self.rx_taps  = rx_taps_best                                # pylint: disable=possibly-used-before-assignment
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
        opt_mode: Optional[OptMode] = None,
        norm_mode: Optional[NormMode] = None,
        unit_amp: Optional[bool] = None,
        dbg_dict: Optional[Dict[str, Any]] = None
    ) -> tuple[float, float, int]:
        """
        Calculate the interference and noise for COM.

        Keyword Args:
            cursor_ix: An optional predetermined cursor index,
                to be used instead of our own estimate.
                (In support of MMSE.)
                Default: None
            opt_mode: Optimization mode.
                Default: None (i.e. - Use ``self.opt_mode``.)
            norm_mode: The tap weight normalization mode to use.
                Default: None (i.e. - Use ``self.norm_mode``.)
            unit_amp: Enforce unit pulse response amplitude when True.
                (For comparing ``przf()`` results to ``mmse()`` results.)
                Default: None (i.e. - Use ``self.unit_amp``.)
            dbg_dict: Optional dictionary into which debugging values may be stashed,
                for later analysis.
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
                - ``tx_taps``
                - ``rx_taps``

                (This assumption is embedded into the ``gen_pulse_resps()`` function.)

            2. Fills in the ``com_results`` dictionary w/ various useful values for debugging.

        ToDo:
            1. ``DER0 / 2`` in ``Ani`` calculation?
        """

        # Honor any mode overrides.
        opt_mode  = opt_mode  or self.opt_mode
        norm_mode = norm_mode or self.norm_mode
        if unit_amp is None:
            unit_amp = self.unit_amp

        # Copy instance variables.
        L = self.com_params.L
        M = self.nspui
        RLM = self.com_params.RLM
        freqs = self.freqs
        nDFE = len(self.com_params.dfe_min)

        self.set_status("Calculating COM...")
        pulse_resps = self.gen_pulse_resps(dfe_taps=array([]))  # DFE taps are included explicitly, below.
        vic_pulse_resp = pulse_resps[0]
        if cursor_ix is None:
            cursor_ix = loc_curs(vic_pulse_resp, self.nspui,
                                 array(self.com_params.dfe_max), array(self.com_params.dfe_min))
        curs_uis, curs_ofst = divmod(cursor_ix, M)
        vic_curs_val = vic_pulse_resp[cursor_ix]
        # Missing `2*` is also missing from `Ani` definition; so, they cancel in COM calc.
        As = RLM * vic_curs_val / (L - 1)
        ymax = 1.1 * As
        npts = 2 * min(int(ymax / 0.00001), 1_000) + 1  # Note 1 of 93A.1.7.1; MUST BE ODD!
        y = np.linspace(-ymax, ymax, npts)
        ystep = 2 * ymax / (npts - 1)

        # Sec. 93A.1.7.2
        varX = (L**2 - 1) / (3 * (L - 1)**2)                                    # (93A-29)
        df = freqs[1] - freqs[0]
        if self.nRxTaps:
            Hrx = calc_Hffe(
                freqs, 1 / self.fb, array(self.rx_taps).flatten(),
                self.nRxTaps - self.nRxPreTaps - 1, hasCurs=True)
        else:
            Hrx = np.ones(len(freqs))
        varN = self.com_params.eta_0 * (abs(self.Hr * self.Hctf * Hrx)**2).sum() * (df / 1e9)  # (93A-35) + Hffe
        varTx = vic_curs_val**2 * pow(10, -self.com_params.SNR_TX / 10)                     # (93A-30)
        hJ = calc_hJ(vic_pulse_resp, As, cursor_ix, self.nspui)
        _, pJ = delta_pmf(self.com_params.A_DD * hJ, L=L, y=y)
        varG = varTx + self.com_params.sigma_Rj**2 * varX * (hJ**2).sum() + varN              # (93A-41)
        pG = np.exp(-y**2 / (2 * varG)) / np.sqrt(TWOPI * varG) * ystep         # (93A-42), but converted to PMF.
        pN = np.convolve(pG, pJ, mode='same')                                   # (93A-43)

        # Sec. 93A.1.7.3
        self.set_status("Sec. 93A.1.7.3")
        # - ISI (Inconsistent w/ IEEE 802.3-22, but consistent w/ v2.60 of MATLAB code.)
        n_pre = min(5, curs_uis)
        # Sample every M points, such that we include our identified cursor sample.
        isi_sample_slice = slice(curs_ofst, len(vic_pulse_resp), M)
        isi_select_slice = slice(curs_uis - n_pre, curs_uis + 101)  # Ignore everything beyond 100 UI after cursor.
        tISI = self.times[isi_sample_slice][isi_select_slice]
        hISI = vic_pulse_resp[isi_sample_slice][isi_select_slice].copy()
        hISI[n_pre] = 0  # No ISI at cursor.
        dfe_slice = slice(n_pre + 1, n_pre + 1 + nDFE)
        dfe_tap_weights = np.maximum(                                           # (93A-26)
            self.com_params.dfe_min,
            np.minimum(
                self.com_params.dfe_max,
                (hISI[dfe_slice] / vic_curs_val)))
        hISI[dfe_slice] -= dfe_tap_weights * vic_curs_val
        hISI *= As
        _, pISI = delta_pmf(hISI, L=L, y=y, dbg_dict=dbg_dict)  # `hISI` from (93A-27); `p(y)` as per (93A-40)
        varISI = varX * (hISI**2).sum()  # (93A-31)

        # - Crosstalk
        xt_samps = []
        pks = []  # For debugging.
        py = pISI.copy()
        for pulse_resp in pulse_resps[1:]:  # (93A-44)
            i = np.argmax([(pulse_resp[m::M]**2).sum() for m in range(M)])  # (93A-33)
            samps = pulse_resp[i::M]
            xt_samps.append(samps)
            _, pk = delta_pmf(samps, L=L, y=y)  # For debugging.
            pks.append(pk)
            py = np.convolve(py, pk, mode='same')
        py = np.convolve(py, pN, mode='same')  # (93A-45)

        # Final calculation
        Py = np.cumsum(py)
        Py /= Py[-1]  # Enforce cumulative probability distribution.
        Ani = -y[np.where(Py >= self.com_params.DER_0)[0][0]]  # ToDo: `DER0 / 2`?

        # Store some results.
        self.com_rslts['As']          = As
        self.com_rslts['Ani']         = Ani
        self.com_rslts['pulse_resps'] = pulse_resps
        self.com_rslts['cursor_ix']   = cursor_ix
        self.com_rslts['sigma_Tx']    = np.sqrt(varTx)
        self.com_rslts['sigma_G']     = np.sqrt(varG)
        self.com_rslts['sigma_N']     = np.sqrt(varN)
        self.com_rslts['sigma_ISI']   = np.sqrt(varISI)
        self.com_rslts['tISI']        = tISI
        self.com_rslts['hISI']        = hISI
        self.com_rslts['pG']          = pG
        self.com_rslts['pN']          = pN
        self.com_rslts['pJ']          = pJ
        self.com_rslts['pISI']        = pISI
        self.com_rslts['py']          = py
        self.com_rslts['Py']          = Py
        self.com_rslts['y']           = y
        self.com_rslts['pks']         = pks
        self.com_rslts['dfe_taps']    = dfe_tap_weights
        self.com_rslts['xt_samps']    = xt_samps

        return (As, Ani, cursor_ix)

    about_str = """
      <H2><em>PyChOpMarg</em> - A Python implementation of COM, as per IEEE 802.3-22 Annex 93A/178A.</H2>\n
      <strong>By:</strong> David Banas <capn.freako@gmail.com><p>\n
      <strong>On:</strong> November 1, 2024<p>\n
      <strong>At:</strong> v2.2.1\n
      <H3>Useful Links</H3>\n
      (You'll probably need to: right click, select <em>Copy link address</em>, and paste into your browser.)
        <UL>\n
          <LI><a href="https://github.com/capn-freako/PyChOpMarg"><em>GitHub</em> Home</a>
          <LI><a href="https://pypi.org/project/PyChOpMarg/"><em>PyPi</em> Home</a>
          <LI><a href="https://readthedocs.org/projects/pychopmarg/"><em>Read the Docs</em> Home</a>
        </UL>
    """


if __name__ == "__main__":
    # from pychopmarg.cli import cli
    # cli()
    print("Sorry, the PyChOpMarg package is currently only usable as a library.")
    print("It's GUI is currently broken.")
