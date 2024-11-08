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
from pathlib import Path
from typing  import Any, Dict, Optional, TypeVar

import numpy as np  # type: ignore
import skrf  as rf  # type: ignore
from numpy             import array
from scipy.interpolate import interp1d

from pychopmarg.common   import Rvec, Cvec, PI, TWOPI
from pychopmarg.config.ieee_8023by import IEEE_8023by
from pychopmarg.config.template import COMParams
from pychopmarg.noise    import NoiseCalc
from pychopmarg.optimize import NormMode, mmse, przf
from pychopmarg.utility  import (
    import_s32p, sdd_21, sDieLadderSegment, sPkgTline, sCshunt, filt_pr_samps,
    delta_pmf, mk_combs, calc_Hffe, calc_Hctle, calc_H21, calc_hJ, loc_curs)

T = TypeVar('T', Any, Any)


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
    tmax = float(10e-9)  # system time vector maximum (s).
    fmax = float(40e9)   # system frequency vector maximum (Hz).
    com_params = IEEE_8023by

    # Linear EQ
    tx_taps: Rvec = array([])
    rx_taps: Rvec = array([])
    dfe_taps: Rvec = array([])
    nRxTaps: int = 0
    nRxPreTaps: int = 0     # `dw` from `com_params`
    gDC = 0.0               # Choices are in `com_params.g_DC`.
    gDC2 = 0.0              # Choices are in `com_params.g_DC2`.

    # Channel file(s)
    chnl_s32p = Path("")
    chnl_s4p_thru = Path("")
    chnl_s4p_fext1 = Path("")
    chnl_s4p_fext2 = Path("")
    chnl_s4p_fext3 = Path("")
    chnl_s4p_fext4 = Path("")
    chnl_s4p_fext5 = Path("")
    chnl_s4p_fext6 = Path("")
    chnl_s4p_next1 = Path("")
    chnl_s4p_next2 = Path("")
    chnl_s4p_next3 = Path("")
    chnl_s4p_next4 = Path("")
    chnl_s4p_next5 = Path("")
    chnl_s4p_next6 = Path("")
    vic_chnl_ix = int(1)
    chnls: list[tuple[rf.Network, str]] = []
    pulse_resps_nopkg: list[Rvec] = []
    pulse_resps_noeq:  list[Rvec] = []
    cursor_ix: int = 0

    # Package
    zp_sel = 0  # package length selector

    def __init__(self, com_params: COMParams, debug: bool = False):
        """
        Args:
            com_params: The COM parameters for this instance.

        Keyword Args:
            debug: Gather/report certain debugging information when ``True``.
                Default: ``False``
        """

        self.com_params = com_params
        self.debug = debug

        self.nRxTaps = len(com_params.rx_taps_max)
        self.nRxPreTaps = com_params.dw

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

        assert self.chnl_s32p or self.chnl_s4p_thru, RuntimeError(
            "You must, at least, select a thru path channel file, either 32 or 4 port.")

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

        self.chnls = self.get_chnls()
        self.set_status("Optimizing EQ...")
        assert self.opt_eq(do_opt_eq=do_opt_eq, tx_taps=tx_taps), RuntimeError("EQ optimization failed!")
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
        tstep = self.ui / self.nspui
        return np.arange(0, self.tmax + tstep, tstep)

    @property
    def fstep(self) -> float:
        "Frequency step (Hz)"
        return self.com_params.fstep

    @property
    def freqs(self) -> Rvec:
        "System frequency vector (Hz); decoupled from system time vector!"
        return np.arange(0, self.fmax + self.fstep, self.fstep)

    @property
    def t_irfft(self) -> Rvec:
        "`irfft()` result time index (s) (i.e. - time vector coupled to frequency vector)."
        Ts = 0.5 / self.fmax  # Sample period satisfying Nyquist criteria.
        return array([n * Ts for n in range(2 * (len(self.freqs) - 1))])

    @property
    def Xsinc(self) -> Rvec:
        """Frequency domain sinc(f) corresponding to Rect(ui) in time domain."""
        ui = self.ui
        Ts = self.t_irfft[1]
        w = int(ui / Ts)
        return w * np.sinc(ui * self.freqs)

    @property
    def Ht(self) -> Cvec:
        """
        Return the voltage transfer function, H(f), associated w/ the Tx risetime,
        according to (93A-46).
        """
        f = self.freqs / 1e9  # 93A-46 calls for f in GHz.
        return np.exp(-2 * (PI * f * self.com_params.T_r / 1.6832)**2)

    @property
    def Hr(self) -> Cvec:
        """
        Return the voltage transfer function, H(f), of the Rx AFE,
        according to (93A-20).
        """
        f = self.freqs / (self.com_params.f_r * self.fb)
        return 1 / (1 - 3.414214 * f**2 + f**4 + 2.613126j * (f - f**3))

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
    def tx_combs(self) -> list[list[float]]:
        "All possible Tx tap weight combinations."
        trips = list(zip(self.com_params.tx_taps_min,
                         self.com_params.tx_taps_max,
                         self.com_params.tx_taps_step))
        return mk_combs(trips)

    @property
    def gamma1(self) -> float:
        "Reflection coefficient looking out of the left end of the channel."
        Rd = self.com_params.R_d
        R0 = self.com_params.R_0
        return (Rd - R0) / (Rd + R0)

    @property
    def gamma2(self) -> float:
        "Reflection coefficient looking out of the right end of the channel."
        return self.gamma1

    @property
    def sDieLadder(self) -> rf.Network:
        "On-die parasitic capacitance/inductance ladder network."
        Cd = list(map(lambda x: x / 1e12, self.com_params.C_d))
        Ls = list(map(lambda x: x / 1e9, self.com_params.L_s))
        R0 = [self.com_params.R_0] * len(Cd)
        rslt = rf.network.cascade_list(list(map(lambda trip: sDieLadderSegment(self.freqs, trip), zip(R0, Cd, Ls))))
        return rslt

    @property
    def sPkgRx(self) -> rf.Network:
        "Rx package response."
        return self.sC(self.com_params.C_p / 1e12) ** self.sZp ** self.sDieLadder

    @property
    def sPkgTx(self) -> rf.Network:
        "Tx package response."
        return self.sDieLadder ** self.sZp ** self.sC(self.com_params.C_p / 1e12)

    @property
    def sPkgNEXT(self) -> rf.Network:
        "NEXT package response."
        return self.sDieLadder ** self.sZpNEXT ** self.sC(self.com_params.C_p / 1e12)

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

    # - Channels
    def get_chnls(self) -> list[tuple[rf.Network, str]]:
        """Import all channels from Touchstone file(s)."""
        chnl_s32p = self.chnl_s32p
        if chnl_s32p.exists() and chnl_s32p.is_file():
            return self.get_chnls_s32p_wPkg()
        return self.get_chnls_s4p_wPkg()

    def get_chnls_s32p_wPkg(self) -> list[tuple[rf.Network, str]]:
        """Augment imported s32p channels, w/ package response."""
        return self.add_pkgs(self.get_chnls_s32p_noPkg())

    def get_chnls_s4p_wPkg(self) -> list[tuple[rf.Network, str]]:
        """Augment imported s4p channels, w/ package response."""
        return self.add_pkgs(self.get_chnls_s4p_noPkg())

    def get_chnls_s32p_noPkg(self) -> list[tuple[rf.Network, str]]:
        """Import s32p file, w/o package."""
        if not self.chnl_s32p:
            return []
        ntwks = import_s32p(self.chnl_s32p, self.vic_chnl_ix)
        self.fmax = ntwks[0][0].f[-1]

        # Generate the pulse responses before adding the packages, for reference.
        self.pulse_resps_nopkg = self.gen_pulse_resps(ntwks, apply_eq=False)

        return ntwks

    def get_chnls_s4p_noPkg(self) -> list[tuple[rf.Network, str]]:
        """Import s4p files, w/o package."""
        if not self.chnl_s4p_thru:
            return []
        ntwks = [(sdd_21(rf.Network(self.chnl_s4p_thru)), 'THRU')]
        fmax = ntwks[0][0].f[-1]
        for fname in [self.chnl_s4p_fext1, self.chnl_s4p_fext2, self.chnl_s4p_fext3,
                      self.chnl_s4p_fext4, self.chnl_s4p_fext5, self.chnl_s4p_fext6]:
            if fname.exists() and fname.is_file():
                ntwk = sdd_21(rf.Network(fname))
                ntwks.append((ntwk, 'FEXT'))
                if ntwk.f[-1] < fmax:
                    fmax = ntwk.f[-1]
        for fname in [self.chnl_s4p_next1, self.chnl_s4p_next2, self.chnl_s4p_next3,
                      self.chnl_s4p_next4, self.chnl_s4p_next5, self.chnl_s4p_next6]:
            if fname.exists() and fname.is_file():
                ntwk = sdd_21(rf.Network(fname))
                ntwks.append((sdd_21(rf.Network(fname)), 'NEXT'))
                if ntwk.f[-1] < fmax:
                    fmax = ntwk.f[-1]
        self.fmax = fmax

        # Generate the pulse responses before adding the packages, for reference.
        self.pulse_resps_nopkg = self.gen_pulse_resps(ntwks, apply_eq=False)

        return ntwks

    def add_pkgs(self, ntwks: list[tuple[rf.Network, str]]) -> list[tuple[rf.Network, str]]:
        """Add package response to raw channels and generate pulse responses."""
        if not ntwks:
            return []
        _ntwks = list(map(self.add_pkg, ntwks))
        self.pulse_resps_noeq = self.gen_pulse_resps(_ntwks, apply_eq=False)
        return _ntwks

    def add_pkg(self, ntwk: tuple[rf.Network, str]) -> tuple[rf.Network, str]:
        """Add package response to raw channel."""
        ntype = ntwk[1]
        if ntype == 'NEXT':
            return (self.sPkgNEXT ** ntwk[0] ** self.sPkgRx, ntype)
        return (self.sPkgTx ** ntwk[0] ** self.sPkgRx, ntype)

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
        return calc_H21(self.freqs, s2p, self.gamma1, self.gamma2)

    def H(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
        self, s2p: rf.Network, tx_taps: Optional[Rvec] = None,
        gDC: Optional[float] = None, gDC2: Optional[float] = None,
        rx_taps: Optional[Rvec] = None, dfe_taps: Optional[Rvec] = None,
        passive_RxFFE: bool = False
    ) -> Cvec:
        """
        Return the voltage transfer function, H(f), of a complete COM signal path,
        according to (93A-19).

        Args:
            s2p: Two port network of interest.

        Keyword Args:
            tx_taps: Tx FFE tap weights.
                Default: None (i.e. - Use ``self.tx_taps``.)
            gDC: CTLE first stage d.c. gain (dB).
                Default: None (i.e. - Use ``self.gDC``.)
            gDC2: CTLE second stage d.c. gain (dB).
                Default: None (i.e. - Use ``self.gDC2``.)
            rx_taps: Rx FFE tap weights.
                Default: None (i.e. - Use ``self.rx_taps``.)
            dfe_taps: Rx DFE tap weights.
                Default: None (i.e. - Use ``self.dfe_taps``.)
            passive_RxFFE: Enforce passivity of Rx FFE when True.
                Default: True

        Returns:
            Complex voltage transfer function of complete path.

        Raises:
            ValueError: If given network is not two port.

        Notes:
            1. It is in this processing step that linear EQ is first applied.

            2. Any unprovided EQ values are taken from the ``COM`` instance.
            If you really want to omit a particular EQ component then call with:

                - ``tx_taps``: []
                - ``rx_taps``: [1.0]
                - ``gDC``/``gDC2``: 0
        """

        assert s2p.s[0, :, :].shape == (2, 2), ValueError(
            f"I can only convert 2-port networks. {s2p}")

        freqs = self.freqs
        tb = 1 / self.fb
        gDC     = gDC     or self.gDC
        gDC2    = gDC2    or self.gDC2
        if tx_taps is None:
            tx_taps = self.tx_taps
        if rx_taps is None:
            rx_taps = self.rx_taps
        if dfe_taps is None:
            dfe_taps = self.dfe_taps

        Htx  = calc_Hffe(freqs, tb, array(tx_taps).flatten(), 3)
        H21  = self.H21(s2p)
        Hr   = self.Hr
        Hctf = self.calc_Hctf(gDC=gDC, gDC2=gDC2)
        rslt = Htx * H21 * Hr * Hctf
        nRxTaps = len(rx_taps)
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

        Xsinc = self.Xsinc
        p = np.fft.irfft(Xsinc * H)
        spln = interp1d(self.t_irfft, p)  # `p` is not yet in our system time domain!
        return spln(self.times)           # Now, it is.

    def gen_pulse_resps(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, ntwks: Optional[list[tuple[rf.Network, str]]] = None,
        gDC: Optional[float] = None, gDC2: Optional[float] = None,
        tx_taps: Optional[Rvec] = None, rx_taps: Optional[Rvec] = None,
        dfe_taps: Optional[Rvec] = None, apply_eq: bool = True
    ) -> list[Rvec]:
        """
        Generate pulse responses for all networks.

        Keyword Args:
            ntwks: The list of networks to generate pulse responses for.
                Default: None (i.e. - Use ``self.chnls``.)
            gDC: Rx CTLE first stage d.c. gain.
                Default: None (i.e. - Use ``self.gDC``.)
            gDC2: Rx CTLE second stage d.c. gain.
                Default: None (i.e. - Use ``self.gDC2``.)
            tx_taps: Desired Tx tap weights.
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

        gDC  = gDC  or self.gDC  # The more Pythonic way, but doesn't work for lists in newer versions of Python.
        gDC2 = gDC2 or self.gDC2
        if ntwks is None:
            ntwks = self.chnls
        if tx_taps is None:
            tx_taps = self.tx_taps
        if rx_taps is None:
            rx_taps = self.rx_taps
        if dfe_taps is None:
            dfe_taps = self.dfe_taps

        tx_taps = array(tx_taps)
        rx_taps = array(rx_taps)
        dfe_taps = array(dfe_taps)

        pulse_resps = []
        for ntwk, ntype in ntwks:
            if apply_eq:
                if ntype == 'NEXT':
                    pr = self.pulse_resp(self.H(
                        ntwk, np.zeros(tx_taps.shape), gDC=gDC, gDC2=gDC2, rx_taps=rx_taps, dfe_taps=dfe_taps))
                else:
                    pr = self.pulse_resp(self.H(
                        ntwk, tx_taps,                 gDC=gDC, gDC2=gDC2, rx_taps=rx_taps, dfe_taps=dfe_taps))
            else:
                pr = self.pulse_resp(self.H21(ntwk))

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
        tx_taps: Rvec,
        gDC: Optional[float] = None, gDC2: Optional[float] = None,
        rx_taps: Optional[Rvec] = None,
        opt_mode: Optional[OptMode] = None,
        norm_mode: Optional[NormMode] = None,
        unit_amp: Optional[bool] = None
    ) -> float:
        """
        Calculate the **figure of merit** (FOM), given the existing linear EQ settings.
        Optimize Rx FFE taps if they aren't specified by caller.

        Args:
            tx_taps: The Tx FFE tap weights, excepting the cursor.
                (The cursor takes whatever is left.)

        Keyword Args:
            gDC: CTLE first stage d.c. gain.
                Default: None (i.e. - Use ``self.gDC``.)
            gDC2: CTLE second stage d.c. gain.
                Default: None (i.e. - Use ``self.gDC2``.)
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

            2. When not provided, the values for ``gDC`` and ``gDC2`` are taken from the ``COM`` instance.

            3. Unlike other member functions of the ``COM`` class,
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

        pulse_resps_preFFE = self.gen_pulse_resps(  # Assumes no Rx FFE/DFE.
            tx_taps=array(tx_taps), gDC=gDC, gDC2=gDC2, rx_taps=array([1.0]), dfe_taps=array([]))
        match opt_mode:
            case OptMode.PRZF:
                # Step a - Pulse response construction.
                pulse_resps = pulse_resps_preFFE
                if nRxTaps:              # If we have an Rx FFE...
                    if rx_taps is None:  # If we received no explicit override of the Rx FFE tap weight values,
                        rx_taps, _, pr_samps = przf(  # then optimize them.
                            pulse_resps_preFFE[0], M, nRxTaps, nRxPreTaps, nDFE,
                            array(rx_taps_min), array(rx_taps_max), array(bmin), array(bmax),
                            norm_mode=norm_mode, unit_amp=unit_amp)
                    pulse_resps = self.gen_pulse_resps(
                        tx_taps=array(tx_taps), gDC=gDC, gDC2=gDC2, rx_taps=array(rx_taps), dfe_taps=array([]))

                # Step b - Cursor identification.
                vic_pulse_resp = array(pulse_resps[0])  # Note: Includes any Rx FFE, but not DFE.
                vic_peak_loc = np.argmax(vic_pulse_resp)
                cursor_ix = loc_curs(vic_pulse_resp, self.nspui,
                                     array(self.com_params.dfe_max), array(self.com_params.dfe_min))

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
                varISI = varX * sum(hISI**2)  # (93A-31)

                # Step f - Jitter noise.
                hJ = calc_hJ(vic_pulse_resp, As, cursor_ix, self.nspui)
                varJ = (self.com_params.A_DD**2 + self.com_params.sigma_Rj**2) * varX * sum(hJ**2)  # (93A-32)

                # Step g - Crosstalk.
                varXT = 0.
                for pulse_resp in pulse_resps[1:]:  # (93A-34)
                    # pylint: disable=consider-using-generator
                    varXT += max([sum(array(filt_pr_samps(pulse_resp[m::M], As))**2)
                                  for m in range(M)])  # (93A-33)
                varXT *= varX

                # Step h - Spectral noise.
                df = freqs[1]
                Hctle = self.calc_Hctf(gDC=gDC, gDC2=gDC2)
                varN = (self.com_params.eta_0 / 1e9) * sum(abs(self.Hr * Hctle)**2) * df  # (93A-35)

                # Step i - FOM calculation.
                fom = 10 * np.log10(As**2 / (varTx + varISI + varJ + varXT + varN))  # (93A-36)

            case OptMode.MMSE:
                theNoiseCalc = NoiseCalc(
                    L, tb, 0, times, pulse_resps_preFFE[0], pulse_resps_preFFE[1:],
                    freqs, self.Ht, self.H21(self.chnls[0][0]), self.Hr, self.calc_Hctf(gDC, gDC2),
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
            def check_taps(tx_taps: Rvec, t0_min: float = self.com_params.c0_min) -> bool:
                if (1 - sum(abs(array(tx_taps)))) < t0_min:
                    return False
                return True

            fom_max = -1000.0
            fom_max_changed = False
            for _gDC2 in self.com_params.g_DC2:
                for _gDC in self.com_params.g_DC:
                    for _tx_taps in self.tx_combs:
                        if not check_taps(array(_tx_taps)):
                            continue
                        fom = self.calc_fom(
                            array(_tx_taps), gDC=_gDC, gDC2=_gDC2,
                            opt_mode=opt_mode, norm_mode=norm_mode, unit_amp=unit_amp)
                        if fom > fom_max:
                            fom_max_changed = True
                            fom_max = fom
                            gDC2_best = _gDC2
                            gDC_best = _gDC
                            tx_taps_best = array(_tx_taps)
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
            fom = self.calc_fom(tx_taps)
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
        Hrx = calc_Hffe(
            self.freqs, 1 / self.fb, array(self.rx_taps).flatten(),
            self.nRxTaps - self.nRxPreTaps - 1, hasCurs=True)
        varN = self.com_params.eta_0 * sum(abs(self.Hr * self.Hctf * Hrx)**2) * (df / 1e9)  # (93A-35) + Hffe
        varTx = vic_curs_val**2 * pow(10, -self.com_params.SNR_TX / 10)                     # (93A-30)
        hJ = calc_hJ(vic_pulse_resp, As, cursor_ix, self.nspui)
        _, pJ = delta_pmf(self.com_params.A_DD * hJ, L=L, y=y)
        varG = varTx + self.com_params.sigma_Rj**2 * varX * sum(hJ**2) + varN              # (93A-41)
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
        varISI = varX * sum(hISI**2)  # (93A-31)

        # - Crosstalk
        xt_samps = []
        pks = []  # For debugging.
        py = pISI.copy()
        for pulse_resp in pulse_resps[1:]:  # (93A-44)
            i = np.argmax([sum(array(pulse_resp[m::M])**2) for m in range(M)])  # (93A-33)
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
