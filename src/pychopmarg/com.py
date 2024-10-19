#!/usr/bin/env python

"""
The Channel Operating Margin (COM) model, as per IEEE 802.3-22 Annex 93A.

Original author: David Banas <capn.freako@gmail.com>

Original date:   February 29, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

Notes:
    1. Throughout this file, equation numbers refer to Annex 93A of the IEEE 802.3-22 standard.
    2. Throughout this file, reference may be made to the following, via "[n]" syntax:
        [1] - Healey, A., Hegde, R., _Reference receiver framework for 200G/lane electrical interfaces and PHYs_, IEEE P802.3dj Task Force, January 2024 (r4).
    3. The superficial two dimmensional nature of the Tx/Rx FFE tap weight vectors is imposed on us by the GUI machinery.
        Let's keep it confined to just the GUI code!

ToDo:
    1. Provide type hints for imports.
    2. Move non-class code to a new file.
"""

import numpy as np  # type: ignore
import skrf  as rf  # type: ignore

from enum   import Enum
from typing import Any, Optional, TypeVar

from chaco.api import (
    ArrayPlotData,
    GridPlotContainer,
    Plot,
)  # type: ignore
from chaco.tools.api   import ZoomTool
from numpy.typing      import NDArray
from scipy.interpolate import interp1d
from traits.api        import (
    Array,
    Bool,
    cached_property,
    Enum,
    File,
    Float,
    HasTraits,
    Int,
    List,
    Property,
    Str,
)  # type: ignore

from pychopmarg.common   import Rvec, Cvec, COMParams, PI, TWOPI
from pychopmarg.noise    import NoiseCalc
from pychopmarg.optimize import mmse, przf
from pychopmarg.utility  import import_s32p, sdd_21, sDieLadderSegment, delta_pmf

# Globals
# These are used by `calc_Hffe()`, to minimize the size of its cache.
# They are initialized by `COM.__init__()`.
gFreqs: Rvec = None  # type: ignore
gFb: float = None  # type: ignore
gC0min: float = None  # type: ignore
gNtaps: int = None  # type: ignore
PLOT_SPACING = 20

T = TypeVar('T', Any, Any)

class OptMode(Enum):
    PRZF = 1
    MMSE = 2


def from_dB(x: float) -> float:
    """Convert from (dB) to real, assuming square law applies."""
    return pow(10, x / 20)


def all_combs(xss: list[list[T]]) -> list[list[T]]:
    """
    Generate all combinations of input.

    Args:
        xss([[T]]): The lists of candidates for each position in the final output.

    Returns:
        All possible combinations of input lists.
    """
    if not xss:
        return [[]]
    head, *tail = xss
    yss = all_combs(tail)
    return [[x] + ys for x in head for ys in yss]


def mk_combs(trips: list[tuple[float, float, float]]) -> list[list[float]]:
    """
    Make all possible combinations of tap weights, given a list of "(min, max, step)" triples.

    Args:
        trips: A list of "(min, max, step)" triples, one per weight.

    Returns:
        combs: A list of lists of tap weights, including all possible combinations.
    """
    ranges = []
    for trip in trips:
        if trip[2]:  # non-zero step?
            ranges.append(list(np.arange(trip[0], trip[1] + trip[2], trip[2])))
        else:
            ranges.append([0.0])
    return all_combs(ranges)


def calc_Hffe(tap_weights: Rvec, n_post: int, isRx: bool = False) -> Cvec:
    """
    Calculate the voltage transfer function, H(f), for a FFE, according to (93A-21).

    Args:
        tap_weights: The vector of filter tap weights, excluding the cursor tap.
        n_post: The number of post-cursor taps.

    Keyword Args:
        isRx: Tap weights include cursor when True.
            Default: False

    Returns:
        The complex voltage transfer function, H(f), for the FFE.

    Raises:
        RuntimeError: If certain global variables haven't been initialized.
            (See Note 1, below.)

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

    ToDo:
        1. Pull this function back inside the `COM` class, and do manual memoization.
        2. Fix broken usage for Rx FFE.
    """

    assert len(gFreqs) and gFb and gC0min and gNtaps, RuntimeError(
        f"Called before global variables were initialized!\n\tgFreqs: \
        {gFreqs}, gFb: {gFb}, gC0min: {gC0min}, gNtaps: {gNtaps}")

    cs = list(np.array(tap_weights).flatten())
    if not isRx:
        c0 = 1 - sum(list(map(abs, tap_weights)))
        if c0 < gC0min:
            raise ValueError(f"Calculated `c0` for Tx FFE ({c0}) is below allowed minimum ({gC0min})!")
        cs.insert(-n_post, c0)
    return sum(list(map(lambda n_c: n_c[1] * np.exp(-1j * TWOPI * n_c[0] * gFreqs / gFb),
                        enumerate(cs))))


def calc_Hdfe(tap_weights: Rvec) -> Cvec:
    """
    Calculate the voltage transfer function, H(f), for a DFE.

    Args:
        tap_weights: The vector of filter tap weights.

    Returns:
        The complex voltage transfer function, H(f), for the DFE.
    """

    assert len(gFreqs) and gFb, RuntimeError(
        f"Called before global variables were initialized!\n\tgFreqs: {gFreqs}, gFb: {gFb}")

    return 1 / (1 - sum(list(map(lambda n_b: n_b[1] * np.exp(-1j * TWOPI * (n_b[0] + 1) * gFreqs / gFb),
                                 enumerate(tap_weights)))))


def print_taps(ws: list[float]) -> str:
    """Return formatted tap weight values."""
    n_ws = len(ws)
    if n_ws == 0:
        return ""
    res = f"{ws[0]:5.2f}"
    if n_ws > 1:
        if n_ws > 8:
            for w in ws[1:8]:
                res += f" {w:5.2f}"
            res += f"\n{ws[8]:5.2f}"
            for w in ws[9:]:
                res += f" {w:5.2f}"
        else:
            for w in ws[1:]:
                res += f" {w:5.2f}"
    return res


class COM(HasTraits):
    """
    Encoding of the IEEE 802.3-22 Annex 93A "Channel Operating Margin"
    (COM) specification, as a Python class making use of the Enthought
    Traits/UI machinery, for both calculation efficiency and easy GUI display.
    """

    status_str = Str("Ready")
    debug = Bool(False)

    # Independent variable definitions (Defaults from IEEE 802.3by.)
    # Units are SI, except for: (dB), `zp`, and `eta0`.
    # - System time/frequency vectors (decoupled!)
    FB = 25e9 * 66. / 64.  # DO NOT REMOVE! Initializing other Floats w/ `fb` doesn't work!
    fb = Float(FB)  # Baud. rate.
    nspui = Int(32)  # samples per UI.
    tmax = Float(10e-9)  # system time vector maximum.
    fstep = Float(10e6)  # system frequency vector step.
    fmax = Float(40e9)  # system frequency vector maximum.
    # - Tx Drive Settings
    Av = Float(0.4)  # victim drive voltage (V).
    Afe = Float(0.4)  # FEXT drive voltage (V).
    Ane = Float(0.6)  # NEXT drive voltage (V).
    L = Int(2)  # number of output levels.
    RLM = Float(1.0)  # ratio level mismatch.
    tr = Float(10e-12)  # Tx output risetime (s).
    # - Linear EQ
    nTxTaps      = 6
    tx_n_post    = Int(3)
    tx_taps_pos  = Array(shape=(nTxTaps,), dtype=int,   value=[-3, -2, -1,    1,    2,  3])
    tx_taps_min  = Array(shape=(nTxTaps,), dtype=float, value=[ 0., 0., 0.,   0.,   0., 0.])
    tx_taps_max  = Array(shape=(nTxTaps,), dtype=float, value=[ 0., 0., 0.,   0.,   0., 0.])
    tx_taps_step = Array(shape=(nTxTaps,), dtype=float, value=[ 0., 0., 0.02, 0.02, 0., 0.])
    tx_taps      = Array(shape=(nTxTaps,), dtype=float, value=[ 0.] * nTxTaps)  # Tx FFE tap weights.
    c0_min = Float(0)  # minimum allowed Tx FFE main tap value.
    fr = Float(0.75)  # AFE corner frequency (fb)
    gDC_vals = List([-x for x in range(13)])  # D.C. gain of Rx CTLE first stage (dB).
    gDC = Enum(0, values="gDC_vals")
    gDC2_vals = List([0.,])  # D.C. gain of Rx CTLE second stage (dB).
    gDC2 = Enum(0, values="gDC2_vals")
    fz = Float(FB / 4.)  # CTLE zero frequency.
    fp1 = Float(FB / 4.)  # CTLE first pole frequency.
    fp2 = Float(FB)  # CTLE second pole frequency.
    fLF = Float(1e6)  # CTLE low-f corner frequency.
    opt_mode_vals = List([OptMode.PRZF, OptMode.MMSE])
    opt_mode = Enum(OptMode.PRZF, values="opt_mode_vals")
    # - FFE/DFE
    N_DFE = 1  # DO NOT REMOVE! Initializing other Ints using `nDFE` doesn't work!
    nDFE = Int(N_DFE)  # number of DFE taps.
    bmax = List([1.0] * N_DFE)  # DFE maximum tap values.
    bmin = List([-1.0] * N_DFE)  # DFE minimum tap values.
    dfe_taps = Array(shape=(N_DFE,), dtype=float, value=[0.] * N_DFE)  # Rx FFE tap weights.
    nRxTaps      = 15
    nRxPreTaps   = 3
    rx_taps_min  = Array(shape=(nRxTaps,), dtype=float, value=[0.] * nRxTaps)
    rx_taps_max  = Array(shape=(nRxTaps,), dtype=float, value=[0.] * nRxTaps)
    rx_taps      = Array(shape=(nRxTaps,), dtype=float, value=[0.] * nRxTaps)  # Rx FFE tap weights.
    
    # - Package & Die Modeling
    # -- MKS
    R0 = Float(50.0)  # system reference impedance.
    Rd = Float(55.0)  # on-die termination impedance.
    Cd = List([0.04e-12, 0.09e-12, 0.11e-12])  # parasitic die capacitances.
    Cb = Float(0.00e-12)  # parasitic bump capacitance.
    Cp = Float(0.18e-12)  # parasitic ball capacitance.
    Ls = List([0.13e-9, 0.15e-9, 0.14e-9])  # parasitic die inductances.
    # -- ns & mm
    zp_vals = List([12, 33])
    zp = Enum(values="zp_vals")
    zp_B = Float(1.8)
    zc = List([87.5, 92.5])  # package transmission line characteristic impedances (Ohms).
    gamma0 = Float(5.0e-4)   # propagation loss constant (1/mm)
    a1 = Float(8.9e-4)       # first polynomial coefficient (sqrt_ns/mm)
    a2 = Float(2.0e-4)       # second polynomial coefficient (ns/mm)
    tau = Float(6.141e-3)    # propagation delay (ns/mm)
    # - Noise & DER
    sigma_Rj = Float(0.01)  # random jitter standard deviation (ui).
    Add = Float(0.05)  # deterministic jitter amplitude (ui).
    eta0 = Float(5.2e-8)  # spectral noise density (V^2/GHz).
    TxSNR = Float(27)  # Tx signal-to-noise ratio (dB).
    DER0 = Float(1e-5)  # detector error rate threshold.
    # - Channel file(s)
    chnl_s32p = File("", entries=5, filter=["*.s32p"], exists=False)
    chnl_s4p_thru = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_fext1 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_fext2 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_fext3 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_fext4 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_fext5 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_fext6 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_next1 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_next2 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_next3 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_next4 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_next5 = File("", entries=5, filter=["*.s4p"], exists=True)
    chnl_s4p_next6 = File("", entries=5, filter=["*.s4p"], exists=True)
    vic_chnl_ix = Int(1)
    # - Results
    # -- COM
    com = Float(0.0)
    com_As = Float(0.0)
    com_cursor_ix = Int(0)
    com_sigma_Tx = Float(0.0)
    com_sigma_G = Float(0.0)
    com_sigma_N = Float(0.0)
    com_dfe_taps = Str(print_taps([0.0] * N_DFE))
    # -- FOM
    fom = Float(0.0)
    fom_As = Float(0.0)
    Ani = Float(0.0)
    fom_cursor_ix = Int(0)
    sigma_ISI = Float(0.0)
    sigma_J = Float(0.0)
    sigma_XT = Float(0.0)
    sigma_Tx = Float(0.0)
    sigma_N = Float(0.0)
    fom_tx_taps = Str(print_taps([0.0] * nTxTaps))
    fom_dfe_taps = Str(print_taps([0.0] * N_DFE))

    # Plots (plot containers, actually).
    plotdata = ArrayPlotData()
    plotdata.set_data("t_ns", np.linspace(0., 10., 100))
    plotdata.set_data("pulse_raw", np.zeros(100))
    plotdata.set_data("pulse_pkg", np.zeros(100))
    plotdata.set_data("pulse_eq", np.zeros(100))
    # - pulse responses.
    plot1 = Plot(plotdata, padding_left=75)
    plot1.plot(("t_ns", "pulse_raw"), name="Raw", type="line", color="red")
    plot1.plot(("t_ns", "pulse_pkg"), name="w/ Pkg", type="line", color="green")
    plot1.plot(("t_ns", "pulse_eq"), name="w/ EQ", type="line", color="blue")
    plot1.title = "Pulse Responses"
    plot1.index_axis.title = "Time (ns)"
    plot1.value_axis.title = "p(t) (mV)"
    plot1.legend.visible = True
    plot1.legend.align = "ur"
    zoom1 = ZoomTool(plot1, tool_mode="range", axis="index", always_on=False)
    plot1.overlays.append(zoom1)
    cont1 = GridPlotContainer(shape=(1, 1), spacing=(PLOT_SPACING, PLOT_SPACING))
    cont1.add(plot1)

    about_str = """
      <H2><em>PyChOpMarg</em> - A Python implementation of COM, as per IEEE 802.3-22 Annex 93A.</H2>\n
      <strong>By:</strong> David Banas <capn.freako@gmail.com><p>\n
      <strong>On:</strong> August 12, 2024<p>\n
      <strong>At:</strong> v1.1.3\n
      <H3>Useful Links</H3>\n
      (You'll probably need to: right click, select <em>Copy link address</em>, and paste into your browser.)
        <UL>\n
          <LI><a href="https://github.com/capn-freako/PyChOpMarg"><em>GitHub</em> Home</a>
          <LI><a href="https://pypi.org/project/PyChOpMarg/"><em>PyPi</em> Home</a>
          <LI><a href="https://readthedocs.org/projects/pychopmarg/"><em>Read the Docs</em> Home</a>
        </UL>
    """

    def _fb_changed(self):
        "Keep global version in sync."
        global gFb
        gFb = self.fb

    def _c0_min_changed(self):
        "Keep global version in sync."
        global gC0min
        gC0min = self.c0_min

    def _tx_taps_min_changed(self):
        "Keep global version in sync."
        global gNtaps
        gNtaps = len(self.tx_taps_min.flatten())

    # Dependent variable definitions
    # - Unit interval (s).
    ui = Property(Float, depends_on=["fb"])

    @cached_property
    def _get_ui(self):
        """Unit interval (s)."""
        return 1 / self.fb

    # - system time vector; decoupled from system frequency vector!
    times = Property(Array, depends_on=["fb", "nspui", "tmax"])

    @cached_property
    def _get_times(self):
        """System times (s)."""
        tstep = self.ui / self.nspui
        rslt = np.arange(0, self.tmax + tstep, tstep)
        self.plotdata.set_data("t_ns", rslt * 1e9)
        return rslt

    # - system frequency vector; decoupled from system time vector!
    freqs = Property(Array, depends_on=["fstep", "fmax"])

    @cached_property
    def _get_freqs(self):
        """System frequencies (Hz)."""
        global gFreqs
        freqs = np.arange(0, self.fmax + self.fstep, self.fstep)
        gFreqs = freqs
        return freqs

    # - `irfft()` time vector; decoupled from system time vector!
    t_irfft = Property(Array, depends_on=["freqs"])

    @cached_property
    def _get_t_irfft(self):
        """`irfft()` result time index (s)."""
        Ts = 0.5 / self.fmax  # Sample period satisfying Nyquist criteria.
        return np.array([n * Ts for n in range(2 * (len(self.freqs) - 1))])

    # - Rect(ui) in the frequency domain.
    Xsinc = Property(Array, depends_on=["ui", "freqs"])

    @cached_property
    def _get_Xsinc(self):
        """Frequency domain sinc(f) corresponding to Rect(ui)."""
        ui = self.ui
        Ts = self.t_irfft[1]
        w = int(ui / Ts)
        return w * np.sinc(ui * self.freqs)

    # - Tx risetime transfer function.
    Ht = Property(Array, depends_on=['tr'])

    @cached_property
    def _get_Ht(self):
        """
        Return the voltage transfer function, H(f), associated w/ the Tx risetime,
        according to (93A-46).
        """
        f = self.freqs / 1e9  # 93A-46 calls for f in GHz.
        return np.exp(-2 * (PI * f * self.tr / 1.6832)**2)

    # - Rx AFE voltage transfer function.
    Hr = Property(Array, depends_on=['freqs'])

    @cached_property
    def _get_Hr(self):
        """
        Return the voltage transfer function, H(f), of the Rx AFE,
        according to (93A-20).
        """
        f = self.freqs / (self.fr * self.fb)
        return 1 / (1 - 3.414214 * f**2 + f**4 + 2.613126j * (f - f**3))

    # - Rx CTLE voltage transfer function.
    Hctf = Property(Array, depends_on=['freqs', 'gDC', 'gDC2', 'fz', 'fp1', 'fp2', 'fLF'])

    @cached_property
    def _get_Hctf(self):
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
            1. The instance's current value(s) for `gDC` and `gDC2` are used if not provided.
                (Necessary, to accommodate sweeping when optimizing EQ.)
        """
        if gDC is None:  # ToDo: Simplify syntax.
            gDC = self.gDC
        if gDC2 is None:
            gDC2 = self.gDC2
        g1 = from_dB(gDC)
        g2 = from_dB(gDC2)
        f = self.freqs
        fz = self.fz
        fp1 = self.fp1
        fp2 = self.fp2
        fLF = self.fLF
        num = (g1 + 1j * f / fz) * (g2 + 1j * f / fLF)
        den = (1 + 1j * f / fp1) * (1 + 1j * f / fp2) * (1 + 1j * f / fLF)
        return num / den

    # - All possible Tx/Rx tap weight combinations.
    tx_combs = Property(List, depends_on=['tx_taps_min', 'tx_taps_max', 'tx_taps_step'])
    # rx_combs = Property(List, depends_on=['rx_taps_min', 'rx_taps_max', 'rx_taps_step'])

    @cached_property
    def _get_tx_combs(self):
        """
        All possible Tx tap weight combinations.
        """
        trips = list(zip(self.tx_taps_min.flatten(), self.tx_taps_max.flatten(), self.tx_taps_step.flatten()))
        return mk_combs(trips)

    # @cached_property
    # def _get_rx_combs(self):
    #     """
    #     All possible Rx tap weight combinations.
    #     """
    #     trips = list(zip(self.rx_taps_min.flatten(), self.rx_taps_max.flatten(), self.rx_taps_step.flatten()))
    #     return mk_combs(trips)

    # - Reflection coefficients
    gamma1 = Property(Float, depends_on=['Rd', 'R0'])
    gamma2 = Property(Float, depends_on=['gamma1'])

    @cached_property
    def _get_gamma1(self):
        """
        Reflection coefficient looking out of the left end of the channel.
        """
        Rd = self.Rd
        R0 = self.R0
        return (Rd - R0) / (Rd + R0)

    @cached_property
    def _get_gamma2(self):
        """
        Reflection coefficient looking out of the right end of the channel.
        """
        return self.gamma1

    # - Die/Package response
    sDieLadder = Property(depends_on=['R0', 'freqs', 'Cd', 'Ls'])  # noqa E221
    sPkgRx   = Property(depends_on=['zc', 'R0', 'zp', 'freqs', 'Cd', 'Cp', 'Ls'])  # noqa E221
    sPkgTx   = Property(depends_on=['zc', 'R0', 'zp', 'freqs', 'Cd', 'Cp', 'Ls'])  # noqa E221
    sPkgNEXT = Property(depends_on=['zc', 'R0', 'zp', 'freqs', 'Cd', 'Cp', 'Ls'])  # noqa E221
    sZp      = Property(depends_on=['zc', 'R0', 'zp', 'freqs'])  # noqa E221
    sZpNEXT  = Property(depends_on=['zc', 'R0', 'zp', 'freqs'])  # noqa E221

    @cached_property
    def _get_sDieLadder(self) -> rf.Network:
        """
        On-die parasitic capacitance/inductance ladder network.
        """

        Cd = self.Cd
        Ls = self.Ls
        R0 = [self.R0] * len(Cd)
        rslt = rf.network.cascade_list(list(map(lambda trip: sDieLadderSegment(self.freqs, trip), zip(R0, Cd, Ls))))
        return rslt

    @cached_property
    def _get_sPkgRx(self) -> rf.Network:
        """
        Rx package response.
        """
        # return self.sC(self.Cp) ** self.sZp ** self.sC(self.Cd)
        return self.sC(self.Cp) ** self.sZp ** self.sDieLadder

    @cached_property
    def _get_sPkgTx(self) -> rf.Network:
        """
        Tx package response.
        """
        # return self.sC(self.Cd) ** self.sZp ** self.sC(self.Cp)
        return self.sDieLadder ** self.sZp ** self.sC(self.Cp)

    @cached_property
    def _get_sPkgNEXT(self) -> rf.Network:
        """
        NEXT package response.
        """
        # return self.sC(self.Cd) ** self.sZpNEXT ** self.sC(self.Cp)
        return self.sDieLadder ** self.sZpNEXT ** self.sC(self.Cp)

    @cached_property
    def _get_sZp(self) -> rf.Network:
        return self.calc_sZp()

    @cached_property
    def _get_sZpNEXT(self) -> rf.Network:
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

        zc = self.zc
        assert len(zc) in [1, 2], ValueError(
            f"Length of `zc` ({len(zc)}) must be 1 or 2!")

        if NEXT:
            zp = self.zp_vals[0]
        else:
            zp = self.zp
        if len(zc) == 1:
            zps = [zp]
        else:
            zps = [zp, self.zp_B]

        f      = self.freqs
        f_GHz  = f / 1e9               # noqa E221
        r0     = self.R0
        a1     = self.a1      # sqrt(ns)/mm
        a2     = self.a2      # ns/mm
        tau    = self.tau     # ns/mm
        gamma0 = self.gamma0  # 1/mm
        gamma1 = a1 * (1 + 1j)

        def gamma2(f: float) -> complex:
            "f in GHz!"
            return a2 * (1 - 1j * (2 / PI) * np.log(f)) + 1j * TWOPI * tau

        def gamma(f: float) -> complex:
            "Return complex propagation coefficient at frequency f (GHz)."
            if f == 0:
                return gamma0
            else:
                return gamma0 + gamma1 * np.sqrt(f) + gamma2(f) * f

        g = np.array(list(map(gamma, f_GHz)))

        def mk_s2p(z_pair: tuple[float, float]) -> rf.Network:
            """
            Make two port network for a leg of T-line.

            Args:
                z_pair: Pair consisting of:
                    - zc: Characteristic impedance of leg (Ohms).
                    - zp: Length of leg (mm).

            Returns:
                s2p: Two port network for the leg.
            """
            zc, zp = z_pair
            rho = (zc - 2 * r0) / (zc + 2 * r0)  # noqa E221
            s11 = s22 = rho * (1 - np.exp(-g * 2 * zp)) / (1 - rho**2 * np.exp(-g * 2 * zp))
            s21 = s12 = (1 - rho**2) * np.exp(-g * zp)  / (1 - rho**2 * np.exp(-g * 2 * zp))
            return rf.Network(s=np.array(list(zip(zip(s11, s21), zip(s21, s11)))), f=f, z0=r0)

        return rf.network.cascade_list(list(map(mk_s2p, zip(zc, zps))))

    # - Channels
    def get_chnls(self) -> list[tuple[rf.Network, str]]:
        """Import all channels from Touchstone file(s)."""
        if self.chnl_s32p:
            return self.get_chnls_s32p_wPkg()
        else:
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
        pulse_resps_nopkg = self.gen_pulse_resps(ntwks, apply_eq=False)
        self.plotdata.set_data("pulse_raw", pulse_resps_nopkg[0])
        self.pulse_resps_nopkg = pulse_resps_nopkg

        return ntwks

    # @cached_property
    def get_chnls_s4p_noPkg(self) -> list[tuple[rf.Network, str]]:
        """Import s4p files, w/o package."""
        if not self.chnl_s4p_thru:
            return []
        ntwks = [(sdd_21(rf.Network(self.chnl_s4p_thru)), 'THRU')]
        fmax = ntwks[0][0].f[-1]
        for fname in [self.chnl_s4p_fext1, self.chnl_s4p_fext2, self.chnl_s4p_fext3,
                      self.chnl_s4p_fext4, self.chnl_s4p_fext5, self.chnl_s4p_fext6]:
            if fname:
                ntwk = sdd_21(rf.Network(fname))
                ntwks.append((ntwk, 'FEXT'))
                if ntwk.f[-1] < fmax:
                    fmax = ntwk.f[-1]
        for fname in [self.chnl_s4p_next1, self.chnl_s4p_next2, self.chnl_s4p_next3,
                      self.chnl_s4p_next4, self.chnl_s4p_next5, self.chnl_s4p_next6]:
            if fname:
                ntwk = sdd_21(rf.Network(fname))
                ntwks.append((sdd_21(rf.Network(fname)), 'NEXT'))
                if ntwk.f[-1] < fmax:
                    fmax = ntwk.f[-1]
        self.fmax = fmax

        # Generate the pulse responses before adding the packages, for reference.
        pulse_resps_nopkg = self.gen_pulse_resps(ntwks, apply_eq=False)
        self.plotdata.set_data("pulse_raw", pulse_resps_nopkg[0])
        self.pulse_resps_nopkg = pulse_resps_nopkg

        return ntwks

    def add_pkgs(self, ntwks: list[tuple[rf.Network, str]]) -> list[tuple[rf.Network, str]]:
        """Add package response to raw channels and generate pulse responses."""
        if not ntwks:
            return []
        _ntwks = list(map(self.add_pkg, ntwks))
        pulse_resps_noeq = self.gen_pulse_resps(_ntwks, apply_eq=False)
        self.rslts['vic_pulse_pk'] = max(pulse_resps_noeq[0]) * 1_000  # (mV)
        self.pulse_resps_noeq = pulse_resps_noeq
        self.plotdata.set_data("pulse_pkg", pulse_resps_noeq[0])
        return _ntwks

    def add_pkg(self, ntwk: tuple[rf.Network, str]) -> tuple[rf.Network, str]:
        """Add package response to raw channel."""
        ntype = ntwk[1]
        if ntype == 'NEXT':
            return (self.sPkgNEXT ** ntwk[0] ** self.sPkgRx, ntype)
        else:
            return (self.sPkgTx ** ntwk[0] ** self.sPkgRx, ntype)

    # Logging / Debugging
    def set_status(self, status: str) -> None:
        "Set the GUI status string and print it if we're debugging."
        self.status_str = status
        if self.debug:
            print(status, flush=True)

    # Reserved functions
    def __init__(self, debug: bool = False):
        """
        Instance creation only! Must still initialize.
        """

        # Super-class initialization is ABSOLUTELY NECESSARY, in order
        # to get all the Traits/UI machinery setup correctly.
        super().__init__()

        self.debug = debug

        self.rslts = {}
        self.fom_rslts = {}
        self.dbg = {}

        self.c0_min = 0.62
        self.tx_taps_min = [0., 0., -0.18, -0.38, 0., 0.]

        self.set_status("Ready")

    @classmethod
    def init(cls, params: dict, chnl_fnames: list[str], vic_id: int,
             zp_sel: int = 1, num_ui: int = 500, gui: bool = False):
        """
        Legacy initializer supports my VITA notebook, which was created before
        PyChOpMarg was altered to support a GUI.
        """
        obj = cls()
        obj.set_params(params)
        assert zp_sel > 0 and zp_sel <= len(obj.zp_vals), RuntimeError(
            "`zp_sel` out of range!")
        if len(chnl_fnames) == 1:
            obj.chnl_s32p = chnl_fnames[0]
        else:
            obj.chnl_s4p_thru = chnl_fnames[0]
            if len(chnl_fnames) > 1:
                obj.chnl_s4p_fext1 = chnl_fnames[1]
            if len(chnl_fnames) > 2:
                obj.chnl_s4p_fext2 = chnl_fnames[2]
            if len(chnl_fnames) > 3:
                obj.chnl_s4p_next1 = chnl_fnames[3]
            if len(chnl_fnames) > 4:
                obj.chnl_s4p_next2 = chnl_fnames[4]
            if len(chnl_fnames) > 5:
                obj.chnl_s4p_next3 = chnl_fnames[5]
        obj.vic_chnl_ix = vic_id
        obj.zp = obj.zp_vals[zp_sel - 1]
        obj.num_ui = num_ui
        obj.gui = gui
        return obj

    def __call__(self,
                 do_opt_eq: bool    = True,
                 tx_taps:   Rvec    = None,
                 opt_mode:  OptMode = OptMode.PRZF
                ) -> float:
        """
        Calculate the COM value.

        Keyword Args:
            do_opt_eq: Perform optimization of linear equalization when True.
                Default: True
            tx_taps: Used when `do_opt_eq` = False.
                Default: None
            opt_mode: Method for EQ optimization.
                Default: OptMode.PRZF (i.e. - Use Pulse Response Zero Forcing.)

        Returns:
            COM: The calculated COM value (dB).

        Raises:
            RuntimeError if throughput channel is missing.
            RuntimeError if EQ optimization fails.
        """

        assert self.chnl_s32p or self.chnl_s4p_thru, RuntimeError(
            "You must, at least, select a thru path channel file, either 32 or 4 port.")
        self.chnls = self.get_chnls()
        self.set_status("Optimizing EQ...")
        self.opt_mode = opt_mode
        assert self.opt_eq(do_opt_eq=do_opt_eq, tx_taps=tx_taps, opt_mode=opt_mode), RuntimeError(
            "EQ optimization failed!")
        # print(f"After EQ optimization, sigma_ISI: {self.rslts['sigma_ISI']} mV", flush=True)
        self.set_status("Calculating noise...")
        As, Ani, self.cursor_ix = self.calc_noise()
        # print(f"After noise calculation, sigma_ISI: {self.rslts['sigma_ISI']} mV", flush=True)
        com = 20 * np.log10(As / Ani)
        self.As = As
        self.Ani = Ani
        self.com = com
        self.rslts['com'] = com
        self.rslts['As'] = As * 1e3
        self.rslts['Ani'] = Ani * 1e3
        self.set_status(f"Ready; COM = {com: 5.1f} dB")
        return com

    # Initializers
    def set_params(self, params: COMParams) -> None:
        """
        Set the COM instance parameters, according to the given dictionary.

        Args:
            params: Dictionary of COM parameter values.

        Raises:
            KeyError: If an expected key is not found in the provided dictionary.
            ValueError: If certain invariants aren't met.
        """

        # Set default parameter values, as necessary.
        if 'z_c' not in params:
            params['zc'] = 78.2
        if 'C_b' not in params:
            params['C_b'] = 0.0
        if 'f_LF' not in params:
            params['f_LF'] = 0.001
        if 'g_DC2' not in params:
            params['g_DC2'] = 0.0
        if 'T_r' not in params:
            params['T_r'] = 10e-12

        # Capture parameters, adjusting units as necessary to keep all but
        # (dB), package values, and `eta0` SI.
        # System
        self.fb = params['fb'] * 1e9
        self.nspui = params['M']
        self.L = params['L']
        self.fstep = params['fstep']
        # Tx EQ
        self.tr = params['T_r']
        self.tx_taps_min = params['tx_taps_min']
        self.tx_taps_max = params['tx_taps_max']
        self.tx_taps_step = params['tx_taps_step']
        assert len(self.tx_taps_min) == len(self.tx_taps_max) == len(self.tx_taps_step), ValueError(
            f"The lengths of keys: tx_taps_min, tx_taps_max, and tx_taps_step, must match!")
        self.c0_min = params['c0_min']
        # Rx EQ
        self.bmin = params['dfe_min']
        self.bmax = params['dfe_max']
        assert len(self.bmin) == len(self.bmax), ValueError(
            f"The lengths of keys: dfe_min, and dfe_max, must match!")
        self.Nb = len(self.bmin)
        self.rx_taps_min = params['rx_taps_min']
        self.rx_taps_max = params['rx_taps_max']
        assert len(self.rx_taps_min) == len(self.rx_taps_max), ValueError(
            f"The lengths of keys: `rx_taps_min` and `rx_taps_max` must match!")
        self.nRxTaps = len(self.rx_taps_min)
        self.nRxPreTaps = params['dw']
        self.fr = params['f_r']
        self.fz = params['f_z'] * 1e9
        self.fp1 = params['f_p1'] * 1e9
        self.fp2 = params['f_p2'] * 1e9
        self.fLF = params['f_LF'] * 1e9
        self.gDC_vals = params['g_DC']
        self.gDC2_vals = params['g_DC2']
        # Package
        self.Rd = params['R_d']
        self.R0 = params['R_0']
        self.Cd = list(map(lambda x: x / 1e12, params['C_d']))
        self.Cb = params['C_b'] / 1e12
        self.Cp = params['C_p'] / 1e12
        self.Ls = list(map(lambda x: x / 1e9,  params['L_s']))
        self.Av = params['A_v']
        self.Afe = params['A_fe']
        self.Ane = params['A_ne']
        self.DER0 = params['DER_0']
        self.sigma_Rj = params['sigma_Rj']
        self.Add = params['A_DD']
        self.eta0 = params['eta_0']
        self.TxSNR = params['SNR_TX']
        self.zc = params['z_c']
        self.zp_vals = params['z_p']
        self.gamma0 = params['gamma0']
        self.a1 = params['a1']
        self.a2 = params['a2']
        self.tau = params['tau']

        # Stash input parameters, for future reference.
        self.params = params

    # General functions
    def sC(self, c: float) -> rf.Network:
        """
        Return the 2-port network corresponding to a shunt capacitance,
        according to (93A-8).

        Args:
            c: Value of shunt capacitance (F).

        Returns:
            2-port network equivalent to shunt capacitance, calculated at given frequencies.

        Raises:
            None
        """

        r0 = self.R0
        freqs = self.freqs
        w = TWOPI * freqs
        s = 1j * w
        s2p = np.array(
            [1 / (2 + _s * c * r0) * np.array(
                [[-_s * c * r0, 2],
                 [2, -_s * c * r0]])
             for _s in s])
        return rf.Network(s=s2p, f=freqs, z0=[2 * r0, 2 * r0])

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
            2. After this step, the package and R0/Rd mismatch have been accounted for,
                but not the EQ.
        """

        assert s2p.s[0].shape == (2, 2), ValueError("I can only convert 2-port networks.")
        s2p = s2p.extrapolate_to_dc()
        s2p.interpolate_self(self.freqs)
        g1 = self.gamma1
        g2 = self.gamma2
        s11 = s2p.s11.s.flatten()
        s12 = s2p.s12.s.flatten()
        s21 = s2p.s21.s.flatten()
        s22 = s2p.s22.s.flatten()
        dS = s11 * s22 - s12 * s21
        return (s21 * (1 - g1) * (1 + g2)) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * dS)

    def H(self, s2p: rf.Network, tx_taps: Optional[Rvec] = None,
          gDC: Optional[float] = None, gDC2: Optional[float] = None,
          rx_taps: Optional[Rvec] = None, dfe_taps: Optional[Rvec] = None) -> Cvec:
        """
        Return the voltage transfer function, H(f), of a complete COM signal path,
        according to (93A-19).

        Args:
            s2p: Two port network of interest.

        Keyword Args:
            tx_taps: Tx FFE tap weights.
                Default: None (i.e. - Use `self.tx_taps`.)
            gDC: CTLE first stage d.c. gain (dB).
                Default: None (i.e. - Use `self.gDC`.)
            gDC2: CTLE second stage d.c. gain (dB).
                Default: None (i.e. - Use `self.gDC2`.)
            rx_taps: Rx FFE tap weights.
                Default: None (i.e. - Use `self.rx_taps`.)
            dfe_taps: Rx DFE tap weights.
                Default: None (i.e. - Use `self.dfe_taps`.)

        Returns:
            Complex voltage transfer function of complete path.

        Raises:
            ValueError: If given network is not two port.

        Notes:
            1. It is in this processing step that linear EQ is first applied.
            2. Any unprovided EQ values are taken from the `COM` instance.
                If you really want to omit a particular EQ component then call with:

                - `tx_taps`/`rx_taps`: []
                - `gDC`/`gDC2`: 0
        """

        assert s2p.s[0, :, :].shape == (2, 2), ValueError(
            f"I can only convert 2-port networks. {s2p}")

        gDC     = gDC     or self.gDC
        gDC2    = gDC2    or self.gDC2
        if tx_taps is None:
            tx_taps = self.tx_taps
        if rx_taps is None:
            rx_taps = self.rx_taps
        if dfe_taps is None:
            dfe_taps = self.dfe_taps
        Htx  = calc_Hffe(np.array(tx_taps).flatten(), self.tx_n_post)
        H21  = self.H21(s2p)
        Hr   = self.Hr
        Hctf = self.calc_Hctf(gDC=gDC, gDC2=gDC2)
        Hrx  = calc_Hffe(np.array(rx_taps).flatten(), self.nRxTaps - self.nRxPreTaps - 1, isRx=True)
        Hdfe = calc_Hdfe(dfe_taps)
        rslt = Htx * H21 * Hr * Hctf * Hrx  # * Hdfe
        if max(abs(rslt)) == 0:
            print(f"max(abs(Htx)): {max(abs(Htx))}")
            print(f"max(abs(H21)): {max(abs(H21))}")
            print(f"max(abs(Hr)): {max(abs(Hr))}")
            print(f"max(abs(Hctf)): {max(abs(Hctf))}")
            print(f"Hrx: {Hrx}")
            # print(f"max(abs(Hrx)): {max(abs(Hrx))}")
            raise

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
        Ts = self.t_irfft[1]
        p = np.fft.irfft(Xsinc * H)
        p_mag = np.abs(p)
        try:
            p_beg = np.where(p_mag > 0.01 * max(p_mag))[0][0] - int(5 * self.ui / Ts)  # Give it some "front porch".
        except Exception:
            print(f"max(p_mag): {max(p_mag)}; len(p_mag): {len(p_mag)}")
            print(f"max(abs(H)): {max(abs(H))}; len(H): {len(H)}")
            raise
        spln = interp1d(self.t_irfft, np.roll(p, -p_beg))  # `p` is not yet in our system time domain!
        return spln(self.times)                            # Now, it is.

    def gen_pulse_resps(
        self, ntwks: Optional[list[tuple[rf.Network, str]]] = None,
        gDC: Optional[float] = None, gDC2: Optional[float] = None,
        tx_taps: Optional[Rvec] = None, rx_taps: Optional[Rvec] = None,
        dfe_taps: Optional[Rvec] = None, apply_eq: bool = True
    ) -> list[Rvec]:
        """
        Generate pulse responses for all networks.

        Keyword Args:
            ntwks: The list of networks to generate pulse responses for.
                Default: None (i.e. - Use `self.chnls`.)
            gDC: Rx CTLE first stage d.c. gain.
                Default: None (i.e. - Use `self.gDC`.)
            gDC2: Rx CTLE second stage d.c. gain.
                Default: None (i.e. - Use `self.gDC2`.)
            tx_taps: Desired Tx tap weights.
                Default: None (i.e. - Use `self.tx_taps`.)
            rx_taps: Desired Rx FFE tap weights.
                Default: None (i.e. - Use `self.rx_taps`.)
            dfe_taps: Desired Rx DFE tap weights.
                Default: None (i.e. - Use `self.dfe_taps`.)
            apply_eq: Include linear EQ when True; otherwise, exclude it.
                (Allows for pulse response generation of terminated, but unequalized, channel.)
                Default: True

        Returns:
            List of pulse responses.

        Raises:
            None

        Notes:
            1. Assumes `self.gDC`, `self.gDC2`, `self.tx_taps`, `self.rx_taps`, and `self.dfe_taps`
                have been set correctly, if the equivalent function parameters have not been provided.
            2. To generate pulse responses that include all linear EQ except the Rx FFE/DFE
                (i.e. - pulse responses suitable for Rx FFE/DFE tap weight optimization, via either `optimize.przf()` or `optimize.mmse()`)
                set `rx_taps` equal to: [1.0] and `dfe_taps` equal to: [].
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

        pulse_resps = []
        for ntwk, ntype in ntwks:
            if apply_eq:
                if ntype == 'NEXT':
                    pr = self.pulse_resp(self.H(ntwk, np.zeros(tx_taps.shape), gDC=gDC, gDC2=gDC2, rx_taps=rx_taps, dfe_taps=dfe_taps))
                else:
                    pr = self.pulse_resp(self.H(ntwk, tx_taps,                 gDC=gDC, gDC2=gDC2, rx_taps=rx_taps, dfe_taps=dfe_taps))
            else:
                pr = self.pulse_resp(self.H21(ntwk))

            if ntype == 'THRU':
                pr *= self.Av
            elif ntype == 'NEXT':
                pr *= self.Ane
            else:
                pr *= self.Afe

            pulse_resps.append(pr)

        return pulse_resps

    def filt_pr_samps(self, pr_samps: Rvec, As: float, rel_thresh: float = 0.001) -> Rvec:
        """
        Filter a list of pulse response samples for minimum magnitude.

        Args:
            pr_samps: The pulse response samples to filter.
            As: Signal amplitude, as per 93A.1.6.c.

        Keyword Args:
            rel_thresh: Filtration threshold (As).
                Default: 0.001 (i.e. - 0.1%, as per Note 2 of 93A.1.7.1)

        Returns:
            The subset of `pr_samps` passing filtration.
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

        Keyword Args:
            rel_thresh: Filtration threshold (As).
                Default: 0.001 (i.e. - 0.1%, as per Note 2 of 93A.1.7.1)

        Returns:
            The calculated slopes around the valid samples.
        """

        M = self.nspui
        thresh = As * rel_thresh
        valid_pr_samp_ixs = np.array(list(filter(lambda ix: abs(pulse_resp[ix]) >= thresh,
                                                 range(cursor_ix, len(pulse_resp) - 1, M))))
        m1s = pulse_resp[valid_pr_samp_ixs - 1]
        p1s = pulse_resp[valid_pr_samp_ixs + 1]
        return (p1s - m1s) / (2 / M)  # (93A-28)

    def loc_curs(self, pulse_resp: Rvec, max_range: int = 1, eps: float = 0.001) -> int:
        """
        Locate the cursor position for the given pulse response,
        according to (93A-25) and (93A-26) (i.e. - Muller-Mueller criterion).

        Args:
            pulse_resp: The pulse response of interest.

        Keyword Args:
            max_range: The search radius, from the peak (UI).
                Default: 1
            eps: Threshold for declaring floating point value to be zero.
                Default: 0.001

        Returns:
            The index in the given pulse response vector of the cursor.

        Notes:
            1. As per v3.70 of the COM MATLAB code, we only minimize the
                residual of (93A-25); we don't require solving it exactly.
                (We do, however, give priority to exact solutions.)
        """

        M = self.nspui
        dfe_max = self.bmax
        dfe_min = self.bmin

        # Minimize Muller-Mueller criterion, within `max_range` of peak,
        # giving priority to exact solutions, as per the spec.
        peak_loc = np.argmax(pulse_resp)
        res_min = 1e6
        zero_res_ixs = []
        for ix in range(peak_loc - M * max_range, peak_loc + M * max_range):
            # Anticipate the DFE first tap value, observing its limits:
            b_1 = min(dfe_max[0],
                      max(dfe_min[0],
                          pulse_resp[ix + M] / pulse_resp[ix]))                          # (93A-26)
            # And include the effect of that tap when checking the Muller-Mueller condition:
            res = abs(pulse_resp[ix - M] - (pulse_resp[ix + M] - b_1 * pulse_resp[ix]))  # (93A-25)
            if res < eps:        # "Exact" match?
                zero_res_ixs.append(ix)
            elif res < res_min:  # Keep track of best solution, in case no exact matches.
                ix_best = ix
                res_min = res
        if len(zero_res_ixs):  # Give priority to "exact" matches if there were any.
            pre_peak_ixs = list(filter(lambda x: x <= peak_loc, zero_res_ixs))
            if len(pre_peak_ixs):
                return pre_peak_ixs[-1]  # Standard says to use first one prior to peak in event of multiple solutions.
            return zero_res_ixs[0]       # They're all post-peak; so, return first (i.e. - closest to peak).
        return ix_best                   # No exact solutions; so, return that which yields minimum error.

    def calc_fom(self, tx_taps: Rvec,
                 gDC: Optional[float] = None, gDC2: Optional[float] = None,
                 rx_taps: Optional[Rvec] = None) -> float:
        """
        Calculate the *figure of merit* (FOM), given the existing linear EQ settings.
        Optimize Rx FFE taps if they aren't specified by caller.

        Args:
            tx_taps: The Tx FFE tap weights, excepting the cursor.
                (The cursor takes whatever is left.)

        Keyword Args:
            gDC: CTLE first stage d.c. gain.
                Default: None (i.e. - Use `self.gDC`.)
            gDC2: CTLE second stage d.c. gain.
                Default: None (i.e. - Use `self.gDC2`.)
            rx_taps: Rx FFE tap weight overrides.
                Default: None (i.e. - Optimize Rx FFE tap weights.)

        Returns:
            The resultant figure of merit.

        Raises:
            None

        Notes:
            1. See: IEEE 802.3-2022 93A.1.6.
            2. When not provided, the values for `gDC` and `gDC2` are taken from the `COM` instance.
            3. Unlike other member functions of the ``COM`` class,
                this function _optimizes_ the Rx FFE tap weights when they are not provided.
        """

        L = self.L
        M = self.nspui
        freqs = self.freqs
        chnls = self.chnls
        nDFE = len(self.bmin)

        # Step a - Pulse response construction.
        if rx_taps is None:
            pulse_resps = self.gen_pulse_resps(chnls, tx_taps=np.array(tx_taps), gDC=gDC, gDC2=gDC2, rx_taps=[1.0])  # ToDo: Check this.
            rx_taps, dfe_taps, _ = przf(pulse_resps[0], M, self.nRxTaps, self.nRxPreTaps, nDFE, self.rx_taps_min, self.rx_taps_max, self.bmin, self.bmax)
        pulse_resps = self.gen_pulse_resps(chnls, tx_taps=np.array(tx_taps), gDC=gDC, gDC2=gDC2, rx_taps=np.array(rx_taps))

        # Step b - Cursor identification.
        try:
            vic_pulse_resp = np.array(pulse_resps[0])
        except Exception:
            print(len(pulse_resps), len(chnls))
            raise
        vic_peak_loc = np.argmax(vic_pulse_resp)
        cursor_ix = self.loc_curs(vic_pulse_resp)

        # Step c - As.
        vic_curs_val = vic_pulse_resp[cursor_ix]
        As = self.RLM * vic_curs_val / (L - 1)

        # Step d - Tx noise.
        varX = (L**2 - 1) / (3 * (L - 1)**2)  # (93A-29)
        varTx = vic_curs_val**2 * pow(10, -self.TxSNR / 10)  # (93A-30)

        # Step e - ISI.
        # This is not compliant to the standaard, but is consistent w/ v2.60 of MATLAB code.
        n_pre = cursor_ix // M
        first_pre_ix = cursor_ix - n_pre * M
        vic_pulse_resp_isi_samps = np.concatenate((vic_pulse_resp[first_pre_ix:cursor_ix:M],
                                                   vic_pulse_resp[cursor_ix + M::M]))
        vic_pulse_resp_post_samps = vic_pulse_resp_isi_samps[n_pre:]
        dfe_tap_weights = np.maximum(  # (93A-26)
            self.bmin,
            np.minimum(
                self.bmax,
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
        varJ = (self.Add**2 + self.sigma_Rj**2) * varX * sum(hJ**2)  # (93A-32)

        # Step g - Crosstalk.
        varXT = 0
        for pulse_resp in pulse_resps[1:]:  # (93A-34)
            varXT += max([sum(np.array(self.filt_pr_samps(pulse_resp[m::M], As))**2) for m in range(M)])  # (93A-33)
        varXT *= varX

        # Step h - Spectral noise.
        df = freqs[1]
        # varN = self.eta0 * sum(abs(self.Hr * self.calc_Hctf(gDC=gDC, gDC2=gDC2))**2) * (df / 1e9)  # (93A-35)
        varN = (self.eta0 / 1e9) * sum(abs(self.Hr * self.calc_Hctf(gDC=gDC, gDC2=gDC2))**2) * df

        # Step i - FOM calculation.
        # fom = 10 * np.log10(As**2 / (varTx + varISI + varJ + varXT + varN))  # (93A-36)
        fom = -10 * np.log10(varTx + varISI + varJ + varXT + varN)  # Assumes unit peak victim pulse response amplitude.

        # Stash our calculation results.
        self.fom_rslts['pulse_resps'] = pulse_resps
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

        return fom

    def opt_eq(
        self, do_opt_eq: bool = True, tx_taps: Optional[Rvec] = None,
        opt_mode: OptMode = OptMode.PRZF
    ) -> bool:
        """
        Find the optimum values for the linear equalization parameters:
        c[n], gDC, gDC2, and w[n] as per IEEE 802.3-22 93A.1.6
        (or, [1] slide 11 if MMSE has been chosen).

        Keyword Args:
            do_opt_eq: Perform optimization of linear EQ when True.
                Default: True
            tx_taps: Used when `do_opt_eq` = False.
                Default: None
            opt_mode: Optimization mode.
                Default: PRZF

        Returns:
            True if no errors encountered; False otherwise.
        """

        if do_opt_eq:
            # Run the nested optimization loops.
            def check_taps(tx_taps: Rvec, t0_min: float = self.c0_min) -> bool:
                if (1 - sum(abs(np.array(tx_taps)))) < t0_min:
                    return False
                else:
                    return True

            fom_max = -1000.0
            fom_max_changed = False
            foms = []
            for gDC2 in self.gDC2_vals:
                for gDC in self.gDC_vals:
                    for tx_taps in self.tx_combs:
                        if not check_taps(np.array(tx_taps)):
                            continue
                        match opt_mode:
                            case OptMode.PRZF:
                                fom = self.calc_fom(tx_taps, gDC=gDC, gDC2=gDC2)
                                rx_taps = self.fom_rslts['rx_taps']
                            case OptMode.MMSE:
                                pulse_resps = self.gen_pulse_resps(tx_taps=np.array(tx_taps), gDC=gDC, gDC2=gDC2, rx_taps=[1.0], dfe_taps=[])
                                theNoiseCalc = NoiseCalc(
                                    self.L, 1/self.fb, 0, self.times, pulse_resps[0], pulse_resps[1:],
                                    self.freqs, self.Ht, self.H21(self.chnls[0][0]), self.Hr, self.calc_Hctf(gDC, gDC2),
                                    self.eta0, self.Av, self.TxSNR, self.Add, self.sigma_Rj)
                                rslt = mmse(theNoiseCalc, self.nRxTaps, self.nRxPreTaps, self.Nb, self.RLM, self.L,
                                            self.bmin, self.bmax, self.rx_taps_min, self.rx_taps_max)
                                fom = rslt["fom"]
                                rx_taps = rslt["rx_taps"]
                                # In the PRZF branch, these are set by calc_fom().
                                self.fom_rslts['dfe_tap_weights'] = rslt["dfe_tap_weights"]
                                self.fom_rslts['vic_pulse_resp'] = rslt["vic_pulse_resp"]
                                self.fom_rslts['cursor_ix'] = rslt["cursor_ix"]
                                self.fom_rslts['As'] = self.RLM * rslt["vic_pulse_resp"][rslt["cursor_ix"]] / (self.L - 1)
                                self.fom_rslts['varTx'] = rslt["varTx"]
                                self.fom_rslts['varISI'] = rslt["varISI"]
                                self.fom_rslts['varJ'] = rslt["varJ"]
                                self.fom_rslts['varXT'] = rslt["varXT"]
                                self.fom_rslts['varN'] = rslt["varN"]
                                self.fom_rslts['mse'] = rslt['mse'] if 'mse' in rslt else 0
                            case _:
                                raise ValueError(f"Unrecognized optimization mode: {opt_mode}, requested!")                    
                        foms.append(fom)
                        if fom > fom_max:
                            fom_max_changed = True
                            fom_max = fom
                            gDC2_best = gDC2
                            gDC_best = gDC
                            tx_taps_best = tx_taps
                            rx_taps_best = rx_taps
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
                            # TEMPORARY DEBUGGING ONLY!:
                            if opt_mode == OptMode.MMSE:
                                self.mmse_rslt = rslt
                                self.theNoiseCalc = theNoiseCalc
        else:
            assert tx_taps, RuntimeError("You must define `tx_taps` when setting `do_opt_eq` False!")
            fom = self.calc_fom(tx_taps)
            foms = [fom]
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
            return False
        self.fom = fom_max
        self.mse = mse_best
        self.gDC2 = gDC2_best
        self.gDC = gDC_best
        self.tx_taps = tx_taps_best
        self.rx_taps = rx_taps_best
        self.fom_tx_taps = print_taps(tx_taps_best)
        self.fom_dfe_taps = print_taps(dfe_tap_weights_best)
        self.dfe_taps = dfe_tap_weights_best
        self.fom_cursor_ix = cursor_ix_best
        self.fom_As = As_best
        self.sigma_ISI = np.sqrt(varISI_best)
        self.sigma_J = np.sqrt(varJ_best)
        self.sigma_XT = np.sqrt(varXT_best)
        # These two are also calculated by `calc_noise()`, but are not overwritten.
        self.sigma_Tx = np.sqrt(varTx_best)
        self.sigma_N = np.sqrt(varN_best)
        self.foms = foms
        self.plotdata.set_data("pulse_eq", vic_pulse_resp)
        self.vic_pulse_resp = vic_pulse_resp
        self.rslts['fom'] = fom_max
        self.rslts['gDC'] = gDC_best
        self.rslts['gDC2'] = gDC2_best
        self.rslts['tx_taps'] = tx_taps_best[2:4]
        self.rslts['rx_taps'] = rx_taps_best
        self.rslts['dfe_taps'] = dfe_tap_weights_best
        self.rslts['sigma_ISI'] = self.sigma_ISI * 1e3
        self.rslts['sigma_XT'] = self.sigma_XT * 1e3
        self.rslts['sigma_J'] = self.sigma_J * 1e3
        return True

    def calc_noise(self, cursor_ix: Optional[int] = None) -> tuple[float, float, int]:
        """
        Calculate the interference and noise for COM.

        Keyword Args:
            cursor_ix: An optional predetermined cursor index,
                to be used instead of our own estimate.
                (In support of MMSE.)
                Default: None

        Returns:
            - signal amplitude
            - noise + interference amplitude (V)
            - cursor location within victim pulse response vector

        Notes:
            1. Assumes the following instance variables have been set:
                - gDC
                - gDC2
                - tx_taps
                - rx_taps
                (This assumption is embedded into the `gen_pulse_resps()` function.)
            2. Warns if `2*As/npts` rises above 10 uV, against standard's recommendation.
        """

        L = self.L
        M = self.nspui
        RLM = self.RLM
        freqs = self.freqs
        nDFE = len(self.bmin)

        # pulse_resps = self.gen_pulse_resps(self.chnls, np.array(self.tx_taps))  # ToDo: Add rx_taps.
        self.set_status("Calculating COM...")
        pulse_resps = self.gen_pulse_resps(rx_taps=[1.0], dfe_taps=[])  # No Rx FFE/DFE.
        self.vic_pulse_resp_noRxFFE = pulse_resps[0]
        match self.opt_mode:
            case OptMode.PRZF:
                rx_taps, dfe_taps, pr_samps = przf(pulse_resps[0], M, self.nRxTaps, self.nRxPreTaps, nDFE, self.rx_taps_min, self.rx_taps_max, self.bmin, self.bmax)
            case OptMode.MMSE:
                theNoiseCalc = NoiseCalc(
                    L, 1/self.fb, 0, self.times, pulse_resps[0], pulse_resps[1:],
                    self.freqs, self.Ht, self.H21(self.chnls[0][0]), self.Hr, self.Hctf,
                    self.eta0, self.Av, self.TxSNR, self.Add, self.sigma_Rj)
                rslt = mmse(theNoiseCalc, self.nRxTaps, self.nRxPreTaps, self.Nb, RLM, L,
                            self.bmin, self.bmax, self.rx_taps_min, self.rx_taps_max)
                rx_taps  = rslt["rx_taps"]
                dfe_taps = rslt["dfe_tap_weights"]
                pr_samps = rslt["h"]
            case _:
                raise ValueError(f"Unrecognized optimization mode: {self.opt_mode}, requested!")                    
        self.rx_taps  = rx_taps
        self.dfe_taps = dfe_taps
        self.pr_samps = pr_samps
        # pulse_resps = self.gen_pulse_resps()  # Include all EQ.
        pulse_resps = self.gen_pulse_resps(dfe_taps=[])  # DFE taps are included explicitly, below.
        vic_pulse_resp = pulse_resps[0]
        if cursor_ix is None:
            cursor_ix = self.loc_curs(vic_pulse_resp)
        curs_uis, curs_ofst = divmod(cursor_ix, M)
        vic_curs_val = vic_pulse_resp[cursor_ix]
        As = RLM * vic_curs_val / (L - 1)
        npts = 2 * max(int(As / 0.001), 1_000) + 1  # Note 1 of 93A.1.7.1; MUST BE ODD!
        y = np.linspace(-As, As, npts)
        ystep = 2 * As / (npts - 1)

        delta = np.zeros(npts)
        delta[npts // 2] = 1
        varX = (L**2 - 1) / (3 * (L - 1)**2)  # (93A-29)
        df = freqs[1] - freqs[0]

        # Sec. 93A.1.7.2
        # ToDo: `pG` and `pN` are both very large, relative to `pJ`; am I mixing PDFs w/ PMFs?
        varN = self.eta0 * sum(abs(self.Hr * self.Hctf)**2) * (df / 1e9)    # (93A-35)
        varTx = vic_curs_val**2 * pow(10, -self.TxSNR / 10)                 # (93A-30)
        hJ = self.calc_hJ(vic_pulse_resp, As, cursor_ix)
        _, pJ = delta_pmf(self.Add * hJ, L=L, RLM=RLM, y=y)
        self.dbg['pJ'] = pJ
        self.dbg['hJ'] = hJ
        varG = varTx + self.sigma_Rj**2 * varX * sum(hJ**2) + varN          # (93A-41)
        pG = np.exp(-y**2 / (2 * varG)) / np.sqrt(TWOPI * varG) * ystep     # (93A-42), but converted to PMF.
        pN = np.convolve(pG, pJ, mode='same')                               # (93A-43)

        # Sec. 93A.1.7.3
        self.set_status("Sec. 93A.1.7.3")
        # - ISI (Inconsistent w/ IEEE 802.3-22, but consistent w/ v2.60 of MATLAB code.)
        n_pre = min(5, curs_uis)
        isi_sample_slice = slice(curs_ofst, len(vic_pulse_resp), M)  # Sample every M points, such that we include our identified cursor sample.
        isi_select_slice = slice(curs_uis - n_pre, curs_uis + 101)   # Ignore everything beyond 100 UI after cursor.
        tISI = self.times[isi_sample_slice][isi_select_slice]
        hISI = vic_pulse_resp[isi_sample_slice][isi_select_slice].copy()
        hISI[n_pre] = 0  # No ISI at cursor.
        dfe_slice = slice(n_pre + 1, n_pre + 1 + nDFE)
        dfe_tap_weights = np.maximum(  # (93A-26)
            self.bmin,
            np.minimum(
                self.bmax,
                (hISI[dfe_slice] / vic_curs_val)))
        hISI[dfe_slice] -= dfe_tap_weights * vic_curs_val
        _, py = delta_pmf(hISI, L=L, RLM=RLM, y=y)  # `hISI` from (93A-27); `p(y)` as per (93A-40)
        varISI = varX * sum(hISI**2)  # (93A-31)
        self.com_tISI = tISI
        self.com_hISI = hISI

        # - Crosstalk
        self.rslts['py0'] = py.copy()  # For debugging.
        xt_samps = []
        pks = []  # For debugging.
        for pulse_resp in pulse_resps[1:]:  # (93A-44)
            i = np.argmax([sum(np.array(pulse_resp[m::M])**2) for m in range(M)])  # (93A-33)
            samps = pulse_resp[i::M]
            xt_samps.append(samps)
            _, pk = delta_pmf(samps, L=L, RLM=RLM, y=y)  # For debugging.
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
        self.pulse_resps   = pulse_resps
        self.com_cursor_ix = cursor_ix
        self.com_sigma_Tx  = np.sqrt(varTx)
        self.com_sigma_G   = np.sqrt(varG)
        self.com_sigma_N   = np.sqrt(varN)
        self.com_sigma_ISI = np.sqrt(varISI)
        self.rslts['pG'] = pG
        self.rslts['pN'] = pN
        self.rslts['py'] = py
        self.rslts['Py'] = Py
        self.rslts['y']  = y
        self.com_dfe_taps = print_taps(dfe_tap_weights)
        self.dfe_taps = dfe_tap_weights
        self.com_As = As
        self.rslts['sigma_G']   = self.com_sigma_G  * 1e3
        self.rslts['sigma_Tx']  = self.com_sigma_Tx * 1e3
        self.rslts['sigma_N']   = self.com_sigma_N  * 1e3
        self.rslts['sigma_ISI'] = self.com_sigma_ISI  * 1e3

        return (As,
        # return (1.0,
                abs(np.where(Py >= self.DER0)[0][0] - npts // 2) * ystep,
                cursor_ix)

if __name__ == "__main__":
    from pychopmarg.cli import cli
    cli()
