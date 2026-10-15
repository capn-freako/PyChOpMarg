"""
COM parameter definition template for *PyChOpMarg*

Original author: David Banas <capn.freako@gmail.com>

Original date:   November 6 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from pychopmarg.common import Rvec


# @dataclass(frozen=True)  # We take COM parameters to be immutable.
@dataclass()
class COMParams:  # pylint: disable=too-many-instance-attributes
    """
    Template definition for COM parameters.

    ToDo:
        1. Use `metadata` to provide units:

            .. code-block:: python

                @dataclass
                class Product:
                    name: str
                    price: float = field(metadata={"unit": "USD"})

        2. Is there a `helptext` field available in `metadata`?

        3. If we ever want to exclude some fields from comparison operations:

            .. code-block:: python

                @dataclass
                class CustomTask:
                    name: str
                    priority: int = field(compare=False)
    """

    # General
    fb: float                   # (GBaud)
    fstep: float                # (GHz)
    L: int                      # modulation levels
    M: int                      # samples per UI
    DER_0: float                # detector error ratio
    T_r: float                  # Tx output driver risetime (ns)
    RLM: float                  # relative level mismatch
    A_v: float                  # victim amplitude (V)
    A_fe: float                 # far end aggressor amplitude (V)
    A_ne: float                 # near end aggressor amplitude (V)
    R_0: float                  # system reference impedance (Ohms)
    # Noise
    A_DD: float                 # (UI)
    SNR_TX: float               # (dB)
    eta_0: float                # (V^2/GHz)
    sigma_Rj: float             # (UI)
    # CTLE
    f_z: float                  # (GHz)
    f_p1: float                 # (GHz)
    f_p2: float                 # (GHz)
    f_LF: float                 # (GHz)
    g_DC: list[float]           # (dB)
    g_DC2: list[float]          # (dB)
    # Tx FFE
    tx_taps_min: list[float]
    tx_taps_max: list[float]
    tx_taps_step: list[float]
    c0_min: float
    # Rx EQ
    f_r: float                  # (fb)
    # dfe_min: list[float]
    # dfe_max: list[float]
    dfe_min: Rvec
    dfe_max: Rvec
    # rx_taps_min: list[float]
    # rx_taps_max: list[float]
    rx_taps_min: Rvec
    rx_taps_max: Rvec
    dw: int
    # Die & Package (Class A Test 1) Sec. 178.10.1
    # R_d: list[float]            # (Ohms); as per config_com_ieee8023_93a=100GBASE-KR4.xls
    R_d: Rvec                   # (Ohms); as per config_com_ieee8023_93a=100GBASE-KR4.xls
    C_d: list[float]            # (nF)
    C_b: list[float]            # (nF)
    C_p: list[float]            # (nF)
    L_s: list[float]            # (nH)
    z_c: list[float]            # (Ohms)
    z_p: list[float]            # (mm)
    gamma0: float               # (1/mm)
    a1: float                   # (sqrt_ns/mm)
    a2: float                   # (ns/mm)
    tau: float                  # (ns/mm)
    # New as of 802.3dj
    z_pB: float = 1.8           # (mm)
    # Per-end package segment lengths (mm), for the selected test case (MATLAB COM's "z_p (TX)", etc.):
    # one per package transmission line segment; `None` means ``[z_p[zp_sel], z_pB]``.
    zp_tx: Optional[list[float]] = None
    zp_rx: Optional[list[float]] = None
    zp_next: Optional[list[float]] = None
    zp_fext: Optional[list[float]] = None
    # Every package test case to evaluate (MATLAB COM's "z_p select"): each a dict of any of "zp_tx", "zp_rx", "zp_next",
    # and "zp_fext" (as above), overriding those fields. COM is the lowest over them, as in MATLAB COM.
    # `None`: one case, given by `zp_tx`, etc.
    zp_cases: Optional[list[dict[str, list[float]]]] = None
    # EQ search control (MATLAB COM's "I/O control" settings of the same names)
    local_search: int = 0           # "Local Search": EQ grid search radius (grid steps); 0 = full search.
    ts_srch_mode: str = "full-sweep"  # "TS_SRCH_MODE": cursor timing search, "full-sweep" or "middle".
    ts_track: bool = False          # (PyChOpMarg only) Centre each "middle" timing search on the best so far.
    # Quantization noise (178A.1.7.6), as MATLAB COM's "N_qb" and "P_qc"
    N_qb: int = 0                   # noise-equivalent quantization bits; 0 = no quantization noise
    P_qc: float = 1e-7              # quantization clipping probability
    # Floating Rx FFE taps (178A.1.8.1): with `Nwg` > 0, `rx_taps_min`/`rx_taps_max` span every tap position up to
    # `Nmax` (their length), of which the first `Nfix` are fixed and the rest available to `Nwg` groups of `Nwf`.
    Nfix: Optional[int] = None      # number of fixed-position taps; `None`: all
    Nwg: int = 0                    # number of floating tap groups
    Nwf: int = 0                    # number of taps per floating tap group
    # MLSD reference receiver (178A.1.10), as MATLAB COM's "MLSD"/"MLSE" and "add_rx_noise"
    mlsd: bool = False              # include MLSD
    dCOM_an: float = 3.0            # penalty for added receiver noise (dB); D3.3: the minimum COM limit
    # Host boards (MATLAB COM's "Include PCB"), added between each package and the channel: C_0 at the package,
    # a transmission line, and C_1 at the channel (as `devmodel.PartialHost`). Defaults are MATLAB COM's.
    include_pcb: int = 0            # 0: none; 1: at both ends; 2: at the receiver only
    zbp_tx: float = 151.0           # victim transmitter board length (mm)
    zbp_rx: float = 151.0           # victim receiver board length (mm)
    zbp_next: float = 72.0          # NEXT aggressor transmitter board length (mm)
    zbp_fext: float = 72.0          # FEXT aggressor transmitter board length (mm)
    brd_Z_c: list[float] = field(default_factory=lambda: [109.8, 109.8])  # (Ohms) [Tx, Rx]
    brd_gamma0_a1_a2: list[float] = field(default_factory=lambda: [0.0, 4.114e-4, 2.547e-4])  # (1/mm, ...)
    brd_tau: float = 6.191e-3       # (ns/mm)
    brd_C0: list[float] = field(default_factory=lambda: [0.0, 0.0])  # (nF) [Tx, Rx]
    brd_C1: list[float] = field(default_factory=lambda: [0.0, 0.0])  # (nF) [Tx, Rx]

    def __post_init__(self):  # pylint: disable=too-many-statements
        "Validate incoming COM parameters."

        def check_range(
            x_name: str,
            x_val: float,
            x_min: float,
            x_max: float
        ) -> None:
            "Check range of a scalar, or of every value in a (possibly nested) list."

            _x_min = float(x_min)
            _x_max = float(x_max)

            def chk_rng(nm, val):
                try:
                    _x_val = float(val)
                    assert _x_min <= _x_val <= _x_max, ValueError(
                        f"`{nm}` ({_x_val:.3g}) is out of range ([{_x_min:.3g}, {_x_max:.3g}])!")
                except Exception as err:
                    raise ValueError(
                        "\n\t".join([
                            f"Error converting `{nm}` ({val}):",
                            str(err)])) from err

            def chk_all(nm, val):  # Lists may nest (e.g. one row of `C_d` per device).
                if isinstance(val, (list, tuple, np.ndarray)):
                    for n, x in enumerate(val):
                        chk_all(f"{nm}[{n}]", x)
                else:
                    chk_rng(nm, val)

            chk_all(x_name, x_val)

        check_range("fb", self.fb, 10, 300)
        check_range("fstep", self.fstep, 0.001, 1)
        check_range("L", self.L, 2, 16)
        check_range("M", self.M, 32, 256)
        check_range("DER_0", self.DER_0, 1e-12, 1e-2)
        check_range("T_r", self.T_r, 0.001, 0.050)
        check_range("RLM", self.RLM, 0.7, 1.0)
        check_range("A_v", self.A_v, 0.2, 1.0)
        check_range("A_fe", self.A_fe, 0.1, 0.8)
        check_range("A_ne", self.A_ne, 0.1, 0.8)
        check_range("R_0", self.R_0, 25, 100)
        check_range("A_DD", self.A_DD, 0.0, 1.0)
        check_range("SNR_TX", self.SNR_TX, 25, 35)
        check_range("eta_0", self.eta_0, 1e-10, 1e-6)
        check_range("sigma_Rj", self.sigma_Rj, 0.0, 1.0)
        check_range("N_qb", self.N_qb, 0, 16)
        assert self.include_pcb in (0, 1, 2), ValueError(f"`include_pcb` ({self.include_pcb}) must be 0, 1, or 2!")
        if self.Nwg:
            assert self.Nfix is not None and self.dw < self.Nfix and self.Nwf > 0 \
                and self.Nfix + self.Nwg * self.Nwf <= len(self.rx_taps_max), ValueError(
                    f"Floating Rx FFE taps (`Nwg` = {self.Nwg}, `Nwf` = {self.Nwf}) need `Nfix` ({self.Nfix}) > `dw` "
                    f"({self.dw}), and room for them among the {len(self.rx_taps_max)} tap positions!")
        check_range("P_qc", self.P_qc, 1e-15, 0.1)
        check_range("f_z", self.f_z, 1.0, 1000)
        check_range("f_p1", self.f_p1, 1.0, 1000)
        check_range("f_p2", self.f_p2, 1.0, 1000)
        check_range("f_LF", self.f_LF, 0.1, 10.)
        check_range("g_DC", self.g_DC, -30, 0)
        check_range("g_DC2", self.g_DC2, -20, 0)
        check_range("tx_taps_min", self.tx_taps_min, -1, 1)
        check_range("tx_taps_max", self.tx_taps_max, -1, 1)
        check_range("tx_taps_step", self.tx_taps_step, 0., 0.25)  # 0 for a tap that isn't swept.
        assert len(self.tx_taps_min) == len(self.tx_taps_max) == len(self.tx_taps_step), ValueError(
            ", ".join([f"Lengths of Tx tap weight mins ({len(self.tx_taps_min)})",
                       f"maxs ({len(self.tx_taps_max)})",
                       f"and steps ({len(self.tx_taps_step)})",
                       "must be equal!"]))
        check_range("c0_min", self.c0_min, 0., 1.)
        check_range("f_r", self.f_r, 0.1, 1.0)
        check_range("dfe_min", self.dfe_min, -1, 1)
        check_range("dfe_max", self.dfe_max, -1, 1)
        assert len(self.dfe_min) == len(self.dfe_max), ValueError(
            f"Lengths of DFE tap weight mins ({len(self.dfe_min)}) and maxs ({len(self.dfe_max)}) must be equal!")
        check_range("rx_taps_min", self.rx_taps_min, -1, 1)
        check_range("rx_taps_max", self.rx_taps_max, -1, 1)
        assert len(self.rx_taps_min) == len(self.rx_taps_max), ValueError(
            " ".join([f"Lengths of Rx FFE tap weight mins ({len(self.rx_taps_min)})",
                      f"and maxs ({len(self.rx_taps_max)}) must be equal!"]))
        check_range("dw", self.dw, 0, len(self.rx_taps_max))
        check_range("R_d", self.R_d, 25, 100)
        check_range("C_d", self.C_d, 1e-6, 1e-2)  # (nF)
        check_range("C_b", self.C_b, 0., 1e-2)    # (nF)
        check_range("C_p", self.C_p, 1e-6, 1e-2)  # (nF)
        check_range("L_s", self.L_s, 0.01, 10.)
        check_range("z_c", self.z_c, 25, 100)
        check_range("z_p", self.z_p, 0.001, 100.)
        for nm in ("zp_tx", "zp_rx", "zp_next", "zp_fext"):
            if getattr(self, nm) is not None:
                check_range(nm, getattr(self, nm), 0.001, 100.)
        for n, case in enumerate(self.zp_cases or []):
            for nm, zp in case.items():
                assert nm in ("zp_tx", "zp_rx", "zp_next", "zp_fext"), ValueError(
                    f"`zp_cases[{n}]` has an unknown key: {nm}!")
                check_range(f"zp_cases[{n}][{nm}]", zp, 0.001, 100.)
        check_range("gamma0", self.gamma0, 1e-7, 1e-3)
        check_range("a1", self.a1, 1e-7, 1e-3)
        check_range("a2", self.a2, 1e-7, 1e-3)
        check_range("tau", self.tau, 1e-7, 1e-2)
        assert int(self.local_search) == self.local_search >= 0, ValueError(
            f"`local_search` ({self.local_search}) must be a non-negative integer!")
        self.local_search = int(self.local_search)
        assert self.ts_srch_mode in ("full-sweep", "middle"), ValueError(
            f"`ts_srch_mode` ('{self.ts_srch_mode}') must be either 'full-sweep' or 'middle'!")
