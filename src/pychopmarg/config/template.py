"""
COM parameter definition template for *PyChOpMarg*

Original author: David Banas <capn.freako@gmail.com>

Original date:   November 6 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class COMParams():  # pylint: disable=too-many-instance-attributes
    "Template definition for COM parameters."

    # General
    fb: float                   # (GBaud)
    fstep: float                # (Hz)
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
    dfe_min: list[float]
    dfe_max: list[float]
    rx_taps_min: list[float]
    rx_taps_max: list[float]
    dw: int
    # Die & Package (Class A Test 1) Sec. 178.10.1
    R_d: list[float]            # (Ohms); as per config_com_ieee8023_93a=100GBASE-KR4.xls
    C_d: list[float]            # (pF)
    C_b: list[float]            # (pF)
    C_p: list[float]            # (pF); as per config_com_ieee8023_93a=100GBASE-KR4.xls
    L_s: list[float]            # (nH)
    z_c: list[float]            # (Ohms)
    z_p: list[float]            # (mm)
    gamma0: float               # (1/mm)
    a1: float                   # (sqrt_ns/mm)
    a2: float                   # (ns/mm)
    tau: float                  # (ns/mm)
    # New as of 802.3dj
    z_pB: float = 1.8           # (mm)

    def __post_init__(self):
        "Validate incoming COM parameters."

        def check_range(
            x_name: str,
            x_val: int | float,
            x_min: int | float,
            x_max: int | float
        ) -> None:
            "Check range of scalar of list of values."

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
                            str(err)]))

            if isinstance(x_val, list) or isinstance(x_val, np.ndarray):
                map(lambda n_x: chk_rng(f"{x_name}[{n_x[0]}]", n_x[1]), enumerate(x_val))
            else:
                chk_rng(x_name, x_val)

        check_range("fb", self.fb, 10, 300)
        check_range("fstep", self.fstep, 1e6, 1000e6)
        check_range("L", self.L, 2, 16)
        check_range("M", self.M, 32, 256)
        check_range("DER_0", self.DER_0, 1e-12, 1e-2)
        check_range("T_r", self.T_r, 0.001, 0.050)
        check_range("RLM", self.RLM, 0.7, 1.0)
        check_range("A_v", self.A_v, 0.2, 1.0)
        check_range("A_fe", self.A_v, 0.1, 0.8)
        check_range("A_ne", self.A_v, 0.1, 0.8)
        check_range("R_0", self.R_0, 25, 100)
        check_range("A_DD", self.A_DD, 0.0, 1.0)
        check_range("SNR_TX", self.SNR_TX, 25, 35)
        check_range("eta_0", self.eta_0, 1e-10, 1e-6)
        check_range("sigma_Rj", self.sigma_Rj, 0.0, 1.0)
        check_range("f_z", self.f_z, 1.0, 2 * self.fb)
        check_range("f_p1", self.f_p1, 1.0, 2 * self.fb)
        check_range("f_p2", self.f_p2, 1.0, 5 * self.fb)
        check_range("f_LF", self.f_LF, 0.1, 10.)
        check_range("g_DC", self.g_DC, -30, 0)
        check_range("g_DC2", self.g_DC2, -20, 0)
        check_range("tx_taps_min", self.tx_taps_min, -1, 1)
        check_range("tx_taps_max", self.tx_taps_max, -1, 1)
        check_range("tx_taps_step", self.tx_taps_step, 0.005, 0.025)
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
        check_range("C_d", self.C_d, 0.01, 10.)
        check_range("C_b", self.C_b, 0., 10.)
        check_range("C_p", self.C_p, 0.01, 10.)
        check_range("L_s", self.L_s, 0.01, 10.)
        check_range("z_c", self.z_c, 25, 100)
        check_range("z_p", self.z_p, 0.001, 100.)
        check_range("gamma0", self.gamma0, 1e-7, 1e-3)
        check_range("a1", self.a1, 1e-7, 1e-3)
        check_range("a2", self.a2, 1e-7, 1e-3)
        check_range("tau", self.tau, 1e-7, 1e-2)
