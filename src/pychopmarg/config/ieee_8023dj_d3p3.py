"""
IEEE P802.3dj/D3.3 (24 September 2026) device termination, device package, and partial host channel models,
for the physical layers that use Annex 178A COM (Table 178A-1).

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 29, 2026

Copyright (c) 2026 David Banas; all rights reserved World wide.

Sources:
    - KR (200GBASE-KR1, ...):  Clause 178, Table 178-14.
    - C2C (200GAUI-1 C2C, ...): Annex 176C, Table 176C-9 (identical to Table 178-14).
    - CR (200GBASE-CR1, ...):  Clause 179, Tables 179-21, 179-22, and 179-23.
    - C2M (200GAUI-1 C2M, ...): Annex 176D, Table 176D-6.

For KR and C2C, COM is calculated with the package class(es) the channel is intended to support, twice (Test 1 and
Test 2), with different package transmission line 1 lengths (178.10.1, 176C.7.1). For CR, the transmitter and
receiver models are set by their host classes (HL, HN, HH), whose valid combinations depend on the cable assembly
class (179.11.4, Table 179-21).
"""

from dataclasses import replace

import numpy as np

from pychopmarg.config.ieee_8023dj import IEEE_8023dj
from pychopmarg.config.template import COMParams
from pychopmarg.devmodel import DeviceModel, DevicePackage, DeviceTermination, PartialHost, TlineStage

R0 = 46.25  # Single-ended reference resistance (Ohms); also Rd(t) and Rd(r), in every table below.
FB = 106.25  # Signaling rate (GBd).

# Device termination model: identical in Tables 178-14, 176C-9, 179-22, and 176D-6.
TERMINATION = DeviceTermination(Cd=(40e-6, 90e-6, 110e-6), Ls=(0.13, 0.15, 0.14), Cb=30e-6)
CP = 40e-6  # Single-ended package capacitance at package-to-board interface (nF).

# Device package transmission line parameters, per class: (gamma0, a1, a2, tau).
_CLASS_A = dict(gamma0=5e-4, a1=8.9e-4, a2=2e-4,    tau=6.141e-3)
_CLASS_B = dict(gamma0=5e-4, a1=6.5e-4, a2=2.93e-4, tau=6.141e-3)


def package_class_a(zp1: float) -> DevicePackage:
    "Class A device package, with transmission line 1 length ``zp1`` (mm)."
    return DevicePackage(stages=(TlineStage(zc=87.5, zp=zp1, **_CLASS_A),
                                 TlineStage(zc=92.5, zp=1.8, **_CLASS_A)),
                         Cp=CP)


def package_class_b(zp1: float) -> DevicePackage:
    "Class B device package, with transmission line 1 length ``zp1`` (mm)."
    return DevicePackage(stages=(TlineStage(zc=87.5, zp=zp1, **_CLASS_B),
                                 TlineStage(zc=95.0, zp=2.0, **_CLASS_B),
                                 TlineStage(zc=100., zp=1.3, **_CLASS_B),
                                 TlineStage(zc=78.0, zp=1.5, **_CLASS_B)),
                         Cp=CP)


# KR / C2C: transmission line 1 length (mm), per (class, test), as (Tx, Rx) (Tables 178-14, 176C-9).
ZP1_KR_C2C: dict[tuple[str, int], tuple[float, float]] = {
    ("A", 1): (33.0, 33.0),
    ("A", 2): (12.0, 12.0),
    ("B", 1): (45.0, 44.0),
    ("B", 2): (30.0, 29.0),
}


def kr_c2c_device(pkg_class: str, test: int, is_rx: bool) -> DeviceModel:
    "KR/C2C transmitter (``is_rx=False``) or receiver model, for package class ``pkg_class`` and Test 1 or 2."
    zp1 = ZP1_KR_C2C[(pkg_class, test)][1 if is_rx else 0]
    pkg = package_class_a(zp1) if pkg_class == "A" else package_class_b(zp1)
    return DeviceModel(termination=TERMINATION, package=pkg, Rd=R0)


# CR host classes (Table 179-23) and partial host channel transmission line (Table 179-22).
_CR_HOST_TLINE = dict(zc=92.5, gamma0=0.0, a1=5.95e-4, a2=2.6e-5, tau=5.79e-3)
CR_HOST_CLASSES: dict[str, tuple[str, float, float]] = {  # class: (package class, zp1 (mm), host PCB zp(h) (mm))
    "HL": ("A", 8.0, 22.0),
    "HN": ("B", 15.0, 83.0),
    "HH": ("B", 45.0, 74.0),
}
# Valid (transmitter host class, receiver host class) combinations, per cable assembly class (Table 179-21).
CR_VALID_COMBINATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "CA-A": (("HN", "HL"), ("HN", "HN"), ("HN", "HH"), ("HL", "HL"), ("HL", "HN"), ("HL", "HH"),
             ("HH", "HL"), ("HH", "HN")),
    "CA-B": (("HL", "HL"), ("HL", "HN"), ("HL", "HH"), ("HN", "HL"), ("HN", "HN"), ("HH", "HL")),
    "CA-C": (("HL", "HL"), ("HL", "HN"), ("HN", "HL")),
    "CA-D": (("HL", "HL"),),
}


def cr_device(host_class: str, include_host: bool = True) -> DeviceModel:
    """
    CR transmitter or receiver model for host class ``host_class`` (the models are symmetric; use ``s_rx_model()``
    for the receiver). With ``include_host=False``, the partial host channel is omitted (for channel data that
    already includes the host PCB).
    """
    pkg_class, zp1, zph = CR_HOST_CLASSES[host_class]
    pkg = package_class_a(zp1) if pkg_class == "A" else package_class_b(zp1)
    host = PartialHost(C0=0.0, tline=TlineStage(zp=zph, **_CR_HOST_TLINE), C1=0.0) if include_host else None
    return DeviceModel(termination=TERMINATION, package=pkg, host=host, Rd=R0)


# C2M host model (Table 176D-6): "class B"-like package with a specific transmission line 1 length, plus a partial
# host channel. The module C2M component interfaces at TP1d/TP4d, the device-to-package interface.
C2M_HOST_ZP1 = 45.0
_C2M_HOST_TLINE = dict(zc=92.5, gamma0=0.0, a1=5.95e-4, a2=2.6e-5, tau=5.79e-3)
C2M_HOST_PCB_ZP = 280.0


def c2m_host_device(include_host: bool = False) -> DeviceModel:
    """
    C2M host model. The partial host channel (280 mm of PCB) is omitted by default, because measured C2M channel
    data normally includes the host PCB already.
    """
    host = (PartialHost(C0=0.0, tline=TlineStage(zp=C2M_HOST_PCB_ZP, **_C2M_HOST_TLINE), C1=0.0)
            if include_host else None)
    return DeviceModel(termination=TERMINATION, package=package_class_b(C2M_HOST_ZP1), host=host, Rd=R0)


def c2m_module_device() -> DeviceModel:
    "C2M module end: device termination only (TP1d/TP4d are at the module's device-to-package interface)."
    return DeviceModel(termination=TERMINATION, package=None, host=None, Rd=R0)


# COM parameter values (Tables 178-15, 179-24, 176C-10, 176D-7), as far as `COMParams` can express them.
# Not modeled by PyChOpMarg (so COM here is approximate, e.g. for ranking channels and package cases):
# the Rx FFE's floating tap groups (Nwg, Nwf, Nmax), quantization noise (Nqb, Pqc), and MLSD (KR, CR).
def com_params(group: str, direction: str = "") -> COMParams:
    """
    COM parameters for channel group ``group`` ("KR", "CR", "C2C", or "C2M").

    For C2M, ``direction`` selects the transmitter transition time: "ingress" (module to host, Tr = 0.007 ns) or
    "egress" (host to module, Tr = 0.004 ns).
    """
    if group not in ("KR", "CR", "C2C", "C2M"):
        raise ValueError(f"Unknown channel group: {group!r}!")
    long_rx = group in ("KR", "CR")                     # dw = 6, Nfix = 15 (vs. 5 and 14 for C2C/C2M)
    dw, nfix = (6, 15) if long_rx else (5, 14)
    der0 = {"KR": 2e-4, "CR": 2e-4, "C2C": 0.67e-5, "C2M": 2e-5}[group]
    t_r = 0.007 if (group == "C2M" and direction == "ingress") else 0.004
    return replace(
        IEEE_8023dj,
        fb=FB, L=4, M=32, DER_0=der0, T_r=t_r, RLM=0.95, A_v=0.385, A_fe=0.385, A_ne=0.481, R_0=R0,
        A_DD=0.02, SNR_TX=33.5, eta_0=7.5e-9, sigma_Rj=0.01,
        f_z=FB / 2.5, f_p1=FB / 2.5, f_p2=FB, f_LF=FB / 80,          # f_z1, f_p1, f_p2, and f_z2 = f_p3
        g_DC=[-float(n) for n in range(21)], g_DC2=[-float(n) for n in range(7)],
        tx_taps_min=[0.0, 0.0, -0.34, -0.2, 0.0, 0.0],             # c(-3), c(-2), c(-1), c(1), c(2), c(3)
        tx_taps_max=[0.0, 0.14, 0.0, 0.0, 0.0, 0.0],
        tx_taps_step=[0.0, 0.02, 0.02, 0.02, 0.0, 0.0],
        c0_min=0.54, f_r=0.55,
        dfe_min=np.array([0.0]), dfe_max=np.array([0.85]),
        rx_taps_min=np.array([-0.7] * dw + [1.0] + [-0.7] * (nfix - dw - 1)),
        rx_taps_max=np.array([0.7] * dw + [1.0] + [0.7] * (nfix - dw - 1)),
        dw=dw)


# Package cases, per group, as selectable by name.
PKG_CASES: dict[str, tuple[str, ...]] = {
    "KR":  ("A1", "A2", "B1", "B2"),   # package class and Test number
    "C2C": ("A1", "A2", "B1", "B2"),
    "CR":  ("HL", "HN", "HH"),         # host class
    "C2M": ("host", "module"),         # which C2M component
}


def device_model(group: str, case: str, is_rx: bool, include_host: bool = True) -> DeviceModel:
    """
    Transmitter (``is_rx=False``) or receiver device model for channel group ``group`` and package case ``case``
    (one of ``PKG_CASES[group]``). ``include_host`` controls the partial host channel (CR, and C2M "host").
    """
    if case not in PKG_CASES.get(group, ()):
        raise ValueError(f"Package case {case!r} isn't valid for group {group!r}; choose from {PKG_CASES.get(group)}.")
    if group in ("KR", "C2C"):
        return kr_c2c_device(case[0], int(case[1]), is_rx)
    if group == "CR":
        return cr_device(case, include_host=include_host)
    return c2m_host_device(include_host=include_host) if case == "host" else c2m_module_device()
