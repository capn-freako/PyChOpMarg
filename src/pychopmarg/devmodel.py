"""
Transmitter and receiver S-parameter models of IEEE P802.3dj Annex 178A (178A.1.4 and 178A.1.5).

Original author: David Banas <capn.freako@gmail.com>

Original date:   September 29, 2026

Copyright (c) 2026 David Banas; all rights reserved World wide.

The transmitter model is the cascade of a device termination (a ladder of shunt-C / series-L stages ending in the
bump capacitance), a device package (a cascade of transmission line stages ending in the package-to-board
capacitance), and an optional partial host channel (shunt C, transmission line, shunt C). The receiver model is the
same cascade, built with the receiver's parameter values, with its port order reversed.

All networks returned here are differential-mode 2-ports, referenced to ``2 * R0``. Capacitances and inductances
are single-ended values (nF, nH) and transmission line characteristic impedances are differential (Ohms), exactly
as tabulated in P802.3dj (e.g. Table 178-14); the underlying single-ended formulae of 93A.1.2.2 -- 93A.1.2.4
(which use ``R0`` and ``2 * R0``) yield the differential-mode S-parameters directly.
"""

from dataclasses import dataclass
from typing      import Optional

import numpy as np
import skrf  as rf

from pychopmarg.common          import Rvec
from pychopmarg.utility.sparams import sCshunt, sLseries, sPkgTline


@dataclass(frozen=True)
class TlineStage:
    """One transmission line stage (93A.1.2.4), with its own loss/delay parameters (P802.3dj Table 178A-4)."""
    zc: float       # Differential characteristic impedance (Ohms).
    zp: float       # Length (mm).
    gamma0: float   # (1/mm)
    a1: float       # (ns^1/2 / mm)
    a2: float       # (ns/mm)
    tau: float      # (ns/mm)


@dataclass(frozen=True)
class DeviceTermination:
    """Device termination model parameters (P802.3dj 178A.1.4.1, Table 178A-3)."""
    Cd: tuple[float, ...]   # Single-ended shunt capacitance of each stage, 1..N (nF).
    Ls: tuple[float, ...]   # Single-ended series inductance of each stage, 1..N (nH).
    Cb: float               # Single-ended bump capacitance (nF).

    def __post_init__(self):
        if len(self.Cd) != len(self.Ls):
            raise ValueError(f"`Cd` ({len(self.Cd)}) and `Ls` ({len(self.Ls)}) must have one value per stage!")


@dataclass(frozen=True)
class DevicePackage:
    """Device package model parameters (P802.3dj 178A.1.4.2, Table 178A-4)."""
    stages: tuple[TlineStage, ...]  # Transmission line stages, 1..N, in order from the device termination.
    Cp: float                       # Single-ended package-to-board capacitance (nF).


@dataclass(frozen=True)
class PartialHost:
    """Partial host channel model parameters (P802.3dj 178A.1.4.3, Table 178A-5)."""
    C0: float           # Single-ended board capacitance at the package-to-board interface (nF).
    tline: TlineStage   # Host transmission line.
    C1: float           # Single-ended capacitance at the model-to-measurement interface (nF).


@dataclass(frozen=True)
class DeviceModel:
    """Everything needed to model one end (transmitter or receiver) of a link, per P802.3dj 178A.1.4/178A.1.5."""
    termination: DeviceTermination
    package: Optional[DevicePackage] = None
    host: Optional[PartialHost] = None
    Rd: float = 46.25   # Single-ended device termination resistance (Ohms).


def _diff(ntwk: rf.Network, R0: float) -> rf.Network:
    """Relabel a network built from the single-ended formulae (at ``R0``) as differential, referenced to ``2 * R0``."""
    return rf.Network(s=ntwk.s, f=ntwk.f, z0=2 * R0)


def s_shunt_c(f: Rvec, c: float, R0: float) -> rf.Network:
    """Shunt capacitance ``c`` (single-ended, nF) (93A.1.2.2)."""
    return _diff(sCshunt(f, c * 1e-9, r0=R0), R0)


def s_series_l(f: Rvec, ls: float, R0: float) -> rf.Network:
    """Series inductance ``ls`` (single-ended, nH) (93A.1.2.2a)."""
    return _diff(sLseries(f, ls * 1e-9, r0=R0), R0)


def s_tline(f: Rvec, stage: TlineStage, R0: float) -> rf.Network:
    """Transmission line stage (93A.1.2.4)."""
    return _diff(sPkgTline(f, R0, stage.a1, stage.a2, stage.tau, stage.gamma0, [(stage.zc, stage.zp)]), R0)


def s_device_termination(f: Rvec, term: DeviceTermination, R0: float) -> rf.Network:
    """
    Device termination model S(d) (178A-9, 178A-10): initialized to S(Cb), then each stage
    ``S(n) = cascade(S(Cd(n)), S(Ls(n)))`` is prepended, from n = N down to n = 1.
    """
    s_d = s_shunt_c(f, term.Cb, R0)
    for cd, ls in reversed(list(zip(term.Cd, term.Ls))):
        s_d = s_shunt_c(f, cd, R0) ** s_series_l(f, ls, R0) ** s_d
    return s_d


def s_device_package(f: Rvec, pkg: DevicePackage, R0: float) -> rf.Network:
    """Device package model S(p) (178A-11): initialized to S(Cp), then stages prepended from n = N down to n = 1."""
    s_p = s_shunt_c(f, pkg.Cp, R0)
    for stage in reversed(pkg.stages):
        s_p = s_tline(f, stage, R0) ** s_p
    return s_p


def s_partial_host(f: Rvec, host: PartialHost, R0: float) -> rf.Network:
    """Partial host channel model S(h) (178A-12): ``cascade(S(C0), S(tline), S(C1))``."""
    return s_shunt_c(f, host.C0, R0) ** s_tline(f, host.tline, R0) ** s_shunt_c(f, host.C1, R0)


def s_tx_model(f: Rvec, dev: DeviceModel, R0: float) -> rf.Network:
    """
    Transmitter S-parameter model S(t) (178A-7, 178A-8): device termination, then device package,
    then (optionally) partial host channel. Port 1 faces the device; port 2 faces the channel.
    """
    s_t = s_device_termination(f, dev.termination, R0)
    if dev.package is not None:
        s_t = s_t ** s_device_package(f, dev.package, R0)
    if dev.host is not None:
        s_t = s_t ** s_partial_host(f, dev.host, R0)
    return s_t


def s_rx_model(f: Rvec, dev: DeviceModel, R0: float) -> rf.Network:
    """
    Receiver S-parameter model S(r) (178A.1.5): built as for the transmitter, with the receiver's parameter
    values, then port-reversed, so that port 1 faces the channel and port 2 faces the device.
    """
    s_r = s_tx_model(f, dev, R0)
    s_r.flip()
    return s_r


def gamma_term(Rd: float, R0: float) -> float:
    """Termination reflection coefficient, ``(Rd - R0) / (Rd + R0)`` (178A-3)."""
    return (Rd - R0) / (Rd + R0)


def renormalize_to(ntwk: rf.Network, z0: float) -> rf.Network:
    """
    Copy of 2-port ``ntwk`` renormalized to reference impedance ``z0`` (the differential-mode equivalent of 178A-4,
    for ``z0 = 2 * R0``).
    """
    rslt = ntwk.copy()
    if not np.allclose(rslt.z0, z0):
        rslt.renormalize(z0)  # type: ignore
    return rslt


def terminated_h21(s2p: rf.Network, g1: float, g2: float) -> np.ndarray:
    """
    Voltage transfer function of 2-port ``s2p`` between a source and load with reflection coefficients ``g1``
    and ``g2``, per (178A-3). (It is the ratio of the load voltage to the wave launched into a matched line, i.e.
    twice the ratio to the source's open-circuit voltage.)
    """
    s11, s12, s21, s22 = s2p.s[:, 0, 0], s2p.s[:, 0, 1], s2p.s[:, 1, 0], s2p.s[:, 1, 1]
    return s21 * (1 - g1) * (1 + g2) / (1 - s11 * g1 - s22 * g2 + g1 * g2 * (s11 * s22 - s12 * s21))
