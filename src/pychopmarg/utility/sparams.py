"""
S-parameter utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from pathlib    import Path
from typing     import Optional

import numpy as np  # type: ignore
import skrf  as rf
from numpy import array
from scipy.interpolate  import interp1d

from pychopmarg.common import Rvec, PI, TWOPI, COMNtwk


def sdd_21(ntwk: rf.Network, norm: float = 0.5, renumber: bool = False) -> rf.Network:
    """
    Given a 4-port single-ended network, return its differential throughput
    as a 2-port network.

    Args:
        ntwk: 4-port single ended network.

    Keyword Args:
        norm: Normalization factor. (Default = 0.5)
        renumber: Automatically detect and correct through path when True.
                  Default: False

    Returns:
        Sdd (2-port).

    Notes:
        1. A "1->2/3->4" port ordering convention is assumed when `renumber` is False.
        2. Automatic renumbering should not be used unless a solid d.c. thru path exists.
    """
    mm = se2mm(ntwk, norm=norm, renumber=renumber)
    return rf.Network(frequency=ntwk.f, s=mm.s[:, 0:2, 0:2], z0=mm.z0[:, 0:2])


def se2mm(ntwk: rf.Network, norm: float = 0.5, renumber: bool = False) -> rf.Network:
    """
    Given a 4-port single-ended network,
    return its mixed mode equivalent in the following format:

    .. math::
        \\begin{bmatrix}
            Sdd11 & Sdd12 & Sdc11 & Sdc12 \\\\
            Sdd21 & Sdd22 & Sdc21 & Sdc22 \\\\
            Scd11 & Scd12 & Scc11 & Scc12 \\\\
            Scd21 & Scd22 & Scc21 & Scc22
        \\end{bmatrix}

    Args:
        ntwk: 4-port single ended network.

    Keyword Args:
        norm: Normalization factor. (Default = 0.5)
        renumber: Automatically detect and correct through path when True.
                  Default: False

    Returns:
        Mixed mode equivalent network.

    Notes:
        1. A "1->2/3->4" port ordering convention is assumed when `renumber` is False.
        2. Automatic renumbering should not be used unless a solid d.c. thru path exists.
    """
    # Confirm correct network dimmensions.
    (_, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs == 4, "Touchstone file must have 4 ports!"

    # Detect/correct "1 => 3" port numbering if requested.
    if renumber:
        ix = 1
        if abs(ntwk.s21.s[ix, 0, 0]) < abs(ntwk.s31.s[ix, 0, 0]):  # 1 ==> 3 port numbering?
            ntwk.renumber((1, 2), (2, 1))

    # Convert S-parameter data.
    s = np.zeros(ntwk.s.shape, dtype=complex)
    s[:, 0, 0] = norm * (ntwk.s11 - ntwk.s13 - ntwk.s31 + ntwk.s33).s.flatten()
    s[:, 0, 1] = norm * (ntwk.s12 - ntwk.s14 - ntwk.s32 + ntwk.s34).s.flatten()
    s[:, 0, 2] = norm * (ntwk.s11 + ntwk.s13 - ntwk.s31 - ntwk.s33).s.flatten()
    s[:, 0, 3] = norm * (ntwk.s12 + ntwk.s14 - ntwk.s32 - ntwk.s34).s.flatten()
    s[:, 1, 0] = norm * (ntwk.s21 - ntwk.s23 - ntwk.s41 + ntwk.s43).s.flatten()
    s[:, 1, 1] = norm * (ntwk.s22 - ntwk.s24 - ntwk.s42 + ntwk.s44).s.flatten()
    s[:, 1, 2] = norm * (ntwk.s21 + ntwk.s23 - ntwk.s41 - ntwk.s43).s.flatten()
    s[:, 1, 3] = norm * (ntwk.s22 + ntwk.s24 - ntwk.s42 - ntwk.s44).s.flatten()
    s[:, 2, 0] = norm * (ntwk.s11 - ntwk.s13 + ntwk.s31 - ntwk.s33).s.flatten()
    s[:, 2, 1] = norm * (ntwk.s12 - ntwk.s14 + ntwk.s32 - ntwk.s34).s.flatten()
    s[:, 2, 2] = norm * (ntwk.s11 + ntwk.s13 + ntwk.s31 + ntwk.s33).s.flatten()
    s[:, 2, 3] = norm * (ntwk.s12 + ntwk.s14 + ntwk.s32 + ntwk.s34).s.flatten()
    s[:, 3, 0] = norm * (ntwk.s21 - ntwk.s23 + ntwk.s41 - ntwk.s43).s.flatten()
    s[:, 3, 1] = norm * (ntwk.s22 - ntwk.s24 + ntwk.s42 - ntwk.s44).s.flatten()
    s[:, 3, 2] = norm * (ntwk.s21 + ntwk.s23 + ntwk.s41 + ntwk.s43).s.flatten()
    s[:, 3, 3] = norm * (ntwk.s22 + ntwk.s24 + ntwk.s42 + ntwk.s44).s.flatten()

    # Convert port impedances.
    f = ntwk.f
    z = np.zeros((len(f), 4), dtype=complex)
    z[:, 0] = ntwk.z0[:, 0] + ntwk.z0[:, 2]
    z[:, 1] = ntwk.z0[:, 1] + ntwk.z0[:, 3]
    z[:, 2] = (ntwk.z0[:, 0] * ntwk.z0[:, 2]) / (ntwk.z0[:, 0] + ntwk.z0[:, 2])
    z[:, 3] = (ntwk.z0[:, 1] * ntwk.z0[:, 3]) / (ntwk.z0[:, 1] + ntwk.z0[:, 3])

    return rf.Network(frequency=f, s=s, z0=z)


def import_s32p(  # pylint: disable=too-many-locals
    filename: Path, vic_chnl: int = 1
) -> list[COMNtwk]:
    """Read in a 32-port Touchstone file, and return an equivalent list
    of 8 2-port differential networks: a single victim through channel and
    7 crosstalk aggressors, according to the VITA 68.2 convention.

    Args:
        filename: Name of Touchstone file to read in.

    Keyword Args:
        vic_chnl: Victim channel number (from 1).
            Default = 1

    Returns:
        List of 8 pairs, each consisting of:

            - a 2-port network representing a *differential* channel, and
            - the type of that channel, one of: 'THRU', 'NEXT', or 'FEXT.
              (First element is the victim and the only one of type 'THRU'.)

    Raises:
        ValueError: If Touchstone file is not 32-port.

    Notes:
        1. Input Touchstone file is assumed single-ended.
        2. The differential through and xtalk channels are returned.
        3. Port 2 of all returned channels correspond to the same physical circuit node,
           typically, the Rx input node.
    """

    # Import and sanity check the Touchstone file.
    ntwk = rf.Network(filename)
    (_, rs, cs) = ntwk.s.shape
    assert rs == cs, "Non-square Touchstone file S-matrix!"
    assert rs == 32, f"Touchstone file must have 32 ports!\n\t{ntwk}"

    # Extract the victim and aggressors.
    def ports_from_chnls(left, right):
        """
        Return list of 4 ports (from 0) corresponding to a particular
        left and right channel ID (from 1), assuming "1=>2/3=>4" convention.

        Args:
            left(int): Left side channel number (from 1).
            right(int): Right side channel number (from 1).

        Returns:
            List of ports (from 0) for desired channel.
        """
        left0 = left - 1     # 0-based
        right0 = right - 1
        return [left0 * 4, right0 * 4 + 1, left0 * 4 + 2, right0 * 4 + 3]

    vic_ports = ports_from_chnls(vic_chnl, vic_chnl)
    vic = sdd_21(rf.subnetwork(ntwk, vic_ports))
    vic = (vic, 'THRU')
    if vic_chnl % 2:  # odd?
        vic_rx_ports = [vic_ports[n] for n in [0, 2]]
    else:
        vic_rx_ports = [vic_ports[n] for n in [1, 3]]
    agg_chnls = list(np.arange(8) + 1)
    agg_chnls.remove(vic_chnl)  # type: ignore
    aggs = []
    for agg_chnl in agg_chnls:
        agg_ports = ports_from_chnls(agg_chnl, agg_chnl)
        if agg_chnl % 2:  # odd?
            agg_tx_ports = [agg_ports[n] for n in [1, 3]]
        else:
            agg_tx_ports = [agg_ports[n] for n in [0, 2]]
        sub_ports = np.concatenate(list(zip(agg_tx_ports, vic_rx_ports)))
        subntwk = sdd_21(ntwk.subnetwork(sub_ports))
        if (vic_chnl + agg_chnl) % 2:
            subntwk = (subntwk, 'NEXT')
        else:
            subntwk = (subntwk, 'FEXT')
        aggs.append(subntwk)
    return [vic] + aggs


def sCshunt(freqs: Rvec, c: float, r0: float = 50.0) -> rf.Network:
    """
    Calculate the 2-port network for a shunt capacitance.

    Args:
        freqs: The frequencies at which to calculate network data (Hz).
        c: The capacitance (F).

    Keyword Args:
        r0: The reference impedance for the network (Ohms).
            Default: 50 Ohms.

    Returns:
        The network corresponding to a shunt capacitance, ``c``,
        calculated at the given frequencies, ``freqs``.
    """
    w = TWOPI * freqs
    s = 1j * w
    jwRC = s * r0 * c
    s11 = -jwRC / (2 + jwRC)
    s21 =     2 / (2 + jwRC)
    return rf.Network(s=np.array(list(zip(zip(s11, s21), zip(s21, s11)))), f=freqs, z0=r0)


def sLseries(freqs: Rvec, inductance: float, r0: float = 50.0) -> rf.Network:
    """
    Calculate the 2-port network for a series inductance.

    Args:
        freqs: The frequencies at which to calculate network data (Hz).
        inductance: The inductance (H).

    Keyword Args:
        r0: The reference impedance for the network (Ohms).
            Default: 50 Ohms.

    Returns:
        The network corresponding to a series inductance, ``inductance``,
        calculated at the given frequencies, ``freqs``.
    """
    w = TWOPI * np.array(freqs)
    s = 1j * w
    w2L2 = w**2 * inductance**2
    jwRL = s * r0 * inductance
    R2x2 = 2 * r0**2
    den = 2 * R2x2 + w2L2
    s11 = (w2L2 + 2 * jwRL) / den
    s21 = 2 * (R2x2 - jwRL) / den
    return rf.Network(s=np.array(list(zip(zip(s11, s21), zip(s21, s11)))), f=freqs, z0=r0)


def sDieLadderSegment(freqs: Rvec, trip: tuple[float, float, float]) -> rf.Network:
    """
    Calculate one segment of the on-die parasitic ladder network.

    Args:
        f: List of frequencies to use for network creation (Hz).
        trip: Triple containing:

            - R0: Reference impedance for network (Ohms).
            - Cd: Shunt capacitance (F).
            - Ls: Series inductance (H).

    Returns:
        Two port network for segment.
    """
    R0, Cd, Ls = trip
    return sCshunt(freqs, Cd, r0=R0) ** sLseries(freqs, Ls, r0=R0)


def sPkgTline(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    f: Rvec, r0: float, a1: float, a2: float, tau: float,
    gamma0: float, z_pairs: list[tuple[float, float]]
) -> rf.Network:
    """
    Return the 2-port network corresponding to a package transmission line,
    according to (93A-9:14).

    Args:
        f: Frequencies at which to calculate S-parameters (Hz).
        r0: System reference impedance (Ohms).
        a1: First polynomial coefficient (sqrt_ns/mm).
        a2: Second polynomial coefficient (ns/mm).
        tau: Propagation delay (ns/mm).
        gamma0: Propagation loss constant (1/mm).
        z_pairs: List of pairs defining the T-line segments, each containing:

            - zc: Characteristic impedance of segment (Ohms).
            - zp: Length of segment (mm).

    Returns:
        2-port network equivalent to package transmission line.
    """

    f_GHz  = f / 1e9  # noqa E221
    gamma1 = a1 * (1 + 1j)

    def gamma2(f: float) -> complex:
        "f in GHz!"
        return a2 * (1 - 1j * (2 / PI) * np.log(f)) + 1j * TWOPI * tau

    def gamma(f: float) -> complex:
        "Return complex propagation coefficient at frequency f (GHz)."
        if f == 0:
            return gamma0
        return gamma0 + gamma1 * np.sqrt(f) + gamma2(f) * f

    g = array(list(map(gamma, f_GHz)))  # type: ignore

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
        s11 = rho * (1 - np.exp(-g * 2 * zp)) / (1 - rho**2 * np.exp(-g * 2 * zp))
        s21 = (1 - rho**2) * np.exp(-g * zp)  / (1 - rho**2 * np.exp(-g * 2 * zp))
        return rf.Network(s=array(list(zip(zip(s11, s21), zip(s21, s11)))), f=f, z0=r0)

    return rf.network.cascade_list(list(map(mk_s2p, z_pairs)))


def s2p_pulse_response(s2p: rf.Network, ui: float, t: Optional[Rvec] = None) -> tuple[Rvec, Rvec]:
    """
    Calculate the __pulse__ response of a 2-port network, using the SciKit-RF provided __step__ response.

    Args:
        s2p: The 2-port network to use.
        ui: The unit interval (s).

    Keyword Args:
        t: Optional time vector for use in interpolating the resultant pulse response (s).
            Default: None (Use time vector returned by __SciKit-RF__ step response function.)

    Returns:
        t, p: A pair consisting of:

            - The time values at which the pulse response has been sampled, and
            - The real-valued pulse response samples of the given 2-port network.

    Raises:
        ValueError: If given network is not 2-port.
    """

    # Confirm correct network dimmensions.
    (_, rs, cs) = s2p.s.shape
    assert rs == cs, ValueError("Non-square Touchstone file S-matrix!")
    assert rs == 2, ValueError(f"Touchstone file must have 2 ports, not {rs}!")

    _t, _s = s2p.s21.extrapolate_to_dc().step_response()
    if t is not None:
        krnl = interp1d(_t, _s, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
        s = krnl(t)
    else:
        t = _t
        s = _s
    delta_t = t[1] - t[0]
    nspui = int(np.round(ui / delta_t))
    p = s - np.pad(s, (nspui, 0))[:len(s)]

    return t, p
