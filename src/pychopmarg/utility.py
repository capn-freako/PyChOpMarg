"""
General purpose utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024 (Copied from `pybert.utility`.)

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

import numpy as np  # type: ignore
import skrf as rf


def sdd_21(ntwk: rf.Network, norm: float = 0.5, renumber: bool = False) -> rf.Network:
    """
    Given a 4-port single-ended network, return its differential throughput
    as a 2-port network.

    Args:
        ntwk: 4-port single ended network.

    Keyword Args:
        norm: Normalization factor. (Default = 0.5)
        renumber: Automatically detect correct through path when True.
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
    Given a 4-port single-ended network, return its mixed mode equivalent.

    Args:
        ntwk: 4-port single ended network.

    Keyword Args:
        norm: Normalization factor. (Default = 0.5)
        renumber: Automatically detect correct through path when True.
                  Default: False

    Returns:
        Mixed mode equivalent network, in the following format:
            Sdd11  Sdd12  Sdc11  Sdc12
            Sdd21  Sdd22  Sdc21  Sdc22
            Scd11  Scd12  Scc11  Scc12
            Scd21  Scd22  Scc21  Scc22

    Notes:
        1. A "1->2/3->4" port ordering convention is assumed when `renumber` is False.
        2. Automatic renumbering should not be used unless a solid d.c. thru path exists.
    """
    # Confirm correct network dimmensions.
    (fs, rs, cs) = ntwk.s.shape
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
    z[:, 2] = (ntwk.z0[:, 0] + ntwk.z0[:, 2]) / 2
    z[:, 3] = (ntwk.z0[:, 1] + ntwk.z0[:, 3]) / 2

    return rf.Network(frequency=f, s=s, z0=z)


def import_s32p(filename: str, vic_chnl: int = 1) -> list[tuple[rf.Network, str]]:
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
    (fs, rs, cs) = ntwk.s.shape
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
    agg_chnls = list(np.array(range(8)) + 1)
    agg_chnls.remove(vic_chnl)
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
