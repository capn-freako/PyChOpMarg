"""
General purpose utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 3, 2024 (Copied from `pybert.utility`.)

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

import numpy as np  # type: ignore
import skrf as rf

from typing import Any, Dict, Optional, TypeVar

from scipy.interpolate import interp1d

from pychopmarg.common import Rvec, Cvec, COMParams, PI, TWOPI

T = TypeVar('T', Any, Any)


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


def sCshunt(freqs: list[float], c: float, r0: float = 50.0) -> rf.Network:
    """
    Calculate the 2-port network for a shunt capacitance.

    Args:
        freqs: The frequencies at which to calculate network data (Hz).
        c: The capacitance (F).

    Keyword Args:
        r0: The reference impedance for the network (Ohms).
            Default: 50 Ohms.

    Returns:
        s2p: The network corresponding to a shunt capacitance, `c`,
            calculated at the given frequencies, `freqs`.
    """
    w = TWOPI * np.array(freqs)
    s = 1j * w
    jwRC = s * r0 * c
    s11 = -jwRC / (2 + jwRC)
    s21 =     2 / (2 + jwRC)
    return rf.Network(s=np.array(list(zip(zip(s11, s21), zip(s21, s11)))), f=freqs, z0=r0)


def sLseries(freqs: list[float], l: float, r0: float = 50.0) -> rf.Network:
    """
    Calculate the 2-port network for a series inductance.

    Args:
        freqs: The frequencies at which to calculate network data (Hz).
        l: The inductance (H).

    Keyword Args:
        r0: The reference impedance for the network (Ohms).
            Default: 50 Ohms.

    Returns:
        s2p: The network corresponding to a series inductance, `l`,
            calculated at the given frequencies, `freqs`.
    """
    w = TWOPI * np.array(freqs)
    s = 1j * w
    w2L2 = w**2 * l**2
    jwRL = s * r0 * l
    R2x2 = 2 * r0**2
    den = 2 * R2x2 + w2L2
    s11 = (w2L2 + 2 * jwRL) / den
    s21 = 2 * (R2x2 - jwRL) / den
    return rf.Network(s=np.array(list(zip(zip(s11, s21), zip(s21, s11)))), f=freqs, z0=r0)


def sDieLadderSegment(freqs: list[float], trip: tuple[float, float, float]) -> rf.Network:
    """
    Calculate one segment of the on-die parasitic ladder network.

    Args:
        f: List of frequencies to use for network creation (Hz).
        trip: Triple containing:
            - R0: Reference impedance for network (Ohms).
            - Cd: Shunt capacitance (F).
            - Ls: Series inductance (H).

    Returns:
        s2p: Two port network for segment.
    """
    R0, Cd, Ls = trip
    return sCshunt(freqs, Cd, r0=R0) ** sLseries(freqs, Ls, r0=R0)


def filt_pr_samps(pr_samps: Rvec, As: float, rel_thresh: float = 0.001) -> Rvec:
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


def delta_pmf(
    h_samps: Rvec, L: int = 4, RLM: float = 1.0,
    curs_ix: Optional[int] = None, y: Optional[Rvec] = None,
    dbg_dict: Dict[str, Any] = None
) -> Rvec:
    """
    Calculate the "delta-pmf" for a set of pulse response samples,
    as per (93A-40).

    Args:
        h_samps: Vector of pulse response samples.

    Keyword Args:
        L: Number of modulation levels.
            Default: 4
        RLM: Relative level mismatch.
            Default: 1.0
        curs_ix: Cursor index override.
            Default: None (Means use `argmax()` to find cursor.)
        y: y-values override vector.
            Default: None (Means calculate appropriate y-value vector here.)
        dbg_dict: Optional dictionary into which debugging values may be stashed,
            for later analysis.
            Default: None

    Returns:
        A pair consisting of:
        - the voltages corresponding to the bins, and
        - their probabilities.

    Raises:
        `ValueError` if the given pulse response contains any NaNs.
        `ValueError` if a needed shift exceeds half the result vector length.

    Notes:
        1. The input set of pulse response samples is filtered,
            as per Note 2 of 93A.1.7.1, unless a y-values override
            vector is provided, in which case it is assumed that
            the caller has already done the filtering.
    """

    assert not any(np.isnan(h_samps)), ValueError(
        f"Input contains NaNs at: {np.where(np.isnan(h_samps))[0]}")
    
    if y is None:
        curs_ix = curs_ix or np.argmax(h_samps)
        curs_val = h_samps[curs_ix]
        max_y = 1.1 * curs_val
        npts = 2 * min(int(max_y / 0.00001), 10_000) + 1  # Note 1 of 93A.1.7.1; MUST BE ODD!
        y = np.linspace(-max_y, max_y, npts)
        ystep = 2 * max_y / (npts - 1)
        h_samps_filt = filt_pr_samps(h_samps, max_y)
    else:
        npts = len(y)
        ystep = y[1] - y[0]
        h_samps_filt = h_samps

    delta = np.zeros(npts)
    delta[npts // 2] = 1

    if dbg_dict is not None:
        dbg_dict.update({"h_samps":      h_samps})
        dbg_dict.update({"h_samps_filt": h_samps_filt})
        dbg_dict.update({"ystep":        ystep})

    def pn(hn: float) -> Rvec:
        """
        (93A-39)
        """
        if dbg_dict:
            dbg_dict.update({"hn":     hn})
            dbg_dict.update({"shifts": []})
        _rslt = np.zeros(npts)
        for el in range(L):
            _shift = int((2 * el / (L - 1) - 1) * hn / ystep)
            if dbg_dict:
                dbg_dict["shifts"].append(_shift)
            assert abs(_shift) < npts // 2, ValueError(
                f"Wrap around: _shift: {_shift}, npts: {npts}.")
            _rslt += np.roll(delta, _shift)
        return 1 / L * _rslt

    rslt = delta
    for hn in h_samps_filt:
        _pn = pn(hn)
        rslt = np.convolve(rslt, _pn, mode='same')
    rslt /= sum(rslt)  # Enforce a PMF.

    return y, rslt


def from_dB(x: float) -> float:
    """Convert from (dB) to real, assuming square law applies."""
    return pow(10, x / 20)


def all_combs(xss: list[list[T]]) -> list[list[T]]:
    """
    Generate all combinations of input.

    Args:
        xss: The lists of candidates for each position in the final output.

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


def calc_Hffe(
    freqs: Rvec, td: float,
    tap_weights: Rvec, n_post: int,
    hasCurs: bool = False
) -> Cvec:
    """
    Calculate the voltage transfer function, H(f), for a digital FFE, according to (93A-21).

    Args:
        freqs: Frequencies at which to calculate `Hffe` (Hz).
        td: Tap delay time (s).
        tap_weights: The filter tap weights.
        n_post: The number of post-cursor taps.

    Keyword Args:
        hasCurs: `tap_weights` includes the cursor tap weight when True.
            Default: False (Cursor tap weight will be calculated.)

    Returns:
        The complex voltage transfer function, H(f), for the FFE.

    Raises:
        None
    """

    bs = list(np.array(tap_weights).flatten())
    if not hasCurs:
        b0 = 1 - sum(list(map(abs, tap_weights)))
        bs.insert(-n_post, b0)
    return sum(list(map(lambda n_b: n_b[1] * np.exp(-1j * TWOPI * n_b[0] * td * freqs),
                        enumerate(bs))))


def null_filter(nTaps: int, nPreTaps: int = 0) -> Rvec:
    """
    Construct a null filter w/ `nTaps` taps and (optionally) `nPreTaps` pre-cursor taps.

    Args:
        nTaps: Total number of taps, including the cursor tap.

    Keyword Args:
        nPreTaps: Number of pre-cursor taps.
            Default: 0

    Returns:
        taps: The filter tap weight vector, including the cursor tap weight.
    """

    assert nTaps > 0, ValueError(
        f"`nTaps` ({nTaps}) must be greater than zero!")
    assert nPreTaps < nTaps, ValueError(
        f"`nPreTaps` ({nPreTaps}) must be less than `nTaps` ({nTaps})!")

    taps = zeros(nTaps)
    taps[nPreTaps] = 1.0

    return taps


def from_irfft(x: Rvec, t_irfft: Rvec, t: Rvec, nspui: int) -> Rvec:
    """
    Interpolate `irfft()` output to `t` and subsample at fBaud.

    Args:
        x: `irfft()` results to be interpolated and subsampled.
        t_irfft: Time index vector for `x`.
        t: Desired new time index vector (same units as `t_irfft`).
        nspui: Number of samples per unit interval.

    Returns:
        y: interpolated and subsampled vector.

    Raises:
        IndexError: If length of input doesn't match length of `t_irfft` vector.

    Notes:
        1. Input vector is shifted, such that its peak occurs at `0.1 * max(t)`, before interpolating.
            This is done to:
                - ensure that we don't omit any non-causal behavior,
                which ends up at the end of an IFFT output vector
                when the peak is very near the beginning, and
                - to ensure that the majority of our available time span
                is available for capturing reflections.
        2. The sub-sampling phase is adjusted, so as to ensure that we catch the peak.
    """

    assert len(x) == len(t_irfft), IndexError(
        f"Length of input ({len(x)}) must match length of `t_irfft` vector ({len(t_irfft)})!")

    t_pk = 0.1 * t[-1]                      # target peak location time
    targ_ix = np.where(t_irfft >= t_pk)[0][0]  # target peak vector index, in `x`
    curr_ix = np.argmax(x)                     # current peak vector index, in `x`
    _x = np.roll(x, targ_ix - curr_ix)         # `x` with peak repositioned

    krnl = interp1d(t_irfft, _x, bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    y = krnl(t)
    curs_uis, curs_ofst = divmod(np.argmax(y), nspui)  # Ensure that we capture the peak in the next step.
    return y[curs_ofst::nspui]                         # Sampled at fBaud, w/ peak captured.
