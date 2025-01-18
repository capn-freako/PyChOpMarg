"""
General plotting utilities for PyChOpMarg.

Original author: David Banas <capn.freako@gmail.com>

Original date:   January 17, 2025

Copyright (c) 2025 David Banas; all rights reserved World wide.
"""

from enum   import Enum
from random import choice, sample
from typing import Any, Callable, Optional

from matplotlib         import pyplot as plt  # type: ignore
from matplotlib.axes    import Axes           # type: ignore
from scipy.interpolate  import interp1d

from pychopmarg.com     import COM
from pychopmarg.common  import *
from pychopmarg.utility import s2p_pulse_response


class ZoomMode(Enum):
    "Plot zoom extent."
    FULL     = 1    # Use all available data.
    ISI      = 2    # Show the full ISI sampling used in Rx FFE tap weight optimization.
    PULSE    = 3    # Zoom in on the pulse, to inspect its shape in detail.
    MANUAL   = 4    # x-axis min. & max. specified by caller.
    RELATIVE = 5    # x-axis min. & max. specified by caller, relative to pulse peak location.


def plot_group_samps(
    plot_func: Callable[[COM, str, str, str, dict[str, str], Any, Any], None],
    x_lbl: str, y_lbls: tuple[str, str],
    coms: list[tuple[str, dict[str, str], dict[str, dict[str, COM]]]],
    maxRows: int = 3,
    dx=4, dy=3,
    chnls_by_grp: Optional[dict[str, list[str]]] = None,
    auto_yscale: bool = False
) -> dict[str, list[str]]:
    """
    Call the given plotting function for several randomly chosen channel sets from each available group.

    Args:
        plot_func: The plotting function to use. Should take the following arguments:
        
            - com: The COM object to use for plotting.
            - grp: Channel set group name
            - lbl: Channel set name
            - name: COM set name
            - opts: Plotting options to use
            - ax1: First y-axis
            - ax2: Second y-axis
            
        x_lbl: Label for x-axis
        y_lbls: Pair of labels, one for each y-axis
        coms: List of tuples, each containing:
        
            - name: Identifying name,
            - opts: Plotting options to use,
            - coms: dictionary of COM objects to select from.
                Should be indexed first by group name then by channel set name.

    Keyword Args:
        maxRows: Maximum number of rows desired in resultant plot matrix.
            (Number of columns is equal to number of groups.)
            Default: 3
        dx: Width of individual plots (in.)
            Default: 4
        dy: Height of individual plots (in.)
            Default: 3
        chnls_by_grp: Dictionary of key/value pairs of the form: <group name>: [<channel set name].
            (Used to enforce identical channel set choices across multiple calls.)
            Default: None
        auto_yscale: When ``True``, scale the y-axis to just accommodate the visible portion of the plotted waveforms.
            Default: False
            
    Returns:
        Dictionary containing lists of channel sets used by group name (for subsequent calls).

    Raises:
        KeyError: If there are any inconsistencies in dictionary key naming, either within the list of COMs given or between those COMs and the ``chnls_by_grp`` keyword argument, if provided.
    """
    group_names = list(coms[0][2].keys())
    nCols = len(group_names)
    nRows = min(maxRows, min(list(map(lambda grp: len(list(coms[0][2][grp].keys())),
                                      group_names))))
    fig, axs = plt.subplots(nRows, nCols, figsize=(dx * nCols, dy * nRows))
    try:  # Handle singleton case gracefully.
        assert isinstance(axs[0][0], Axes)
    except:
        axs = [[axs,],]
    n = 0
    print("     ", end="")
    chnls_used: dict[str, list[str]] = {}
    for grp in group_names:
        print(f"{grp : ^45s}", end="")
        chnls_used.update({grp: []})
        if chnls_by_grp:
            chnls = chnls_by_grp[grp]
        else:
            chnls = sample(sorted(coms[0][2][grp].keys()), nRows)  # `sorted` is necessary.
        chnls = sorted(chnls)
        for lbl in chnls:
            chnls_used[grp].append(lbl)
            col, row = divmod(n, nRows)
            ax1 = axs[row][col]
            try:
                ax2 = ax1.twinx()
            except:
                print(f"ax1: {ax1}")
                print(f"axs: {axs}")
                print(f"type(ax1): {type(ax1)}")
                print(f"type(axs): {type(axs)}")
                raise
            plt.tight_layout()
            for nm, opts, d in coms:
                com  = d[grp][lbl]
                plot_func(com, grp, lbl, nm, opts, ax1, ax2)
            if auto_yscale:  # Set y-limits automatically.
                for ax in [ax1, ax2]:
                    xmin, xmax = ax.get_xlim()
                    ymin =  1e6
                    ymax = -1e6
                    for line in ax.lines:
                        xdata, ydata = line.get_data()
                        xmin_ixs = np.where(xdata >= max(xdata[0],  xmin))[0]
                        if len(xmin_ixs):
                            xmin_ix = xmin_ixs[0]
                        else:
                            continue
                        xmax_ix = np.where(xdata >= min(xdata[-1], xmax))[0][0]
                        y_values = ydata[xmin_ix: xmax_ix]
                        if len(y_values):
                            _min_y = min(y_values)
                            _max_y = max(y_values)
                            if _min_y < ymin:
                                ymin = _min_y
                            if _max_y > ymax:
                                ymax = _max_y
                    delta_y = ymax - ymin
                    if delta_y > 0:
                        ymin -= 0.1 * delta_y
                        ymax += 0.1 * delta_y
                    else:
                        ymin = ymax = 0
                    ax.axis(ymin=ymin, ymax=ymax)
            if row == nRows - 1:
                ax1.set_xlabel(x_lbl)
            if col == 0:
                ax1.set_ylabel(y_lbls[0])
            if col == nCols - 1:
                ax2.set_ylabel(y_lbls[1])
            n += 1
    plt.show()
    return chnls_used


def plot_pulse_resps_gen(
    zoom: ZoomMode,
    noeq: bool = False,
    nopkg: bool = False,
    plot_ntwk: bool = True,
    xlims: Optional[tuple[float, float]] = None,
) -> Callable[[COM, str, str, str, dict[str, str], Any, Any], None]:
    """
    Generate a pulse response plotting function for use with ``plot_group_samps()``.

    Args:
        zoom: Zoom mode.

    Keyword Args:
        noeq: Plot unequalized pulse response when ``True``.
            Default: ``False``
        nopkg: Plot raw channel pulse response when ``True``.
            (Takes priority over ``noeq``.)
            Default: ``False``
        plot_ntwk: Add SciKit-RF pulse response estimate to plot when ``True``.
            (Only valid when either ``nopkg`` or ``noeq`` is ``True``.)
            Default: ``True``
        xlims: X-axis min. & max., for use w/ `zoom` = MANUAL.
            Default: None

    Returns:
        Pulse response plotting function suitable for sending to ``plot_group_samps()``.

    ToDo:
        1. Add a fourth pulse response option: pre-FFE.
    """
    
    def plot_pulse_resps(
        com: COM, grp: str, lbl: str, nm: str, opts: dict[str, str], ax1: Any, ax2: Any
    ) -> None:
        """
        Plot pulse response.
    
        Args:
            com: The COM instance to use for plotting.
                (Should already have been called, to optimize EQ.)
            grp: The channel group to use for plotting.
                (Should match the name of an immediate subfolder of your top-level channel data folder.)
            lbl: The channel set name within the channel group.
                (Should match the stem of the "<lbl>_{THRU,NEXT,FEXT}.s4p" file names.)
            nm: Extra identification information available to caller
                (e.g. - "MMSE" vs. "PRZF").
            opts: Plotting options.
                (See the ``matplotlib.pyplot`` documentation.)
            ax1: Axis to use for plotting against the left y-axis.
            ax2: Axis to use for plotting against the right y-axis.            
        """
        
        ui    = com.ui
        nspui = com.nspui
        t     = com.times
        Av    = com.com_params.A_v
        nRxTaps    = len(com.com_params.rx_taps_min)
        nRxPreTaps = com.com_params.dw

        # Find cursor location.
        if nopkg:
            curs_ix = np.argmax(com.pulse_resps_nopkg[0])
        elif noeq:
            curs_ix = np.argmax(com.pulse_resps_noeq[0])
        else:
            curs_ix = com.fom_rslts["cursor_ix"]

        # Plot the data.
        clr = opts["color"]
        if nopkg:
            ax1.plot(t * 1e9, com.pulse_resps_nopkg[0] * 1e3, label=nm, color=clr)
            if plot_ntwk and (nm == "MMSE" or nm == "PyChOpMarg"):
                _, y = s2p_pulse_response(com.chnls_noPkg[0][0][0], ui, t)
                ax1.plot(t * 1e9, y * Av * 1e3, label="SciKit-RF", color=clr, linestyle="dashed")
        elif noeq:
            ax1.plot(t * 1e9, com.pulse_resps_noeq[0] * 1e3, label=nm, color=clr)
            if plot_ntwk and nm == "MMSE":
                _, y = s2p_pulse_response(com.chnls[0][0][0], ui, t)
                ax1.plot(t * 1e9, y * Av * 1e3, label="SciKit-RF", color=clr, linestyle="dashed")
        else:
            ax1.plot(t * 1e9, com.com_rslts["pulse_resps"][0] * 1e3, label=nm, color=clr)

        # Set x-limits appropriately, as per user requested zoom option.
        match zoom:
            case ZoomMode.FULL:
                xmin = t[0]  * 1e9
                xmax = t[-1] * 1e9
            case ZoomMode.ISI:
                first_ix = curs_ix  - 2 * nRxPreTaps * nspui
                last_ix  = first_ix + 4 * nRxTaps    * nspui
                xmin = t[first_ix] * 1e9
                xmax = t[last_ix]  * 1e9
            case ZoomMode.PULSE:
                first_ix = curs_ix  - nRxPreTaps * nspui
                last_ix  = first_ix + nRxTaps    * nspui
                xmin = t[first_ix] * 1e9
                xmax = t[last_ix]  * 1e9
                plt.axvline(t[curs_ix] * 1e9, color=clr, linestyle="-")
                if (not (nopkg or noeq)) and nm == "MMSE":
                    ax1.plot(com.com_rslts["tISI"] * 1e9, com.com_rslts["hISI"] * 1e3, "xk", label="ISI")
            case ZoomMode.MANUAL:
                assert xlims, ValueError(
                    "X-axis limits must be provided in manual zoom mode!"
                )
                xmin=xlims[0]
                xmax=xlims[1]
            case ZoomMode.RELATIVE:
                assert xlims, ValueError(
                    "X-axis limits must be provided in relative zoom mode!"
                )
                curs_t = t[curs_ix]
                xmin=xlims[0] + curs_t * 1e9
                xmax=xlims[1] + curs_t * 1e9
            case _:
                raise RuntimeError(
                    f"Unrecognized zoom mode value ({zoom}) received!"
                )

        # Finalize plot configuration.
        ax1.axis(xmin=xmin, xmax=xmax)
        ax1.legend(loc="upper right")
        ax1.grid()
        plt.title(f"{lbl[-25: -5]}")

    return plot_pulse_resps


def plot_H(H: Cvec) -> None:
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    plt.plot(abs(H))
    plt.subplot(122)
    plt.plot(np.angle(H))
