"""
Default view definition for `PyChOpMarg.COM` class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 26, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from enable.component_editor import ComponentEditor
from traitsui.api import (
    Action,
    FileEditor,
    Group,
    Handler,
    HGroup,
    Item,
    VGroup,
    View,
)


# Event handler.
class MyHandler(Handler):
    """This handler is instantiated by the View and handles user button clicks."""

    def do_calc_com(self, info):
        """Run the COM calculation."""
        the_pychopmarg = info.object
        try:
            the_pychopmarg()
        except Exception as err:
            the_pychopmarg.status_str = str(err)
            raise


# Buttons.
calc_com = Action(name="Calc. COM", action="do_calc_com")


# Formatting utilities.
def to_ns(t: float) -> str:
    """Return formatted time in nanoseconds."""
    return f"{t*1e9:6.2f}"


def to_GHz(f: float) -> str:
    """Return formatted frequency in GHz."""
    return f"{f/1e9:6.2f}"


def to_MHz(f: float) -> str:
    """Return formatted frequency in MHz."""
    return f"{f/1e6:6.2f}"


def to_pF(c: float) -> str:
    """Return formatted capacitance in pF."""
    return f"{c*1e12:6.2f}"


def to_nH(L: float) -> str:
    """Return formatted inductance in nH."""
    return f"{L*1e9:6.2f}"


def to_mV(v: float) -> str:
    """Return formatted voltage in mV."""
    return f"{v*1e3:6.2f}"


# Main window layout definition.
traits_view = View(
    Group(  # Members correspond to top-level tabs.
        VGroup(  # "Config." tab
            HGroup(
                VGroup(
                    HGroup(
                        Item(
                            name="fb", format_func=to_GHz, label="f_b",
                            tooltip="Baud. rate (GHz)",
                        ),
                        Item(label="GHz"),
                    ),
                    Item(
                        name="nspui", label="M",
                        tooltip="Samples per UI",
                    ),
                    HGroup(
                        Item(
                            name="tmax", label="t_max",
                            tooltip="Maximum simulation time (ns)",
                        ),
                        Item(label="ns"),
                    ),
                    HGroup(
                        Item(
                            name="fmax", format_func=to_GHz, label="f_max",
                            tooltip="Maximum system frequency (GHz)",
                        ),
                        Item(label="GHz"),
                    ),
                    HGroup(
                        Item(
                            name="fstep", format_func=to_MHz, label="f_step",
                            tooltip="System frequency step (MHz)",
                        ),
                        Item(label="MHz"),
                    ),
                    label="Time and Frequency",
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        Item(
                            name="Av", label="A_v",
                            tooltip="Victim Tx Vout (V)",
                        ),
                        Item(label="V"),
                    ),
                    HGroup(
                        Item(
                            name="Afe", label="A_fe",
                            tooltip="FEXT Tx Vout (V)",
                        ),
                        Item(label="V"),
                    ),
                    HGroup(
                        Item(
                            name="Ane", label="A_ne",
                            tooltip="NEXT Tx Vout (V)",
                        ),
                        Item(label="V"),
                    ),
                    Item(
                        name="L",
                        tooltip="Modulation levels",
                    ),
                    Item(
                        name="fr", label="f_r",
                        tooltip="AFE corner frequency (fb)",
                    ),
                    label="Voltage and Modulation",
                    show_border=True,
                ),
                VGroup(
                    Item(
                        name="DER0", label="DER_0",
                        tooltip="Detector error rate threshold.",
                    ),
                    HGroup(
                        Item(
                            name="sigma_Rj", label="sigma_Rj",
                            tooltip="Random noise standard deviation (UI)",
                        ),
                        Item(label="UI"),
                    ),
                    HGroup(
                        Item(
                            name="Add", label="A_DD",
                            tooltip="Deterministic noise amplitude (UI)",
                        ),
                        Item(label="UI"),
                    ),
                    HGroup(
                        Item(
                            name="eta0", label="eta_0",
                            tooltip="Spectral noise density (V^2/GHz)",
                        ),
                        Item(label="V^2/GHz"),
                    ),
                    HGroup(
                        Item(
                            name="TxSNR", label="SNR_TX",
                            tooltip="Tx signal-to-noise ratio (dB)",
                        ),
                        Item(label="dB"),
                    ),
                    label="Noise and DER",
                    show_border=True,
                ),
                label="System",
                show_border=True,
            ),
            HGroup(
                VGroup(
                    HGroup(
                        Item(
                            name="chnl_s32p",
                            label="s32p",
                            editor=FileEditor(dialog_style="open", filter=["*.s32p"]),
                        ),
                        Item(
                            name="vic_chnl_ix", label="Vic_ID",
                            enabled_when="chnl_s32p",
                            tooltip="Victim channel index (from 1).",
                        ),
                    ),
                    HGroup(
                        VGroup(
                            Item(
                                name="chnl_s4p_thru",
                                label="s4p_THRU",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_fext1",
                                label="s4p_FEXT1",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_fext2",
                                label="s4p_FEXT2",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_fext3",
                                label="s4p_FEXT3",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_fext4",
                                label="s4p_FEXT4",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_fext5",
                                label="s4p_FEXT5",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_fext6",
                                label="s4p_FEXT6",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                        ),
                        VGroup(
                            Item(
                                label="Note: The `chnl_s32p` field, above, must be empty.",
                                ),
                            Item(
                                name="chnl_s4p_next1",
                                label="s4p_NEXT1",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_next2",
                                label="s4p_NEXT2",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_next3",
                                label="s4p_NEXT3",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_next4",
                                label="s4p_NEXT4",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_next5",
                                label="s4p_NEXT5",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                            Item(
                                name="chnl_s4p_next6",
                                label="s4p_NEXT6",
                                editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                                enabled_when="not chnl_s32p",
                            ),
                        ),
                    ),
                    label="Interconnect",
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        VGroup(
                            HGroup(
                                Item(
                                    name="R0", label="R_0",
                                    tooltip="System reference impedance (Ohms)",
                                ),
                                Item(label="Ohms"),
                            ),
                            HGroup(
                                Item(
                                    name="Rd", label="R_d",
                                    tooltip="On-die termination impedance (Ohms)",
                                ),
                                Item(label="Ohms"),
                            ),
                            HGroup(
                                Item(
                                    name="zp", label="z_p A",
                                    tooltip="Package transmission line length A",
                                ),
                                Item(label="mm"),
                                Item(
                                    name="zp_B", label="z_p B", style="readonly",
                                    tooltip="Package transmission line length B",
                                ),
                                Item(label="mm"),
                            ),
                            HGroup(
                                Item(
                                    name="Ls", format_func=to_nH, label="L_s",
                                    tooltip="Die parasitic inductances (nH)",
                                ),
                                Item(label="nH"),
                            ),
                        ),
                        VGroup(
                            HGroup(
                                Item(
                                    name="Cb", format_func=to_pF, label="C_b",
                                    tooltip="Bumb parasitic capacitance (pF)",
                                ),
                                Item(label="pF"),
                            ),
                            HGroup(
                                Item(
                                    name="Cp", format_func=to_pF, label="C_p",
                                    tooltip="Ball parasitic capacitance (pF)",
                                ),
                                Item(label="pF"),
                            ),
                            Item(label="Note: `L_s` & `C_d` in left->right order."),
                            HGroup(
                                Item(
                                    name="Cd", format_func=to_pF, label="C_d",
                                    tooltip="Die parasitic capacitances (pF)",
                                ),
                                Item(label="pF"),
                            ),
                        ),
                    ),
                    label="Package",
                    show_border=True,
                ),
                label="Channel",
                show_border=True,
            ),
            VGroup(
                Item(name="opt_mode", label="Optimization Mode",
                     tooltip="Linear EQ optimization mode."),
                HGroup(
                    VGroup(
                        Item(
                            name="c0_min", label="c(0)",
                            tooltip="Minimum allowed main tap weight.",
                        ),
                        Item(
                            name="tx_taps_pos", label="c_pos",
                            style="readonly", format_str="%10d",
                            tooltip="Tap positions, relative to main cursor.",
                        ),
                        Item(
                            name="tx_taps_min", label="c_min",
                            tooltip="Minimum tap weights.",
                        ),
                        Item(
                            name="tx_taps_max", label="c_max",
                            tooltip="Maximum tap weights.",
                        ),
                        Item(
                            name="tx_taps_step", label="c_step",
                            tooltip="Tap weight steps.",
                        ),
                        Item(
                            name="tx_taps", label="c_val",
                            tooltip="Tap weight values.",
                        ),
                        label="Tx FFE",
                        show_border=True,
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                name="gDC", label="g_DC",
                                tooltip="CTLE d.c. gain 1 (dB)",
                            ),
                            Item(
                                name="gDC2", label="g_DC2",
                                tooltip="CTLE d.c. gain 2 (dB)",
                            ),
                        ),
                        HGroup(
                            Item(
                                name="fz", format_func=to_GHz, label="f_z",
                                tooltip="CTLE zero frequency (GHz)",
                            ),
                            Item(label="GHz"),
                        ),
                        HGroup(
                            Item(
                                name="fp1", format_func=to_GHz, label="f_p1",
                                tooltip="CTLE first pole frequency (GHz)",
                            ),
                            Item(label="GHz"),
                        ),
                        HGroup(
                            Item(
                                name="fp2", format_func=to_GHz, label="f_p2",
                                tooltip="CTLE second pole frequency (GHz)",
                            ),
                            Item(label="GHz"),
                        ),
                        HGroup(
                            Item(
                                name="fLF", format_func=to_MHz, label="f_LF",
                                tooltip="CTLE low-f corner frequency (MHz)",
                            ),
                            Item(label="MHz"),
                        ),
                        label="Rx CTLE",
                        show_border=True,
                    ),
                    VGroup(
                        Item(
                            name="fLF", format_func=to_MHz, label="f_LF",
                            tooltip="CTLE low-f corner frequency (MHz)",
                        ),
                        label="Rx FFE/DFE",
                        show_border=True,
                    ),
                ),
                label="Equalization",
                show_border=True,
            ),
            label="Config.",
        ),
        VGroup(  # "Results" tab
            HGroup(
                HGroup(
                    VGroup(
                        HGroup(
                            HGroup(
                                Item(name="com", label="COM", format_str="%4.1f", style="readonly"),
                                Item(label="dB"),
                            ),
                            Item(label="     "),
                            HGroup(
                                Item(name="com_As", label="As", format_func=to_mV, style="readonly"),
                                Item(label="mV"),
                            ),
                            Item(label="     "),
                            HGroup(
                                Item(name="Ani", label="Ani", format_func=to_mV, style="readonly"),
                                Item(label="mV"),
                            ),
                        ),
                        Item(name="com_cursor_ix", label="CurPos", style="readonly"),
                        Item(name="com_dfe_taps", label="Rx DFE", style="readonly"),
                        label="Results",
                        show_border=True,
                    ),
                    VGroup(
                        Item(name="com_sigma_Tx", label="Tx", format_func=to_mV, style="readonly"),
                        Item(name="com_sigma_N", label="N", format_func=to_mV, style="readonly"),
                        Item(name="com_sigma_G", label="G", format_func=to_mV, style="readonly"),
                        label="Noise Sigmas (mV)",
                        show_border=True,
                    ),
                    label="COM",
                    show_border=True,
                ),
                HGroup(
                    VGroup(
                        HGroup(
                            HGroup(
                                Item(name="fom", label="FOM", format_str="%4.1f", style="readonly"),
                                Item(label="dB"),
                            ),
                            Item(label="     "),
                            HGroup(
                                Item(name="fom_As", label="As", format_func=to_mV, style="readonly"),
                                Item(label="mV"),
                            ),
                        ),
                        HGroup(
                            Item(name="fom_cursor_ix", label="CurPos", style="readonly"),
                            Item(label="     "),
                            HGroup(
                                Item(name="gDC", label="gDC", format_str="%4.1f", style="readonly"),
                                Item(label="dB"),
                            ),
                            Item(label="     "),
                            HGroup(
                                Item(name="gDC2", label="gDC2", format_str="%4.1f", style="readonly"),
                                Item(label="dB"),
                            ),
                        ),
                        Item(name="fom_dfe_taps", label="Rx DFE", style="readonly"),
                        Item(name="fom_tx_taps", label="Tx FFE", style="readonly"),
                        label="Results",
                        show_border=True,
                    ),
                    VGroup(
                        Item(name="sigma_Tx", label="Tx", format_func=to_mV, style="readonly"),
                        Item(name="sigma_N", label="N", format_func=to_mV, style="readonly"),
                        Item(name="sigma_ISI", label="ISI", format_func=to_mV, style="readonly"),
                        Item(name="sigma_J", label="J", format_func=to_mV, style="readonly"),
                        Item(name="sigma_XT", label="XT", format_func=to_mV, style="readonly"),
                        label="Noise Sigmas (mV)",
                        show_border=True,
                    ),
                    label="FOM",
                    show_border=True,
                ),
            ),
            Item("cont1", editor=ComponentEditor(high_resolution=False), show_label=False, springy=True),
            label="Results",
        ),
        VGroup(  # "About" tab
            Item("about_str", style="readonly", show_label=False,),
            label="About",
        ),
        layout="tabbed",
        springy=True,
        id="tabs",
    ),
    resizable=True,
    handler=MyHandler(),
    buttons=[calc_com],
    statusbar="status_str",
    title="PyChOpMarg",
    # icon=ImageResource("icon.png"),
)
