"""
Default view definition for `PyChOpMarg.COM` class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 26, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

# from enable.component_editor import ComponentEditor
from numpy                   import log10, array
from pyface.image_resource   import ImageResource
from traitsui.api import (  # CloseAction,
    Action,
    CheckListEditor,
    FileEditor,
    Group,
    HGroup,
    Item,
    Menu,
    MenuBar,
    NoButtons,
    ObjectColumn,
    RangeEditor,
    Separator,
    TableEditor,
    TextEditor,
    VGroup,
    View,
    spring,
)
from traitsui.api import ListStrEditor, UItem
from traitsui.ui_editors.array_view_editor import ArrayViewEditor

# Main window layout definition.
traits_view = View(
    Group(  # Members correspond to top-level tabs.
        VGroup(  # "Config." tab
            HGroup(
                VGroup(
                    HGroup(
                        Item(
                            name="fb",
                            # label="fb",
                            tooltip="Baud. rate (GHz)",
                            # show_label=True,
                            # enabled_when="True",
                            # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(label="GHz"),
                    ),
                    Item(
                        name="nspui",
                        # label="nspui",
                        tooltip="Samples per UI",
                    ),
                    HGroup(
                        Item(
                            name="tmax",
                            # label="fb",
                            tooltip="Maximum simulation time (ns)",
                            # show_label=True,
                            # enabled_when="True",
                            # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(label="ns"),
                    ),
                    HGroup(
                        Item(
                            name="fmax",
                            # label="fb",
                            tooltip="Maximum system frequency (GHz)",
                            # show_label=True,
                            # enabled_when="True",
                            # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(label="GHz"),
                    ),
                    HGroup(
                        Item(
                            name="fstep",
                            # label="fb",
                            tooltip="System frequency step (MHz)",
                            # show_label=True,
                            # enabled_when="True",
                            # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(label="MHz"),
                    ),
                    label="Time and Frequency",
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        Item(
                            name="Av",
                            # label="fb",
                            tooltip="Victim Tx Vout (V)",
                            # show_label=True,
                            # enabled_when="True",
                            # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(label="V"),
                    ),
                    HGroup(
                        Item(
                            name="Afe",
                            # label="fb",
                            tooltip="FEXT Tx Vout (V)",
                            # show_label=True,
                            # enabled_when="True",
                            # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(label="V"),
                    ),
                    HGroup(
                        Item(
                            name="Ane",
                            # label="fb",
                            tooltip="NEXT Tx Vout (V)",
                            # show_label=True,
                            # enabled_when="True",
                            # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                        ),
                        Item(label="V"),
                    ),
                    Item(
                        name="L",
                        # label="fb",
                        tooltip="Modulation levels",
                        # show_label=True,
                        # enabled_when="True",
                        # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                    ),
                    Item(
                        name="fr",
                        tooltip="AFE corner frequency (fb)",
                    ),
                    label="Voltage and Modulation",
                    show_border=True,
                ),
                VGroup(
                    Item(
                        name="DER0",
                        tooltip="Detector error rate threshold.",
                    ),
                    HGroup(
                        Item(
                            name="sigma_Rj",
                            tooltip="Random noise standard deviation (UI)",
                        ),
                        Item(label="UI"),
                    ),
                    HGroup(
                        Item(
                            name="Add",
                            tooltip="Deterministic noise amplitude (UI)",
                        ),
                        Item(label="UI"),
                    ),
                    HGroup(
                        Item(
                            name="eta0",
                            tooltip="Spectral noise density (V^2/GHz)",
                        ),
                        Item(label="V^2/GHz"),
                    ),
                    HGroup(
                        Item(
                            name="TxSNR",
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
                    Item(
                        name="chnl_s32p",
                        label="s32p",
                        editor=FileEditor(dialog_style="open", filter=["*.s32p"]),
                    ),
                    Item(
                        name="chnl_s4p_thru",
                        label="s4p_THRU",
                        editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                    ),
                    VGroup(
                        HGroup(
                            Item(
                                name="R0",
                                tooltip="System reference impedance (Ohms)",
                            ),
                            Item(label="Ohms"),
                        ),
                        HGroup(
                            Item(
                                name="Rd",
                                tooltip="On-die termination impedance (Ohms)",
                            ),
                            Item(label="Ohms"),
                        ),
                        HGroup(
                            Item(
                                name="Cd",
                                tooltip="Die parasitic capacitance (pF)",
                            ),
                            Item(label="pF"),
                        ),
                        HGroup(
                            Item(
                                name="Cb",
                                tooltip="Bumb parasitic capacitance (pF)",
                            ),
                            Item(label="pF"),
                        ),
                        HGroup(
                            Item(
                                name="Cp",
                                tooltip="Ball parasitic capacitance (pF)",
                            ),
                            Item(label="pF"),
                        ),
                        label="Package and Die",
                        show_border=True,
                    ),
                    label="Interconnect",
                    show_border=True,
                ),
                VGroup(
                    Item(
                        name="chnl_s4p_fext1",
                        label="s4p_FEXT1",
                        editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                    ),
                    Item(
                        name="chnl_s4p_fext2",
                        label="s4p_FEXT2",
                        editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                    ),
                    Item(
                        name="chnl_s4p_fext3",
                        label="s4p_FEXT3",
                        editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                    ),
                    Item(
                        name="chnl_s4p_next1",
                        label="s4p_NEXT1",
                        editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                    ),
                    Item(
                        name="chnl_s4p_next2",
                        label="s4p_NEXT2",
                        editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                    ),
                    Item(
                        name="chnl_s4p_next3",
                        label="s4p_NEXT3",
                        editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                    ),
                    Item(
                        name="chnl_s4p_next4",
                        label="s4p_NEXT4",
                        editor=FileEditor(dialog_style="open", filter=["*.s4p"]),
                    ),
                    HGroup(
                        Item(
                            name="zp",
                            tooltip="Package transmission line length (mm)",
                            editor=CheckListEditor(values=[(12, "12"), (30, "30")]),
                        ),
                        Item(label="mm"),
                    ),
                    label="Crosstalk",
                    show_border=True,
                ),
                label="Channel",
                show_border=True,
            ),
            HGroup(
                VGroup(
                    Item(
                        name="c0_min",
                        # label="nspui",
                        tooltip="Minimum allowed main tap weight.",
                    ),
                    Item(
                        name="tx_taps_min",
                        label="Min.",
                        tooltip="Minimum tap weights.",
                        # show_label=True,
                        # enabled_when="True",
                        # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                    ),
                    Item(
                        name="tx_taps_max",
                        label="Max.",
                        tooltip="Maximum tap weights.",
                        # show_label=True,
                        # enabled_when="True",
                        # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                    ),
                    Item(
                        name="tx_taps_step",
                        label="Step",
                        tooltip="Tap weight steps.",
                        # show_label=True,
                        # enabled_when="True",
                        # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                    ),
                    Item(
                        name="tx_taps",
                        label="Value",
                        tooltip="Tap weights.",
                        # show_label=True,
                        # enabled_when="True",
                        # editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                    ),
                    label="Tx FFE",
                    show_border=True,
                ),
                VGroup(
                    HGroup(
                        Item(
                            name="gDC",
                            tooltip="CTLE d.c. gain 1 (dB)",
                            editor=CheckListEditor(values=[(-n, str(n)) for n in range(13)]),
                        ),
                        Item(
                            name="gDC2",
                            tooltip="CTLE d.c. gain 2 (dB)",
                            editor=CheckListEditor(values=[(0, "0")]),
                        ),
                    ),
                    HGroup(
                        Item(
                            name="fz",
                            tooltip="CTLE zero frequency (GHz)",
                        ),
                        Item(label="GHz"),
                    ),
                    HGroup(
                        Item(
                            name="fp1",
                            tooltip="CTLE first pole frequency (GHz)",
                        ),
                        Item(label="GHz"),
                    ),
                    HGroup(
                        Item(
                            name="fp2",
                            tooltip="CTLE second pole frequency (GHz)",
                        ),
                        Item(label="GHz"),
                    ),
                    HGroup(
                        Item(
                            name="fLF",
                            tooltip="CTLE low-f corner frequency (MHz)",
                        ),
                        Item(label="MHz"),
                    ),
                    label="Rx CTLE",
                    show_border=True,
                ),
                label="Equalization",
                show_border=True,
            ),
            label="Config.",
        ),
        VGroup(  # "Results" tab
            Item("rslts"),
            Item("fom_rslts"),
            label="Results",
        ),
        layout="tabbed",
        springy=True,
        id="tabs",
    ),
    resizable=True,
    # handler=MyHandler(),
    buttons=NoButtons,
    # statusbar="status_str",
    title="PyChOpMarg",
    # icon=ImageResource("icon.png"),
)
