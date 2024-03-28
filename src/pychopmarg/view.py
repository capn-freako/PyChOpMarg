"""
Default view definition for `PyChOpMarg.COM` class.

Original author: David Banas <capn.freako@gmail.com>

Original date:   March 26, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.
"""

from enable.component_editor import ComponentEditor
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
from traitsui.ui_editors.array_view_editor import ArrayViewEditor

# Main window layout definition.
traits_view = View(
    Group(  # Members correspond to top-level tabs.
        VGroup(  # "Config." tab
            HGroup(
                Item(
                    name="ui",
                    label="UI",
                    tooltip="Unit Interval (ps)",
                    show_label=True,
                    enabled_when="True",
                    editor=TextEditor(auto_set=False, enter_set=True, evaluate=float),
                ),
                Item(label="ps")
            ),
            label="Config."
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
