"""Validation of ``COMParams`` (``config/template.py``)."""

import dataclasses

import pytest

from pychopmarg.config.ieee_8023by import IEEE_8023by
from pychopmarg.config.ieee_8023dj import IEEE_8023dj
from pychopmarg.config.ieee_8023dj_d3p3 import com_params

SHIPPED = {
    "IEEE_8023dj": IEEE_8023dj,
    "IEEE_8023by": IEEE_8023by,
    **{f"D3.3 {g}": com_params(g) for g in ("KR", "C2C", "CR", "C2M")},
}


@pytest.mark.parametrize("params", SHIPPED.values(), ids=SHIPPED.keys())
def test_shipped_configs_validate(params):
    "Every shipped configuration passes `__post_init__`'s range checks."
    dataclasses.replace(params)


@pytest.mark.parametrize("name, value", [
    ("tx_taps_step", [0.02, 0.02, 0.3, 0.02, 0.0, 0.0]),
    ("g_DC", [-40.0, 0.0]),
    ("C_d", [4e-5, 5.0, 1.1e-4]),
])
def test_list_values_are_range_checked(name, value):
    "Each element of a list-valued parameter is range checked (they once weren't checked at all)."
    with pytest.raises(ValueError, match=rf"`{name}\[\d\]`"):
        dataclasses.replace(IEEE_8023dj, **{name: value})
