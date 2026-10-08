"""Every module in the package imports cleanly.

Catches import-time errors (e.g. a ``NameError`` from a dict literal with an unquoted key, in a
module-level decorator) in modules no other test happens to import.
"""

import importlib
import pkgutil

import pytest

import pychopmarg

MODULES = sorted(m.name for m in pkgutil.walk_packages(pychopmarg.__path__, prefix="pychopmarg."))


def test_found_modules():
    assert "pychopmarg.cli" in MODULES


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module(module)
