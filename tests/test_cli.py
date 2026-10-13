"""Tests for the ``pychopmarg`` command line interface."""

import pytest
from click.testing import CliRunner

from pychopmarg import __version__
from pychopmarg.cli import cli

COMMANDS = [[], ["run-notebook"], ["convert-notebook"]]


@pytest.mark.parametrize("help_opt", ["--help", "-h"])
@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: " ".join(c) or "pychopmarg")
def test_help(cmd, help_opt):
    """Every command honors its ``context_settings``: both ``--help`` and ``-h`` print its usage."""
    result = CliRunner().invoke(cli, [*cmd, help_opt])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
