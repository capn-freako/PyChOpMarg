"""Tests for the ``pychopmarg`` command line interface."""

import os

import pytest
from click.testing import CliRunner

import pychopmarg.cli
from pychopmarg import __version__
from pychopmarg.cli import cli
from pychopmarg.common import PROCS_ENV_VAR
from pychopmarg.utility import jupyter

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


@pytest.mark.parametrize("args, n_procs", [([], None), (["--procs", "4"], 4), (["-p", "0"], 0)])
def test_run_notebook_procs(tmp_path, monkeypatch, args, n_procs):
    "``run-notebook --procs N`` hands N to the notebook runner; w/o it, the environment's default stands."
    calls = []
    monkeypatch.setattr(pychopmarg.cli, "execute_notebook", lambda nb, n_procs=None: calls.append((nb, n_procs)))
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}")
    result = CliRunner().invoke(cli, ["run-notebook", str(nb), *args])
    assert result.exit_code == 0, result.output
    assert calls == [(str(nb), n_procs)]


def test_run_notebook_procs_must_be_non_negative(tmp_path):
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}")
    result = CliRunner().invoke(cli, ["run-notebook", str(nb), "--procs", "-1"])
    assert result.exit_code != 0
    assert "--procs" in result.output


@pytest.mark.parametrize("n_procs", [None, 3])
def test_execute_notebook_passes_procs_to_its_kernel(tmp_path, monkeypatch, n_procs):
    "The notebook's kernel gets ``n_procs`` in ``PYCHOPMARG_PROCS`` (``COM``'s default), or the environment as is."
    runs = []
    monkeypatch.setattr(jupyter.subprocess, "run", lambda cmd, check, env: runs.append(env))
    monkeypatch.setenv(PROCS_ENV_VAR, "7")
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}")
    jupyter.execute_notebook(nb, n_procs=n_procs)
    if n_procs is None:
        assert runs == [None]  # (Inherits ours.)
    else:
        assert runs[0][PROCS_ENV_VAR] == "3" and runs[0]["PATH"] == os.environ["PATH"]
