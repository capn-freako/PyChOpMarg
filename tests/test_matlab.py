"""Tests for ``pychopmarg.matlab``, against a stand-in for the MATLAB Engine API (no MATLAB needed)."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pychopmarg import matlab
from pychopmarg.matlab import MFILES_DIR, run_com_matlab


class MatlabExecutionError(Exception):
    "Stand-in for ``matlab.engine.MatlabExecutionError``."


class FakeEngine:
    "Records how it's driven; COM \"fails\" for a thru channel named 'bad'."

    def __init__(self):
        self.paths, self.cwd, self.calls, self.quit_called = [], None, [], False

    def addpath(self, path, nargout=0):
        self.paths.append(path)

    def cd(self, path, nargout=0):
        self.cwd = path

    def pychopmarg_com(self, *args, nargout=1, stdout=None, stderr=None):
        self.calls.append(args)
        stdout.write("COM console output\n")
        if Path(args[2]).stem == "bad":
            raise MatlabExecutionError("Error using com: bad channel")
        return [{"channel_operating_margin_dB": 3.5, "itick": 12.0, "base": "thru",
                 "FIR": [[0.1, 0.2, 0.3]]}]  # (A MATLAB row vector arrives as a nested sequence.)

    def quit(self):
        self.quit_called = True


@pytest.fixture
def engine(monkeypatch):
    "Install a fake `matlab.engine`; return the engine it starts."
    eng = FakeEngine()
    mod_engine = types.ModuleType("matlab.engine")
    mod_engine.start_matlab = lambda: eng
    mod_engine.MatlabExecutionError = MatlabExecutionError
    mod_matlab = types.ModuleType("matlab")
    mod_matlab.engine = mod_engine
    monkeypatch.setitem(sys.modules, "matlab", mod_matlab)
    monkeypatch.setitem(sys.modules, "matlab.engine", mod_engine)
    return eng


def chnl_set(thru, fext=(), next_=()):
    return {"THRU": [Path(thru)], "FEXT": [Path(f) for f in fext], "NEXT": [Path(f) for f in next_]}


def test_runs_each_channel_set_through_the_wrapper(engine, tmp_path):
    com_file = tmp_path / "com" / "com_ieee8023_480.m"
    rslts = run_com_matlab(
        [("grp", chnl_set("a/thru.s4p", ["a/fext1.s4p", "a/fext2.s4p"], ["a/next1.s4p"]))],
        Path("cfg.xls"), com_file)
    # The session: COM's folder and our wrapper on the path; started and quit by this call.
    assert str(com_file.parent.resolve()) in engine.paths
    assert str(MFILES_DIR) in engine.paths
    assert engine.quit_called
    # The call: COM's function name, then absolute paths; aggressor lists stay lists (MATLAB cell arrays).
    (com_func, cfg, thru, fexts, nexts, pulse_resps), = engine.calls
    assert com_func == "com_ieee8023_480"
    assert Path(cfg).is_absolute() and Path(cfg).name == "cfg.xls"
    assert Path(thru).name == "thru.s4p"
    assert [Path(f).name for f in fexts] == ["fext1.s4p", "fext2.s4p"]
    assert [Path(f).name for f in nexts] == ["next1.s4p"]
    assert pulse_resps is False
    # The results: structured data, not console text.
    rslt = rslts["grp"]["thru"]
    assert rslt["error"] is None
    (tc,) = rslt["results"]
    assert tc["channel_operating_margin_dB"] == 3.5
    assert int(tc["itick"]) == 12
    np.testing.assert_array_equal(tc["FIR"], [0.1, 0.2, 0.3])
    assert rslt["stdout"] == "COM console output\n"


def test_failure_is_recorded_and_the_rest_still_run(engine):
    rslts = run_com_matlab([("g", chnl_set("bad.s4p")), ("g", chnl_set("good.s4p"))], Path("cfg.xls"),
                           Path("com_ieee8023_480.m"))
    assert "bad channel" in rslts["g"]["bad"]["error"]
    assert rslts["g"]["bad"]["results"] == []
    assert rslts["g"]["good"]["results"][0]["channel_operating_margin_dB"] == 3.5
    assert engine.quit_called


def test_reuses_a_given_engine(engine):
    eng = matlab.start_matlab(Path("com_ieee8023_480.m"))
    run_com_matlab([("g", chnl_set("t.s4p"))], Path("cfg.xls"), Path("com_ieee8023_480.m"), eng=eng)
    assert not engine.quit_called  # The caller owns it.


def test_missing_engine_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "matlab", None)  # Makes `import matlab.engine` raise ImportError.
    with pytest.raises(ImportError, match="matlabengine"):
        run_com_matlab([("g", chnl_set("t.s4p"))], Path("cfg.xls"), Path("com_ieee8023_480.m"))


def test_wrapper_ships_with_the_package():
    assert (MFILES_DIR / "pychopmarg_com.m").is_file()


def test_pulse_resps_requested(engine):
    run_com_matlab([("g", chnl_set("t.s4p"))], Path("cfg.xls"), Path("com_ieee8023_4p17p0.m"), pulse_resps=True)
    assert engine.calls[0][5] is True
