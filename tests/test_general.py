"""Tests for ``pychopmarg.utility.general``."""

import pytest

from pychopmarg.utility import get_channel_sets


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def test_get_channel_sets(tmp_path):
    "Thru channels and their same-named aggressors, by group; nested channels only when `recursive`."
    _touch(tmp_path / "KR" / "a_thru.s4p")
    _touch(tmp_path / "KR" / "a_next1.s4p")
    _touch(tmp_path / "KR" / "a_fext1.s4p")
    _touch(tmp_path / "KR" / "contrib" / "b_THRU.s4p")
    _touch(tmp_path / "KR" / "contrib" / "b_NEXT1.s4p")
    _touch(tmp_path / "KR" / "a_next2.s4p.bak")  # Not a channel.
    flat = get_channel_sets(tmp_path)
    assert [cs["THRU"][0].name for cs in flat["KR"]] == ["a_thru.s4p"]
    assert [p.name for p in flat["KR"][0]["NEXT"]] == ["a_next1.s4p"]
    assert [p.name for p in flat["KR"][0]["FEXT"]] == ["a_fext1.s4p"]
    deep = get_channel_sets(tmp_path, recursive=True)
    assert [cs["THRU"][0].name for cs in deep["KR"]] == ["a_thru.s4p", "b_THRU.s4p"]
    assert [p.name for p in deep["KR"][1]["NEXT"]] == ["b_NEXT1.s4p"]  # Its own directory's aggressors.
    assert deep["KR"][1]["FEXT"] == []


@pytest.mark.parametrize("thru, aggressors", [
    ("a_thru.s4p", {"a_next1.s4p": "NEXT", "a_fext1.s4p": "FEXT"}),                      # The original convention.
    ("a_THRU.s4p", {"a_extra_NEXT1.s4p": "NEXT", "a_more_FEXT2.s4p": "FEXT"}),           # Text before the type.
    ("a_thru1.s4p", {"a_xtalk1_Fext.s4p": "FEXT", "a_xtalk4_Next.s4p": "NEXT"}),         # Index before the type.
    ("a_thru_TP0_Tx7_to_TP1a_Tx7.s4p", {"a_FEXT_TP0_Tx1_to_TP1a_Tx7.s4p": "FEXT"}),     # A different suffix.
])
def test_aggressor_naming_conventions(tmp_path, thru, aggressors):
    "Contributors name aggressor files variously; each goes to the thru channel whose name it extends."
    for name in [thru, *aggressors, "b_next1.s4p", "a_next1.s4p.bak"]:  # (Another channel's; not Touchstone.)
        _touch(tmp_path / "G" / name)
    (cs,) = get_channel_sets(tmp_path)["G"]
    for kind in ("NEXT", "FEXT"):
        assert [p.name for p in cs[kind]] == sorted(n for n, k in aggressors.items() if k == kind)


def test_aggressors_go_to_the_longest_matching_thru(tmp_path):
    "With thru channels `a_thru` and `a_b_thru`, `a_b_next1` is the latter's, not both's."
    for name in ["a_thru.s4p", "a_b_thru.s4p", "a_next1.s4p", "a_b_next1.s4p"]:
        _touch(tmp_path / "G" / name)
    sets = {cs["THRU"][0].name: [p.name for p in cs["NEXT"]] for cs in get_channel_sets(tmp_path)["G"]}
    assert sets == {"a_thru.s4p": ["a_next1.s4p"], "a_b_thru.s4p": ["a_b_next1.s4p"]}
