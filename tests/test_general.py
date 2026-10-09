"""Tests for ``pychopmarg.utility.general``."""

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
