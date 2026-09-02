import math

import pytest

from src.parse import Obj, ParseError, find_data_files, parse_line, process_counts, read_events

RAW_LINE = (
    "70002;4top;2.9748e-08;234878;0.50707;"
    "b,288384,245272,-0.543959,-1.9042;"
    "j,392047,165199,1.50827,1.99874;"
    "e+,125602,95779.5,0.769969,-0.288634;"
)


def test_parse_line_fields():
    ev = parse_line(RAW_LINE)
    assert ev.event_id == "70002"
    assert ev.process == "4top"
    assert ev.weight == pytest.approx(2.9748e-08)
    assert ev.met == pytest.approx(234878 * 1e-3)
    assert ev.met_phi == pytest.approx(0.50707)
    assert len(ev.objects) == 3


def test_parse_line_scales_mev_to_gev():
    ev = parse_line(RAW_LINE)
    b = ev.objects[0]
    assert b.kind == "b"
    assert b.E == pytest.approx(288384 * 1e-3)
    assert b.pt == pytest.approx(245272 * 1e-3)


def test_parse_line_blank_and_comment_return_none():
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("# a comment") is None


def test_parse_line_exactly_five_fields_is_valid_with_no_objects():
    ev = parse_line("1;4top;1.0;10;0.1")
    assert ev.objects == []


def test_parse_line_too_few_fields_raises():
    with pytest.raises(ParseError):
        parse_line("1;4top;1.0")


def test_parse_line_malformed_object_raises():
    with pytest.raises(ParseError):
        parse_line("1;4top;1.0;10;0.1;j,1,2,3;")  # object group has 4 fields, not 5


def test_event_leptons_and_of_kind_sorted_by_pt():
    ev = parse_line(RAW_LINE)
    leptons = ev.leptons
    assert len(leptons) == 1
    assert leptons[0].kind == "e+"

    jets = ev.of_kind("j", "b")
    assert [o.kind for o in jets] == ["b", "j"]  # b-jet has higher pT
    assert jets[0].pt >= jets[1].pt


def test_obj_kinematics():
    o = Obj(kind="j", E=100.0, pt=30.0, eta=1.0, phi=0.5)
    assert o.px == pytest.approx(30.0 * math.cos(0.5))
    assert o.py == pytest.approx(30.0 * math.sin(0.5))
    assert o.pz == pytest.approx(30.0 * math.sinh(1.0))
    assert o.p == pytest.approx(30.0 * math.cosh(1.0))
    assert o.mass >= 0.0


def test_obj_mass_clips_negative_to_zero():
    # A wildly inconsistent (E, pt, eta) triple would give E^2 - p^2 < 0.
    o = Obj(kind="j", E=1.0, pt=100.0, eta=0.0, phi=0.0)
    assert o.mass == 0.0


def test_charge_and_lepton_flags():
    e_minus = Obj(kind="e-", E=1.0, pt=1.0, eta=0.0, phi=0.0)
    jet = Obj(kind="j", E=1.0, pt=1.0, eta=0.0, phi=0.0)
    assert e_minus.charge == -1
    assert e_minus.is_lepton
    assert jet.charge == 0
    assert not jet.is_lepton


def test_read_events_respects_limit(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text((RAW_LINE + "\n") * 5)
    events = list(read_events(path, limit=3))
    assert len(events) == 3


def test_read_events_skips_malformed_lines_when_not_strict(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(RAW_LINE + "\n" + "broken;line\n" + RAW_LINE + "\n")
    events = list(read_events(path, strict=False))
    assert len(events) == 2


def test_read_events_raises_when_strict(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(RAW_LINE + "\n" + "broken;line\n")
    with pytest.raises(ParseError):
        list(read_events(path, strict=True))


def test_find_data_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.csv").write_text("")
    (tmp_path / "sub" / "b.csv").write_text("")
    (tmp_path / "notes.txt").write_text("")
    found = find_data_files(tmp_path)
    assert {p.name for p in found} == {"a.csv", "b.csv"}


def test_process_counts_sorted_descending():
    events = [
        parse_line(RAW_LINE),
        parse_line(RAW_LINE.replace("4top", "ttbarZ")),
        parse_line(RAW_LINE.replace("4top", "ttbarZ")),
    ]
    counts = process_counts(events)
    assert list(counts.items())[0] == ("ttbarZ", 2)
    assert counts["4top"] == 1
