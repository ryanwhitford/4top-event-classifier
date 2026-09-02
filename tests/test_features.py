import math

import pytest

from src.features import FEATURE_GROUPS, build_dataframe, event_features, feature_columns
from src.parse import Event, Obj


def make_event(process="4top", weight=1e-3, objects=None, met=50.0, met_phi=0.0):
    return Event(
        event_id="1",
        process=process,
        weight=weight,
        met=met,
        met_phi=met_phi,
        objects=objects or [],
    )


def test_event_features_empty_event():
    ev = make_event(objects=[])
    f = event_features(ev)
    assert f["n_jets_all"] == 0
    assert f["n_bjets"] == 0
    assert f["n_leptons"] == 0
    assert f["b_fraction"] == 0.0
    assert f["ht"] == 0.0
    assert math.isnan(f["jet1_pt"])
    assert math.isnan(f["m_bb"])
    assert math.isnan(f["min_dr_jj"])
    assert f["has_same_sign_pair"] == 0.0


def test_event_features_multiplicity_and_ht():
    objects = [
        Obj(kind="b", E=100.0, pt=80.0, eta=0.1, phi=0.0),
        Obj(kind="j", E=60.0, pt=50.0, eta=-0.2, phi=1.0),
        Obj(kind="j", E=40.0, pt=30.0, eta=0.5, phi=-1.0),
    ]
    ev = make_event(objects=objects)
    f = event_features(ev)
    assert f["n_jets_all"] == 3
    assert f["n_bjets"] == 1
    assert f["n_jets_light"] == 2
    assert f["b_fraction"] == pytest.approx(1 / 3)
    assert f["ht"] == pytest.approx(80.0 + 50.0 + 30.0)
    assert f["ht_b"] == pytest.approx(80.0)
    # leading jet (by pT, across b+light) should be the b-jet
    assert f["jet1_pt"] == pytest.approx(80.0)


def test_event_features_same_sign_pair_detection():
    same_sign = [
        Obj(kind="m-", E=50.0, pt=40.0, eta=0.0, phi=0.0),
        Obj(kind="e-", E=50.0, pt=30.0, eta=0.0, phi=0.0),
    ]
    opp_sign = [
        Obj(kind="m-", E=50.0, pt=40.0, eta=0.0, phi=0.0),
        Obj(kind="e+", E=50.0, pt=30.0, eta=0.0, phi=0.0),
    ]
    assert event_features(make_event(objects=same_sign))["has_same_sign_pair"] == 1.0
    assert event_features(make_event(objects=opp_sign))["has_same_sign_pair"] == 0.0


def test_event_features_lepton_charge_sum():
    objects = [
        Obj(kind="m-", E=50.0, pt=40.0, eta=0.0, phi=0.0),
        Obj(kind="e-", E=50.0, pt=30.0, eta=0.0, phi=0.0),
    ]
    f = event_features(make_event(objects=objects))
    assert f["lepton_charge_sum"] == -2.0
    assert f["abs_lepton_charge_sum"] == 2.0


def test_event_features_single_lepton_has_no_same_sign_pair():
    objects = [Obj(kind="e-", E=50.0, pt=30.0, eta=0.0, phi=0.0)]
    f = event_features(make_event(objects=objects))
    assert f["has_same_sign_pair"] == 0.0
    assert not math.isnan(f["mt_lep1_met"])


def test_build_dataframe_labels_signal_processes():
    events = [
        make_event(process="4top"),
        make_event(process="ttbarZ"),
        make_event(process="ttbarW"),
    ]
    df = build_dataframe(events, signal_processes=["4top"])
    assert df["label"].tolist() == [1, 0, 0]
    assert df["process"].tolist() == ["4top", "ttbarZ", "ttbarW"]


def test_build_dataframe_is_case_insensitive_on_signal():
    events = [make_event(process="4Top")]
    df = build_dataframe(events, signal_processes=["4top"])
    assert df["label"].tolist() == [1]


def test_feature_columns_excludes_metadata():
    events = [make_event(process="4top")]
    df = build_dataframe(events, signal_processes=["4top"])
    cols = feature_columns(df)
    assert "process" not in cols
    assert "weight" not in cols
    assert "label" not in cols
    assert "n_jets_all" in cols


def test_feature_groups_columns_are_all_real_features():
    events = [make_event(process="4top")]
    df = build_dataframe(events, signal_processes=["4top"])
    cols = set(feature_columns(df))
    for group, feats in FEATURE_GROUPS.items():
        missing = set(feats) - cols
        assert not missing, f"group {group!r} references unknown columns: {missing}"
