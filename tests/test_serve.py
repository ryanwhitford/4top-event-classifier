"""Tests for the FastAPI serving app.

These exercise the real committed model artifact (models/model.joblib +
models/metadata.json) end to end through the HTTP layer, so a broken
contract between train.py and serve.py (feature order, missing metadata
keys, schema drift) fails CI immediately -- no data download required,
since the model artifact is small enough to commit to the repo.
"""

import pytest
from fastapi.testclient import TestClient

from src.serve import app

client = TestClient(app)

# A "loud" event: high multiplicity, four b-jets, large HT and MET -- the
# textbook four-top signature described in src/features.py's module docstring.
SIGNAL_LIKE_EVENT = {
    "met": 180.0,
    "met_phi": 0.3,
    "objects": [
        {"kind": "b", "E": 220, "pt": 190, "eta": 0.4, "phi": 0.1},
        {"kind": "b", "E": 200, "pt": 170, "eta": -0.3, "phi": 1.2},
        {"kind": "b", "E": 190, "pt": 160, "eta": 0.9, "phi": -1.5},
        {"kind": "b", "E": 180, "pt": 150, "eta": -0.9, "phi": 2.5},
        {"kind": "j", "E": 150, "pt": 130, "eta": 1.5, "phi": -0.5},
        {"kind": "j", "E": 140, "pt": 120, "eta": -1.2, "phi": 1.9},
        {"kind": "e-", "E": 90, "pt": 85, "eta": 0.5, "phi": -2.0},
        {"kind": "m-", "E": 80, "pt": 75, "eta": -0.6, "phi": 0.8},
    ],
}

# A near-empty event: minimal activity, nothing like a same-sign multi-lepton
# four-top candidate -- the textbook background-like shape.
QUIET_EVENT = {"met": 5.0, "met_phi": 0.0, "objects": []}


def test_health_reports_model_loaded():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "four-top-event-classifier"


def test_model_info_shape():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    for key in ("model_type", "sklearn_version", "trained_at", "n_features", "threshold", "test_metrics"):
        assert key in body
    assert body["n_features"] > 0
    assert 0.0 < body["threshold"] < 1.0


def test_predict_signal_like_event_scores_higher_than_quiet_event():
    r_signal = client.post("/predict", json=SIGNAL_LIKE_EVENT)
    r_quiet = client.post("/predict", json=QUIET_EVENT)
    assert r_signal.status_code == 200
    assert r_quiet.status_code == 200

    signal_body = r_signal.json()
    quiet_body = r_quiet.json()

    for body in (signal_body, quiet_body):
        assert 0.0 <= body["score"] <= 1.0
        assert body["label"] in (0, 1)
        assert body["prediction"] == ("signal" if body["label"] == 1 else "background")
        assert body["threshold"] == pytest.approx(signal_body["threshold"])

    # Not a strict contract on the tuned model's exact output, but the whole
    # point of the physics-informed features (HT, b-jet count, MET) is that a
    # loud, high-multiplicity, four-b-jet event should score well above a
    # near-empty one.
    assert signal_body["score"] > quiet_body["score"]


def test_predict_reports_drift_for_out_of_distribution_event():
    r = client.post("/predict", json=QUIET_EVENT)
    assert r.status_code == 200
    drift = r.json()["drift"]
    assert drift["checked"] is True
    # A zero-object event is far outside the training distribution on at
    # least the multiplicity features -- the drift check should catch it.
    assert drift["n_drifting_features"] > 0


def test_predict_rejects_unknown_object_kind():
    bad = {"met": 5.0, "objects": [{"kind": "x", "E": 1.0, "pt": 1.0, "eta": 0.0, "phi": 0.0}]}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_negative_energy():
    bad = {"met": 5.0, "objects": [{"kind": "j", "E": -1.0, "pt": 1.0, "eta": 0.0, "phi": 0.0}]}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_met_is_rejected():
    r = client.post("/predict", json={"objects": []})
    assert r.status_code == 422
