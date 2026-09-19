"""
FastAPI serving app for the Four-Top Event Classifier.

Exposes the trained model (`models/model.joblib` + `models/metadata.json`,
produced by `src/train.py`) over HTTP. The prediction contract accepts the
same raw per-object event representation `src/parse.py` works with (MET plus
a list of reconstructed jets/leptons/photons), and reuses the exact
`event_features` function from `src/features.py` to build the model's input
row -- so there is exactly one place feature engineering happens, whether at
training time or serving time.

Endpoints:
    GET  /            basic service info
    GET  /health      liveness + whether the model loaded
    GET  /model-info  model metadata: features, threshold, training metrics
    POST /predict     score one event, with a lightweight drift check

Run locally:
    uvicorn src.serve:app --reload --port 8000

The drift check compares each incoming feature to its training-set mean/std
(from metadata.json) and flags features that land more than
DRIFT_Z_THRESHOLD standard deviations away. This is not a rigorous
population-level drift test (that needs a *batch* of requests, not one) --
it is a cheap per-request tripwire for "this event looks nothing like
training data", logged so it shows up in Application Insights once deployed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import event_features
from .parse import Event, Obj

logger = logging.getLogger("fourtop.serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# --------------------------------------------------------------------------
# Model + metadata loading
# --------------------------------------------------------------------------

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
DRIFT_Z_THRESHOLD = float(os.environ.get("DRIFT_Z_THRESHOLD", "5.0"))

model = None
metadata: dict = {}
load_error: str | None = None

try:
    model = joblib.load(MODEL_DIR / "model.joblib")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    logger.info(
        "Loaded model from %s (trained_at=%s, threshold=%.3f)",
        MODEL_DIR, metadata.get("trained_at"), metadata.get("threshold", float("nan")),
    )
except Exception as exc:  # noqa: BLE001 -- we want the app to boot regardless
    load_error = str(exc)
    logger.error("Failed to load model from %s: %s", MODEL_DIR, exc)

FEATURES: list[str] = metadata.get("features", [])
THRESHOLD: float = metadata.get("threshold", 0.5)
FEATURE_STATS: dict = metadata.get("feature_stats", {})


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------

ObjectKind = Literal["j", "b", "e-", "e+", "m-", "m+", "g"]


class ObjectIn(BaseModel):
    """One reconstructed object, in the same units src/parse.py produces (GeV)."""

    kind: ObjectKind = Field(..., description="j=jet, b=b-jet, e-/e+=electron, m-/m+=muon, g=photon")
    E: float = Field(..., ge=0, description="Energy [GeV]")
    pt: float = Field(..., ge=0, description="Transverse momentum [GeV]")
    eta: float = Field(..., description="Pseudorapidity")
    phi: float = Field(..., description="Azimuthal angle [rad]")


class PredictRequest(BaseModel):
    met: float = Field(..., ge=0, description="Missing transverse energy [GeV]")
    met_phi: float = Field(0.0, description="MET azimuthal angle [rad]")
    objects: list[ObjectIn] = Field(default_factory=list, description="Reconstructed objects in the event")


class DriftInfo(BaseModel):
    checked: bool
    n_drifting_features: int = 0
    drifting_features: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    score: float
    label: int
    prediction: Literal["signal", "background"]
    threshold: float
    drift: DriftInfo


class ModelInfoResponse(BaseModel):
    model_type: str
    sklearn_version: str
    trained_at: str
    n_features: int
    threshold: float
    signal_processes: list[str]
    test_metrics: dict
    dataset: dict


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(
    title="Four-Top Event Classifier",
    description="Classifies LHC collision events as four-top-quark signal vs. ttbar+boson background.",
    version="1.0.0",
)

# --- optional Application Insights telemetry --------------------------------
# Enabled only when APPLICATIONINSIGHTS_CONNECTION_STRING is set (wired up by
# infra/main.bicep as a Container App env var). Local dev and pytest runs
# never set it, so this is a no-op there. Once enabled, the azure-monitor SDK
# also picks up everything sent through the standard `logging` module -- so
# the drift warning logged in _check_drift() below shows up in Application
# Insights automatically, no separate telemetry call needed.
if os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        configure_azure_monitor()
        FastAPIInstrumentor.instrument_app(app)
        logger.info("Application Insights telemetry enabled")
    except Exception as exc:  # noqa: BLE001 -- telemetry must never block serving
        logger.warning("Could not enable Application Insights telemetry: %s", exc)


@app.get("/")
def root():
    return {
        "service": "four-top-event-classifier",
        "status": "ok" if model is not None else "degraded",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "unhealthy", "model_loaded": model is not None}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")
    return {
        "model_type": metadata.get("model_type", "unknown"),
        "sklearn_version": metadata.get("sklearn_version", "unknown"),
        "trained_at": metadata.get("trained_at", "unknown"),
        "n_features": metadata.get("n_features", len(FEATURES)),
        "threshold": THRESHOLD,
        "signal_processes": metadata.get("signal_processes", []),
        "test_metrics": metadata.get("test_metrics", {}),
        "dataset": metadata.get("dataset", {}),
    }


def _check_drift(row: dict[str, float]) -> DriftInfo:
    """Flag features whose value sits far from its training-set distribution."""
    if not FEATURE_STATS:
        return DriftInfo(checked=False)

    drifting = []
    for name, value in row.items():
        stats = FEATURE_STATS.get(name)
        if stats is None or value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        std = stats.get("std", 0.0)
        if not std:
            continue
        z = abs((value - stats["mean"]) / std)
        if z > DRIFT_Z_THRESHOLD:
            drifting.append(name)

    if drifting:
        logger.warning("Potential input drift on features: %s", drifting)

    return DriftInfo(checked=True, n_drifting_features=len(drifting), drifting_features=drifting)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")

    event = Event(
        event_id="request",
        process="unknown",
        weight=1.0,
        met=req.met,
        met_phi=req.met_phi,
        objects=[Obj(kind=o.kind, E=o.E, pt=o.pt, eta=o.eta, phi=o.phi) for o in req.objects],
    )

    row = event_features(event)
    X = pd.DataFrame([row])[FEATURES]

    score = float(model.predict_proba(X)[:, 1][0])
    label = int(score >= THRESHOLD)
    drift = _check_drift(row)

    logger.info("predict score=%.4f label=%d n_objects=%d", score, label, len(req.objects))

    return PredictResponse(
        score=score,
        label=label,
        prediction="signal" if label == 1 else "background",
        threshold=THRESHOLD,
        drift=drift,
    )
