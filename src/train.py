"""
Train the Four-Top Event Classifier for deployment.

Reproduces the model-selection notebook's final pipeline end-to-end, from raw
DarkMachines CSVs to a saved model + metadata bundle:

    parse -> build_dataframe -> 70/15/15 stratified split
    -> RandomizedSearchCV over HistGradientBoostingClassifier (scoring: average precision)
    -> threshold selection on the validation set by maximum Asimov significance
    -> final evaluation on the held-out test set

This mirrors notebooks/4top_vs_ttbar.ipynb section-for-section (same RNG,
same split, same search space) so the deployed model matches the numbers
reported in the README. See that notebook for the exploratory analysis and
ablations that motivated these choices.

Usage:
    python -m src.train
    python -m src.train --data-dir data/raw --out-dir models --n-iter 30 --cv 3
    python -m src.train --limit 2000      # fast smoke test (subsamples per file)

Outputs (written to --out-dir, default `models/`):
    model.joblib     the fitted sklearn estimator (best_estimator_ from the search)
    metadata.json    feature order, threshold, metrics, and per-feature train
                      statistics (for the serving app's drift check)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from .features import build_dataframe, feature_columns
from .parse import find_data_files, process_counts, read_events

RNG = 42


# ---- physics-significance helpers (identical to the notebook) -------------

def asimov_z(s, b, eps: float = 1e-9):
    """Asimov discovery significance for expected yields s (signal), b (background)."""
    s, b = np.maximum(s, 0.0), np.maximum(b, eps)
    return np.sqrt(2.0 * ((s + b) * np.log1p(s / b) - s))


def yields(scores, y_true, weights, threshold, lumi: float = 1.0):
    """Weighted signal and background yields passing scores >= threshold."""
    sel = scores >= threshold
    s = weights[sel & (y_true == 1)].sum() * lumi
    b = weights[sel & (y_true == 0)].sum() * lumi
    return s, b


# ---- pipeline steps ---------------------------------------------------------

def load_dataframe(data_dir: Path, limit: int | None) -> tuple[pd.DataFrame, list[str]]:
    files = find_data_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No data files found under {data_dir}. See data/README.md for download instructions."
        )
    print(f"Found {len(files)} data file(s) under {data_dir}")

    events = list(read_events(files, limit=limit))
    print(f"Parsed {len(events):,} events")

    counts = process_counts(events)
    signal_processes = [p for p in counts if "4top" in p.lower() or "tttt" in p.lower()]
    if not signal_processes:
        raise RuntimeError(f"No 4-top process found. Processes present: {list(counts)}")
    print("Signal processes :", signal_processes)
    print("Background       :", [p for p in counts if p not in signal_processes])

    df = build_dataframe(events, signal_processes=signal_processes)
    features = feature_columns(df)
    print(f"Design matrix: {df.shape[0]:,} events x {len(features)} features")
    print(f"Signal fraction (raw counts): {df.label.mean():.3%}")
    return df, features, signal_processes, counts


def split_data(df: pd.DataFrame, features: list[str]):
    X, y, w = df[features], df["label"].values, df["weight"].values
    X_tr, X_tmp, y_tr, y_tmp, w_tr, w_tmp = train_test_split(
        X, y, w, test_size=0.30, stratify=y, random_state=RNG
    )
    X_val, X_te, y_val, y_te, w_val, w_te = train_test_split(
        X_tmp, y_tmp, w_tmp, test_size=0.50, stratify=y_tmp, random_state=RNG
    )
    for name, yy in [("train", y_tr), ("validation", y_val), ("test", y_te)]:
        print(f"{name:<11} {len(yy):>8,} events   signal {yy.mean():.3%}")
    return (X_tr, y_tr, w_tr), (X_val, y_val, w_val), (X_te, y_te, w_te)


def tune(X_tr, y_tr, features: list[str], n_iter: int, cv: int, n_jobs: int) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(
            early_stopping=True, validation_fraction=0.15, random_state=RNG
        ),
        param_distributions={
            "learning_rate": loguniform(0.02, 0.3),
            "max_leaf_nodes": randint(15, 96),
            "min_samples_leaf": randint(10, 200),
            "l2_regularization": loguniform(1e-3, 10.0),
            "max_iter": randint(200, 700),
        },
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(cv, shuffle=True, random_state=RNG),
        n_jobs=n_jobs,
        random_state=RNG,
        refit=True,
        verbose=1,
    ).fit(X_tr[features], y_tr)

    print(f"best CV average precision: {search.best_score_:.4f}")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k:<20} {v if isinstance(v, int) else f'{v:.4g}'}")
    return search


def choose_threshold(model, X_val, y_val, w_val, features: list[str], lumi: float):
    p_val = model.predict_proba(X_val[features])[:, 1]
    thresholds = np.linspace(0.01, 0.995, 300)
    z_val = np.array([
        asimov_z(*yields(p_val, y_val, w_val, t, lumi=lumi)) for t in thresholds
    ])
    best_t = float(thresholds[np.nanargmax(z_val)])
    best_z = float(np.nanmax(z_val))
    print(f"Optimal threshold (validation): {best_t:.3f}  (Z={best_z:.2f})")
    return best_t, best_z


def evaluate_test(model, X_te, y_te, w_te, features: list[str], threshold: float, lumi: float):
    p_te = model.predict_proba(X_te[features])[:, 1]
    s_te, b_te = yields(p_te, y_te, w_te, threshold, lumi=lumi)
    s_all = w_te[y_te == 1].sum() * lumi
    b_all = w_te[y_te == 0].sum() * lumi
    metrics = {
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "brier_score": float(brier_score_loss(y_te, p_te)),
        "signal_efficiency": float(s_te / s_all) if s_all > 0 else None,
        "background_rejection": float(1 - b_te / b_all) if b_all > 0 else None,
        "significance_no_selection": float(asimov_z(s_all, b_all)),
        "significance_at_threshold": float(asimov_z(s_te, b_te)),
        "n_test_events": int(len(y_te)),
        "test_signal_fraction": float(y_te.mean()),
    }
    print("TEST SET, at the threshold chosen on validation")
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    return metrics


def feature_stats(X_tr: pd.DataFrame, features: list[str]) -> dict:
    """Per-feature train-set mean/std/NaN-rate (NaN-aware), for the serving
    app's runtime drift check. A feature is 'drifting' at inference time when
    incoming values sit many train-set standard deviations from this mean."""
    stats = {}
    for col in features:
        vals = X_tr[col].to_numpy(dtype=float)
        finite = vals[~np.isnan(vals)]
        stats[col] = {
            "mean": float(np.mean(finite)) if finite.size else 0.0,
            "std": float(np.std(finite)) if finite.size else 0.0,
            "nan_rate": float(np.isnan(vals).mean()),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--n-iter", type=int, default=30, help="RandomizedSearchCV draws (notebook default: 30)")
    parser.add_argument("--cv", type=int, default=3, help="StratifiedKFold splits (notebook default: 3)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap events per input file. For smoke-testing the pipeline only -- omit for a real training run.",
    )
    args = parser.parse_args()

    t0 = time.time()
    df, features, signal_processes, process_counts_all = load_dataframe(args.data_dir, args.limit)

    y_full = df["label"].values
    w_full = df["weight"].values
    lumi = 1e5 / w_full[y_full == 0].sum()
    print(f"LUMI scale factor: {lumi:.6g}")

    (X_tr, y_tr, w_tr), (X_val, y_val, w_val), (X_te, y_te, w_te) = split_data(df, features)

    search = tune(X_tr, y_tr, features, n_iter=args.n_iter, cv=args.cv, n_jobs=args.n_jobs)
    model = search.best_estimator_

    best_t, best_z_val = choose_threshold(model, X_val, y_val, w_val, features, lumi)
    test_metrics = evaluate_test(model, X_te, y_te, w_te, features, best_t, lumi)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    model_path = args.out_dir / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RNG,
        "features": features,
        "n_features": len(features),
        "threshold": best_t,
        "threshold_significance_val": best_z_val,
        "lumi": lumi,
        "signal_processes": signal_processes,
        "process_counts": process_counts_all,
        "best_params": search.best_params_,
        "cv_best_average_precision": float(search.best_score_),
        "search": {"n_iter": args.n_iter, "cv": args.cv},
        "dataset": {
            "n_events_total": int(len(df)),
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_te)),
            "signal_fraction_total": float(df["label"].mean()),
        },
        "test_metrics": test_metrics,
        "feature_stats": feature_stats(X_tr, features),
    }

    metadata_path = args.out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    elapsed = time.time() - t0
    print(f"\nSaved {model_path} and {metadata_path}")
    print(f"Total training time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
