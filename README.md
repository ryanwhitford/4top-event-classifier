# Finding Four-Top-Quark Events at the LHC

A tabular classification problem from particle physics, worked end-to-end: variable-length raw input, physics-informed feature engineering, tree ensembles, and evaluation.

**[→ Read the notebook](notebooks/4top_vs_ttbar.ipynb)**

---

## The task

Given a simulated LHC collision event, was it **four top quarks** (`pp → tttt`) or one of four **`ttbar`+boson** processes that mimic it (`ttbarW`, `ttbarZ`, `ttbarHiggs`, `ttbarWW`)? Four-top production is roughly 1000x rarer than *inclusive* top-pair production, which is the whole reason it's interesting — several theories beyond the Standard Model would first show up as an excess of these events.

A real analysis kills that huge inclusive-`ttbar` rate cheaply with a same-sign/multi-lepton preselection before doing anything else. The dataset here already has that cut applied (every event has 2–3 leptons, all same-sign), so the classifier's actual job isn't "find four-top in a sea of ordinary top pairs" — it's separating four-top from the rarer, comparably-sized `ttbar`+boson processes that survive the same cut. That's a harder and more realistic problem, and it's the one this notebook solves.

## Why it's a good ML exercise

- **The input isn't tabular.** Every event has a different number of reconstructed objects — jets, leptons, photons. One event has 3, another has 14. Building a fixed-width feature matrix from a variable-length object list, without throwing away what matters, is the real modelling decision, made before any model is fit.
- **Two competing imbalances.** Raw event counts are close to 50/50 (oversampled on purpose). But every event also carries a physical weight, and by *weighted* yield the sample skews background-heavy the other way. We train on one, evaluate on the other.
- **A textbook feature turns out to be a dead end** — and finding that out empirically, rather than trusting the domain-knowledge narrative, is itself part of the exercise (see Results).
- **Feature engineering beats model choice.** The same models, run on raw leading-object kinematics vs. a physics-informed feature set, produce a ~29% relative PR-AUC gap.

## Approach

Tree ensembles only — decision tree → random forest → `HistGradientBoostingClassifier` — because they handle this dataset's structurally missing values natively (an event with two b-jets just has no third one) and remain the standard for tabular physics data. We will run this pipeline rigorously: appropriate metrics, train/val/test split, a threshold chosen on the objective, and interpretability that traces back to physical features.

### Feature engineering

Every top quark decays `t → Wb`, so four tops give four b-quarks and four W bosons, against two of each for the background. That predicts three things:

| Family | Features | Predicted signal |
|---|---|---|
| **Multiplicity** | jet / b-jet / lepton counts, b-fraction | More partons per event (up to 12 vs. 6) — the strongest family in practice |
| **Energy scale** | `HT`, `ST`, `m_eff`, invariant mass, centrality | ~4× the rest mass entering the event |
| **Event shape** | sphericity, aplanarity, ΔR statistics, η spread | More isotropic than `ttbar`'s back-to-back topology |
| **Lepton charge** | charge sum, same-sign-pair flag | Discriminating on an inclusive sample — constant and nearly worthless here, since the dataset is already same-sign-preselected (see Results) |

Plus explicit kinematics (`pT`, `η`, `E`, mass) for the leading jets, b-jets, and leptons — 70 features total.

### Pipeline

1. Stream-parse the variable-length event format
2. Exploratory data analysis (EDA)
3. Baseline: raw leading-object kinematics only
4. Decision tree → random forest → gradient boosting
5. Randomised hyperparameter search, scored on average precision
6. Threshold chosen by maximising expected significance on validation, applied once to test
7. Permutation importance + optional SHAP
8. Ablation: retrain with each feature family removed

## Results

Run end-to-end on the real dataset (302,072 events: 152,000 signal / 150,072 background). Metrics below are on the **validation** split unless noted.

| Model | Features | Accuracy | ROC AUC | PR AUC |
|---|---|---|---|---|
| Decision tree, raw kinematics | 11 | 0.629 | 0.661 | 0.628 |
| Random forest, raw kinematics | 11 | 0.632 | 0.667 | 0.641 |
| Decision tree, physics features | 70 | 0.733 | 0.810 | 0.808 |
| Random forest, physics features | 70 | 0.751 | 0.829 | 0.833 |
| Gradient boosting, physics features | 70 | 0.756 | 0.836 | 0.840 |
| Gradient boosting, tuned | 70 | 0.758 | 0.837 | 0.841 |

**Test set** (touched once): ROC AUC 0.836, PR AUC 0.843, Brier score 0.165.

**Threshold chosen by maximising expected significance on validation** (0.843), applied once to test:

|  | No selection | Selected |
|---|---:|---:|
| Signal yield | 2,003.7 | 793.6 |
| Background yield | 15,196.7 | 508.1 |
| S/B | 0.132 | 1.562 |
| Significance Z | 15.9 | 29.4 |

Signal efficiency 39.6%, background rejection 96.7%, **significance gain 1.8×**

**Permutation importance:** multiplicity features dominate (`n_bjets` and `n_jets_all` are the two most important columns overall), followed by leading-object kinematics and energy scale. Event shape and lepton charge are marginal.

**Ablation** confirms it: removing the lepton-charge family changes test PR AUC by +0.0003 — noise, meaning it carries no information on this preselected sample. Removing multiplicity costs −0.0067, the largest drop of any family, and multiplicity alone (7 features) already reaches PR AUC 0.781.

## Getting the data

Not committed here — it belongs to its authors. See [`data/README.md`](data/README.md) for the exact files and license; in short, download the five per-process CSVs from **[10.5281/zenodo.7277951](https://doi.org/10.5281/zenodo.7277951)** into `data/raw/`.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/4top_vs_ttbar.ipynb
```

Run the test suite (hand-built fixtures, no data download required):

```bash
pip install -r requirements-dev.txt
pytest
```

### Docker

```bash
docker build -t 4top-classifier .
docker run -p 8888:8888 -v "$(pwd)/data/raw:/app/data/raw" 4top-classifier
```

Open the URL printed in the logs (includes an access token). Run the tests instead with `docker run 4top-classifier pytest`.

## Repository layout

```
src/parse.py       variable-length event format -> Event records
src/features.py    Event records -> 70-column physics feature matrix
src/train.py       trains + saves models/model.joblib and metadata.json
src/serve.py       FastAPI app serving the trained model
tests/             unit tests for parse.py, features.py, and serve.py
notebooks/         the analysis
models/            trained model artifact + metadata (committed, ~2.5 MB)
infra/             Bicep template + one-time Azure setup script
data/README.md     how to obtain the dataset
data/raw/          the downloaded dataset (not committed, see .gitignore)
```

## Deployment & MLOps

The classifier is also deployed as a small HTTP service, with CI/CD to Azure. This section covers what exists and, more importantly, *why* -- the design choices a from-scratch MLOps setup for a single-model portfolio project actually calls for, versus what it doesn't.

**Pipeline:** `src/train.py` reproduces the notebook's final pipeline end-to-end (same RNG, same 70/15/15 split, same `RandomizedSearchCV` search space, same Asimov-significance threshold selection) and writes `models/model.joblib` + `models/metadata.json` (feature order, threshold, test metrics, and per-feature training statistics). `src/serve.py` is a FastAPI app that loads that artifact and exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + whether the model loaded |
| `GET /model-info` | metadata: features, threshold, training metrics |
| `POST /predict` | score one event (raw objects + MET in, signal/background label out) |

`/predict` accepts the same raw per-object event representation `src/parse.py` works with, and calls the *same* `event_features()` function `train.py` used to build its training matrix -- there is exactly one place feature engineering happens, so training and serving cannot silently drift apart.

**CI/CD:** `.github/workflows/tests.yml` (existing) runs `pytest` on every push/PR. `.github/workflows/deploy.yml` builds `Dockerfile.serve` and rolls it out to an Azure Container App whenever the serving code or the committed model artifact changes on `main`, authenticating via OpenID Connect federated credentials -- no Azure client secret is stored in GitHub at all. `infra/main.bicep` (validated with `bicep build`, not yet applied to a live subscription) provisions Log Analytics, Application Insights, an Azure Container Registry, and a Container Apps environment/app; `infra/setup.sh` is the one-time bootstrap that creates the resource group, deploys the Bicep template, and registers the federated credential.

**Design decisions, and why:**

- *Tuned model, not a fixed baseline.* `train.py` replicates the notebook's `RandomizedSearchCV` rather than shipping a simpler fixed-hyperparameter model, so the deployed model's numbers match this README's. Its CLI defaults to the notebook's own budget (30 draws, 3-fold CV); the model currently committed was trained with a reduced budget (15 draws, 3-fold CV) to fit the compute available at deploy time -- see `models/metadata.json`'s `search` field for exactly what produced the committed artifact, and rerun with the defaults for the full search on a machine with more headroom.
- *A committed artifact, not a model registry.* `model.joblib` is ~2.5 MB, so it's committed directly to git rather than pushed to a registry (MLflow, Azure ML's model registry, etc.). A registry earns its complexity with multiple models, rollback-by-reference, or a team sharing artifacts; a single-model portfolio project doesn't have that problem yet, and git history already gives every past version and a trivial rollback (`git revert`).
- *Manual retraining, not a scheduled job.* The training data is a static, fixed simulation snapshot, not a live stream -- there's nothing for a nightly retrain to pick up. Retraining is a deliberate act: rerun `train.py`, review the metrics, commit the new artifact. That commit is what triggers `deploy.yml`.
- *Container Apps consumption plan, not a managed online endpoint.* Azure ML's managed online endpoints bill hourly for a provisioned VM whether or not it's serving traffic. Container Apps on the consumption plan scales to zero between requests and only costs money while actually handling one -- the right tradeoff for a project funded by a fixed student credit rather than a production SLA.
- *A per-request drift tripwire, not a population drift monitor.* `/predict` compares each incoming feature to its training-set mean/std and logs a warning (visible in Application Insights) when a feature lands more than a few standard deviations out. That's a real limitation, not a full solution: genuine drift detection needs a *distribution* of requests over time, not one at a time -- this just catches individual wildly-out-of-range inputs and gives a hook to build a real one on top of.

## Caveats

The data is simulated with a parameterised detector simulation, so a classifier trained on it isn't directly a classifier for real collision data. The luminosity normalisation is chosen for readability, which makes absolute significance values non-physical — relative comparisons between selections are the meaningful output. The Asimov significance ignores systematic uncertainties, which in a real four-top analysis are substantial and often dominant.

## Data and references

DarkMachines Community, *The 4tops dataset*, Zenodo, [10.5281/zenodo.7277951](https://doi.org/10.5281/zenodo.7277951) (CC-BY-4.0).

Format documented in T. Aarrestad et al., *The Dark Machines Anomaly Score Challenge: Benchmark Data and Model Independent Event Classification for the Large Hadron Collider*, [SciPost Phys. 12, 043 (2022)](https://scipost.org/10.21468/SciPostPhys.12.1.043).
