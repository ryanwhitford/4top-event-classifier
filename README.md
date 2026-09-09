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
tests/             unit tests for parse.py and features.py
notebooks/         the analysis
data/README.md     how to obtain the dataset
data/raw/          the downloaded dataset (not committed, see .gitignore)
```

## Caveats

The data is simulated with a parameterised detector simulation, so a classifier trained on it isn't directly a classifier for real collision data. The luminosity normalisation is chosen for readability, which makes absolute significance values non-physical — relative comparisons between selections are the meaningful output. The Asimov significance ignores systematic uncertainties, which in a real four-top analysis are substantial and often dominant.

## Data and references

DarkMachines Community, *The 4tops dataset*, Zenodo, [10.5281/zenodo.7277951](https://doi.org/10.5281/zenodo.7277951) (CC-BY-4.0).

Format documented in T. Aarrestad et al., *The Dark Machines Anomaly Score Challenge: Benchmark Data and Model Independent Event Classification for the Large Hadron Collider*, [SciPost Phys. 12, 043 (2022)](https://scipost.org/10.21468/SciPostPhys.12.1.043).
