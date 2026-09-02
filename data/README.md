# Getting the data

This project uses the DarkMachines four-top-quark dataset: simulated LHC
collision events for `pp -> tttt` (signal) and four `ttbar`-associated
background processes.

## Download

The dataset is distributed as five per-process CSV files on Zenodo, not a
single archive:

- `4top.csv` — signal, `pp -> tttt`
- `ttbarHiggs.csv` — background, `pp -> ttbar H`
- `ttbarW.csv` — background, `pp -> ttbar W`
- `ttbarWW.csv` — background, `pp -> ttbar WW`
- `ttbarZ.csv` — background, `pp -> ttbar Z`

Download all five from **[10.5281/zenodo.7277951](https://doi.org/10.5281/zenodo.7277951)**
and place them, unmodified, in `data/raw/`:

```bash
mkdir -p data/raw
# download each file from the Zenodo record into data/raw/
```

`src/parse.find_data_files` locates every `*.csv` under `data/raw/`
recursively, so subdirectories are fine if you prefer to keep them separate.

None of these files are committed to the repository (see `.gitignore`) —
they total roughly 100 MB, and the dataset belongs to its authors.

## Format

Each line is one event; lines have different lengths because events contain
different numbers of reconstructed objects:

```
event ID; process; event weight; MET; METphi; obj1; obj2; ...
```

where each object is `type,E,pT,eta,phi`. Object types are `j` (jet), `b`
(b-tagged jet), `e-`/`e+` (electron/positron), `m-`/`m+` (muon/antimuon), and
`g` (photon). Energies and momenta are given in MeV; `src/parse.py` converts
them to GeV on read.

The event weight is the process's cross-section normalisation
(`sigma / N_generated`), constant within a file. It is what makes the raw
per-file event counts and the physically meaningful yields two different
numbers — see the "two kinds of imbalance" discussion in the notebook.

Full format reference: T. Aarrestad et al., *The Dark Machines Anomaly Score
Challenge: Benchmark Data and Model Independent Event Classification for the
Large Hadron Collider*, [SciPost Phys. 12, 043 (2022)](https://scipost.org/10.21468/SciPostPhys.12.1.043),
Section 2.2.

## License

DarkMachines Community, *The 4tops dataset*, Zenodo,
[10.5281/zenodo.7277951](https://doi.org/10.5281/zenodo.7277951) — CC-BY-4.0.
