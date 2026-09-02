"""
Physics-motivated feature engineering for 4-top vs. background classification.

The raw data is a variable-length list of objects per event. Tree ensembles
need a fixed-width matrix, so the question is *what to summarize*. Feeding in
"leading jet px, py, pz" — the naive choice — throws away most of the signal,
because the discriminating information in a 4-top event is not in any single
object. It is in the event as a whole.

The signature of pp -> tttt is:
  * high jet multiplicity (4 tops -> up to 12 partons in the fully hadronic mode)
  * high b-jet multiplicity (each top decays t -> Wb, so 4 real b-jets)
  * very large total hadronic activity (HT), since each top carries ~173 GeV of mass
  * a same-sign lepton pair, which the dominant ttbar background produces only
    through rare charge misidentification. This is the classic 4-top discovery
    channel precisely because it is so background-poor.

Every feature below is chosen for one of those reasons, and grouped so that
the SHAP plots in the notebook can be read as physics rather than as column
indices.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .parse import Event, Obj

# How many leading objects of each type to expand into explicit columns.
N_JETS = 4
N_BJETS = 3
N_LEPTONS = 2


def _dphi(a: float, b: float) -> float:
    """Signed azimuthal separation wrapped into (-pi, pi]."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return d


def _dr(o1: Obj, o2: Obj) -> float:
    return math.hypot(o1.eta - o2.eta, _dphi(o1.phi, o2.phi))


def _inv_mass(objs: Sequence[Obj]) -> float:
    """Invariant mass of a system of objects."""
    if not objs:
        return 0.0
    E = sum(o.E for o in objs)
    px = sum(o.px for o in objs)
    py = sum(o.py for o in objs)
    pz = sum(o.pz for o in objs)
    m2 = E * E - (px * px + py * py + pz * pz)
    return math.sqrt(m2) if m2 > 0.0 else 0.0


def _transverse_mass(pt: float, phi: float, met: float, met_phi: float) -> float:
    """mT of an object plus the missing transverse momentum."""
    m2 = 2.0 * pt * met * (1.0 - math.cos(_dphi(phi, met_phi)))
    return math.sqrt(m2) if m2 > 0.0 else 0.0


def _event_shape(objs: Sequence[Obj]) -> tuple[float, float]:
    """Sphericity and aplanarity from the normalised momentum tensor.

    Multi-top events are more isotropic than the back-to-back ttbar topology,
    so these are cheap, powerful, and completely independent of the counting
    variables above.
    """
    if len(objs) < 2:
        return 0.0, 0.0

    p = np.array([[o.px, o.py, o.pz] for o in objs], dtype=float)
    norm = float(np.sum(p * p))
    if norm <= 0.0:
        return 0.0, 0.0

    tensor = (p.T @ p) / norm
    eig = np.linalg.eigvalsh(tensor)          # ascending
    l3, l2, l1 = float(eig[0]), float(eig[1]), float(eig[2])

    sphericity = 1.5 * (l2 + l3)
    aplanarity = 1.5 * l3
    return sphericity, aplanarity


def event_features(ev: Event) -> dict[str, float]:
    """Turn one `Event` into a flat dict of named features."""
    jets_all = ev.of_kind("j", "b")           # every hadronic object, pT-ordered
    light_jets = ev.of_kind("j")
    bjets = ev.of_kind("b")
    leptons = ev.leptons
    photons = ev.of_kind("g")

    f: dict[str, float] = {}

    # --- Missing transverse energy ------------------------------------------
    # 4-top events contain up to four neutrinos from leptonic W decays.
    f["met"] = ev.met
    f["met_phi"] = ev.met_phi

    # --- Multiplicities -----------------------------------------------------
    # The single most robust 4-top discriminator: you simply get more stuff.
    f["n_jets_all"] = len(jets_all)
    f["n_jets_light"] = len(light_jets)
    f["n_bjets"] = len(bjets)
    f["n_leptons"] = len(leptons)
    f["n_photons"] = len(photons)
    f["n_objects"] = len(ev.objects)
    f["b_fraction"] = len(bjets) / len(jets_all) if jets_all else 0.0

    # --- Energy scale -------------------------------------------------------
    # Four top quarks carry ~4 x 173 GeV of rest mass, so the total hadronic
    # activity of the event is much larger than for ttbar.
    ht = sum(o.pt for o in jets_all)
    ht_b = sum(o.pt for o in bjets)
    lep_pt = sum(o.pt for o in leptons)
    f["ht"] = ht
    f["ht_b"] = ht_b
    f["ht_b_frac"] = ht_b / ht if ht > 0 else 0.0
    f["lepton_pt_sum"] = lep_pt
    f["st"] = ht + lep_pt + ev.met                 # total transverse activity
    f["m_eff"] = ht + ev.met                       # standard SUSY-style variable
    f["total_E"] = sum(o.E for o in ev.objects)
    f["m_all"] = _inv_mass(ev.objects)             # visible invariant mass
    f["centrality"] = (
        sum(o.pt for o in ev.objects) / sum(o.E for o in ev.objects)
        if ev.objects and sum(o.E for o in ev.objects) > 0
        else 0.0
    )

    # --- Lepton charge correlation -----------------------------------------
    # The headline 4-top channel on an inclusive sample: ttbar gives
    # opposite-sign lepton pairs, so a same-sign pair is rare background but
    # common in 4-top. Note that on a sample already preselected to the
    # same-sign/multi-lepton region (as the released DarkMachines files are —
    # see notebook section 2.2), `has_same_sign_pair` is constant by
    # construction and carries no information; it is kept here so this
    # function still does the right thing on an inclusive sample.
    charges = [o.charge for o in leptons]
    f["lepton_charge_sum"] = float(sum(charges))
    f["abs_lepton_charge_sum"] = float(abs(sum(charges)))
    f["has_same_sign_pair"] = float(
        any(
            charges[i] == charges[j]
            for i in range(len(charges))
            for j in range(i + 1, len(charges))
        )
    )

    # --- Leading-object kinematics -----------------------------------------
    # Explicit columns for the hardest objects. Missing slots are NaN, which
    # tree ensembles handle natively (and which *means* "this event has fewer
    # than k objects" rather than "the value is unknown").
    def fill(prefix: str, objs: Sequence[Obj], k: int) -> None:
        for i in range(k):
            o = objs[i] if i < len(objs) else None
            f[f"{prefix}{i + 1}_pt"] = o.pt if o else np.nan
            f[f"{prefix}{i + 1}_eta"] = o.eta if o else np.nan
            f[f"{prefix}{i + 1}_E"] = o.E if o else np.nan
            f[f"{prefix}{i + 1}_mass"] = o.mass if o else np.nan

    fill("jet", jets_all, N_JETS)
    fill("bjet", bjets, N_BJETS)
    fill("lep", leptons, N_LEPTONS)

    # --- Pairwise angular structure ----------------------------------------
    # How the event is laid out, not just how big it is.
    f["m_bb"] = _inv_mass(bjets[:2]) if len(bjets) >= 2 else np.nan
    f["dr_bb"] = _dr(bjets[0], bjets[1]) if len(bjets) >= 2 else np.nan
    f["m_jj"] = _inv_mass(light_jets[:2]) if len(light_jets) >= 2 else np.nan

    if len(jets_all) >= 2:
        drs = [
            _dr(jets_all[i], jets_all[j])
            for i in range(len(jets_all))
            for j in range(i + 1, len(jets_all))
        ]
        f["min_dr_jj"] = min(drs)
        f["mean_dr_jj"] = float(np.mean(drs))
        f["max_dr_jj"] = max(drs)
    else:
        f["min_dr_jj"] = np.nan
        f["mean_dr_jj"] = np.nan
        f["max_dr_jj"] = np.nan

    if leptons:
        f["mt_lep1_met"] = _transverse_mass(
            leptons[0].pt, leptons[0].phi, ev.met, ev.met_phi
        )
        f["dphi_lep1_met"] = abs(_dphi(leptons[0].phi, ev.met_phi))
    else:
        f["mt_lep1_met"] = np.nan
        f["dphi_lep1_met"] = np.nan

    f["dphi_jet1_met"] = (
        abs(_dphi(jets_all[0].phi, ev.met_phi)) if jets_all else np.nan
    )

    # --- Global event shape -------------------------------------------------
    sph, apl = _event_shape(ev.objects)
    f["sphericity"] = sph
    f["aplanarity"] = apl

    # --- Pseudorapidity spread ---------------------------------------------
    if len(jets_all) >= 2:
        etas = [o.eta for o in jets_all]
        f["jet_eta_std"] = float(np.std(etas))
        f["jet_eta_span"] = float(max(etas) - min(etas))
    else:
        f["jet_eta_std"] = np.nan
        f["jet_eta_span"] = np.nan

    return f


# Feature groups, used by the notebook to colour SHAP plots by physics meaning.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "multiplicity": (
        "n_jets_all", "n_jets_light", "n_bjets", "n_leptons",
        "n_photons", "n_objects", "b_fraction",
    ),
    "energy scale": (
        "ht", "ht_b", "ht_b_frac", "lepton_pt_sum", "st", "m_eff",
        "total_E", "m_all", "centrality", "met",
    ),
    "lepton charge": (
        "lepton_charge_sum", "abs_lepton_charge_sum", "has_same_sign_pair",
    ),
    "event shape": (
        "sphericity", "aplanarity", "jet_eta_std", "jet_eta_span",
        "min_dr_jj", "mean_dr_jj", "max_dr_jj", "dr_bb",
    ),
}


def build_dataframe(
    events: Iterable[Event],
    *,
    signal_processes: Sequence[str] = ("4top",),
) -> pd.DataFrame:
    """Build the modelling table: features + `label`, `weight`, `process`.

    `label` is 1 for any process listed in `signal_processes`, else 0.
    """
    sig = {s.lower() for s in signal_processes}

    rows: list[dict[str, float]] = []
    meta: list[tuple[str, float, int]] = []

    for ev in events:
        rows.append(event_features(ev))
        meta.append((ev.process, ev.weight, int(ev.process.lower() in sig)))

    df = pd.DataFrame(rows)
    df["process"] = [m[0] for m in meta]
    df["weight"] = [m[1] for m in meta]
    df["label"] = [m[2] for m in meta]
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model input columns: everything except metadata."""
    return [c for c in df.columns if c not in ("process", "weight", "label")]
