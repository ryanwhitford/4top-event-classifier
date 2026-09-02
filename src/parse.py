"""
Parsing for the DarkMachines variable-length event format.

Each line of a DarkMachines CSV is one collision event, and lines have
*different lengths* because events contain different numbers of objects:

    event ID; process ID; event weight; MET; METphi; obj1,E1,pt1,eta1,phi1; obj2,E2,pt2,eta2,phi2; ...

Object type strings are: j (jet), b (b-jet), e- / e+ (electron/positron),
m- / m+ (muon/antimuon), g (photon).

This is not a rectangular table, so pandas.read_csv cannot load it directly.
The functions here stream the file line by line and produce a list of
`Event` records that `features.py` turns into a fixed-width design matrix.

Reference: "The Dark Machines Anomaly Score Challenge: Benchmark Data and
Model Independent Event Classification for the Large Hadron Collider",
SciPost Phys. 12, 043 (2022), Section 2.2.
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# Energies/momenta in the released files are in MeV. Everything downstream is
# nicer to read in GeV, so we scale once at parse time.
MEV_TO_GEV = 1.0e-3

OBJECT_TYPES = ("j", "b", "e-", "e+", "m-", "m+", "g")

# Lepton flavour and electric charge, keyed by the object type string.
LEPTON_TYPES = {"e-", "e+", "m-", "m+"}
CHARGE = {"e-": -1, "e+": +1, "m-": -1, "m+": +1}


@dataclass(frozen=True)
class Obj:
    """One reconstructed object (jet, b-jet, lepton or photon) in an event."""

    kind: str
    E: float
    pt: float
    eta: float
    phi: float

    @property
    def px(self) -> float:
        return self.pt * math.cos(self.phi)

    @property
    def py(self) -> float:
        return self.pt * math.sin(self.phi)

    @property
    def pz(self) -> float:
        return self.pt * math.sinh(self.eta)

    @property
    def p(self) -> float:
        """Magnitude of the three-momentum."""
        return self.pt * math.cosh(self.eta)

    @property
    def mass(self) -> float:
        """Invariant mass, clipped at zero to absorb float noise."""
        m2 = self.E * self.E - self.p * self.p
        return math.sqrt(m2) if m2 > 0.0 else 0.0

    @property
    def is_lepton(self) -> bool:
        return self.kind in LEPTON_TYPES

    @property
    def charge(self) -> int:
        return CHARGE.get(self.kind, 0)


@dataclass
class Event:
    event_id: str
    process: str
    weight: float
    met: float
    met_phi: float
    objects: list[Obj] = field(default_factory=list)

    def of_kind(self, *kinds: str) -> list[Obj]:
        """Objects of the given type(s), sorted by descending pT."""
        sel = [o for o in self.objects if o.kind in kinds]
        sel.sort(key=lambda o: o.pt, reverse=True)
        return sel

    @property
    def leptons(self) -> list[Obj]:
        return self.of_kind("e-", "e+", "m-", "m+")


class ParseError(ValueError):
    pass


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def parse_line(line: str, *, scale: float = MEV_TO_GEV) -> Event | None:
    """Parse a single event line. Returns None for blank/comment lines."""
    line = line.strip().rstrip(";")
    if not line or line.startswith("#"):
        return None

    fields = [f.strip() for f in line.split(";")]
    if len(fields) < 5:
        raise ParseError(f"expected >=5 semicolon fields, got {len(fields)}: {line[:120]!r}")

    event_id, process, weight, met, met_phi = fields[:5]

    objects: list[Obj] = []
    for group in fields[5:]:
        if not group:
            continue
        parts = [p.strip() for p in group.split(",")]
        if len(parts) != 5:
            raise ParseError(f"object group should have 5 fields, got {parts!r}")
        kind, E, pt, eta, phi = parts
        objects.append(
            Obj(
                kind=kind,
                E=float(E) * scale,
                pt=float(pt) * scale,
                eta=float(eta),
                phi=float(phi),
            )
        )

    return Event(
        event_id=event_id,
        process=process,
        weight=float(weight),
        met=float(met) * scale,
        met_phi=float(met_phi),
        objects=objects,
    )


def read_events(
    paths: Path | str | Sequence[Path | str],
    *,
    limit: int | None = None,
    scale: float = MEV_TO_GEV,
    strict: bool = False,
) -> Iterator[Event]:
    """Stream events from one or more DarkMachines CSV files.

    `limit` caps the number of events *per file*, which keeps the notebook
    responsive while you are still iterating on features.

    With `strict=False` (the default) malformed lines are skipped rather than
    raising, because the released files occasionally end mid-line.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    for raw in paths:
        path = Path(raw)
        n = 0
        with _open(path) as fh:
            for lineno, line in enumerate(fh, start=1):
                try:
                    ev = parse_line(line, scale=scale)
                except ParseError:
                    if strict:
                        raise ParseError(f"{path.name}:{lineno}") from None
                    continue
                if ev is None:
                    continue
                yield ev
                n += 1
                if limit is not None and n >= limit:
                    break


def find_data_files(root: Path | str, pattern: str = "*.csv") -> list[Path]:
    """Locate the extracted DarkMachines CSVs under `root`, recursively."""
    root = Path(root)
    files = sorted(p for p in root.rglob(pattern) if p.is_file())
    if not files:
        files = sorted(p for p in root.rglob(pattern + ".gz") if p.is_file())
    return files


def process_counts(events: Iterable[Event]) -> dict[str, int]:
    """Tally the `process ID` column. Useful for deciding on the label map."""
    counts: dict[str, int] = {}
    for ev in events:
        counts[ev.process] = counts.get(ev.process, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
