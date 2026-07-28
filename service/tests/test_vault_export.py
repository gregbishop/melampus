"""Tests for the sightings-log export.

This output is read later as settled fact rather than as model output, so
its aggregation rules matter more than most. The two that must never regress:
counting per encounter rather than per frame, and refusing to record encounters
the model could not agree with itself about.

Encounter clustering reads xmp:CreateDate straight out of the file bytes and the
exporter never opens the pixels, so stub files carrying only an XMP packet are
sufficient and keep the suite fast.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPORTER = REPO / "tools" / "export_to_vault.py"


def stub_frame(folder: Path, name: str, when: str) -> None:
    """A file the clusterer can date. Pixels are never read by the exporter."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(
        b'\xff\xd8\xff\xe1<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        + f'<rdf:Description xmp:CreateDate="{when}"/>'.encode()
        + b"</x:xmpmeta>"
    )


def result(name: str, species: str, confidence: float = 0.95, abstain: bool = False) -> dict:
    ident = {
        "taxon": "bird", "candidates": [], "age_sex": "adult", "count": 1,
        "behavior": [], "diagnostic_features_visible": True,
        "abstain": abstain, "abstain_reason": "facing away" if abstain else None,
    }
    if not abstain:
        ident["candidates"] = [{
            "common_name": species, "scientific_name": "Egretta sp",
            "confidence": confidence, "reasoning": "",
        }]
    return {"file": name, "content_hash": name, "status": "ok",
            "identification": ident, "model": "test", "seconds": 1.0}


def run_export(tmp_path: Path, frames, results, extra=()) -> str:
    folder = tmp_path / "photos"
    for name, when in frames:
        stub_frame(folder, name, when)
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))
    out = tmp_path / "sightings.md"

    proc = subprocess.run(
        [sys.executable, str(EXPORTER), str(folder), str(results_path), str(out), *extra],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    return out.read_text()


# --------------------------------------------------------------------------- #
def test_a_burst_counts_as_one_sighting_not_many(tmp_path: Path):
    """The rule that keeps a life list honest."""
    frames = [(f"f{i}.jpg", f"2021-03-04T09:15:{i:02d}-05:00") for i in range(30)]
    results = [result(n, "Tricolored Heron") for n, _ in frames]

    text = run_export(tmp_path, frames, results)
    row = next(line for line in text.splitlines() if "Tricolored Heron" in line and "|" in line)
    cells = [c.strip() for c in row.split("|")]
    assert cells[4] == "1", f"30 frames of one bird counted as {cells[4]} sightings"
    assert cells[5] == "30", "frame count should still be reported"


def test_separate_encounters_count_separately(tmp_path: Path):
    """Two visits on different days are two sightings."""
    frames = [("a1.jpg", "2021-03-04T09:15:00-05:00"), ("a2.jpg", "2021-03-04T09:15:02-05:00"),
              ("b1.jpg", "2021-03-09T14:20:00-05:00"), ("b2.jpg", "2021-03-09T14:20:02-05:00")]
    results = [result(n, "Snowy Egret") for n, _ in frames]

    text = run_export(tmp_path, frames, results)
    row = next(line for line in text.splitlines() if "Snowy Egret" in line and "|" in line)
    cells = [c.strip() for c in row.split("|")]
    assert cells[4] == "2"
    assert cells[2] == "2021-03-04", "first-seen date wrong"
    assert cells[3] == "2021-03-09", "last-seen date wrong"


def test_unstable_encounter_is_dropped_not_guessed(tmp_path: Path):
    """If the model cannot agree with itself, record nothing."""
    frames = [(f"f{i}.jpg", f"2021-03-04T09:15:{i:02d}-05:00") for i in range(4)]
    results = [
        result("f0.jpg", "Tricolored Heron"), result("f1.jpg", "Little Blue Heron"),
        result("f2.jpg", "Green Heron"), result("f3.jpg", "Snowy Egret"),
    ]
    text = run_export(tmp_path, frames, results, extra=["--min-agreement", "0.6"])

    recorded = [s for s in ("Tricolored Heron", "Little Blue Heron", "Green Heron",
                            "Snowy Egret") if f"| {s} |" in text]
    assert recorded == [], f"guessed {recorded} from an encounter with no agreement"
    assert "dropped for frame-to-frame disagreement" in text


def test_abstained_encounter_produces_no_species(tmp_path: Path):
    frames = [("f0.jpg", "2021-03-04T09:15:00-05:00"), ("f1.jpg", "2021-03-04T09:15:01-05:00")]
    results = [result("f0.jpg", "", abstain=True), result("f1.jpg", "", abstain=True)]

    text = run_export(tmp_path, frames, results)
    assert "abstained entirely" in text


def test_provisional_warning_present_until_reviewed(tmp_path: Path):
    """The vault must not present unreviewed model output as fact."""
    frames = [("f0.jpg", "2021-03-04T09:15:00-05:00")]
    results = [result("f0.jpg", "Anhinga")]

    provisional = run_export(tmp_path, frames, results)
    assert "PROVISIONAL" in provisional
    assert "unreviewed model output" in provisional

    confirmed = run_export(tmp_path, frames, results, extra=["--reviewed"])
    assert "human-confirmed" in confirmed
    assert "PROVISIONAL" not in confirmed


def test_output_carries_vault_frontmatter(tmp_path: Path):
    frames = [("f0.jpg", "2021-03-04T09:15:00-05:00")]
    text = run_export(tmp_path, frames, [result("f0.jpg", "Anhinga")])

    assert text.startswith("---\n"), "missing YAML frontmatter the vault requires"
    for key in ("tier:", "status:", "reviewed:"):
        assert key in text.split("---")[1], f"frontmatter missing {key}"
    assert "[[melampus]]" in text, "should link back to the project note"
