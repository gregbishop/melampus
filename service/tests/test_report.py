"""Scoring tests.

The first dev-set report claimed 45.5% overall accuracy because "Tricolored Heron"
and "Tri-colored Heron" compared as different species. These tests exist so that a
measurement bug cannot masquerade as a model result again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from melampus.report import _bucket, _matches, score, taxon_key
from melampus.schema import Candidate, Identification, ImageResult, Taxon


# --------------------------------------------------------------------------- #
# name normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Tricolored Heron", "Tri-colored Heron"),
        ("Black-crowned Night-Heron", "Black-crowned Night Heron"),
        ("  snowy   egret ", "Snowy Egret"),
        ("American alligator", "American Alligator"),
    ],
)
def test_hyphenation_and_spacing_variants_converge(left: str, right: str):
    assert taxon_key(left) == taxon_key(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Boat-tailed Grackle", "Great-tailed Grackle"),
        ("Little Blue Heron", "Tricolored Heron"),
        ("Snowy Egret", "Great Egret"),
    ],
)
def test_genuinely_different_species_stay_distinct(left: str, right: str):
    assert taxon_key(left) != taxon_key(right)


def test_scientific_name_rescues_an_unusual_common_name():
    """The taxon is right even when the vernacular name is phrased oddly."""
    assert _matches(
        taxon_key("Tri-colored Heron"), taxon_key("Egretta tricolor"),
        taxon_key("Tricolored Heron"), taxon_key("Egretta tricolor"),
    )
    assert _matches(
        taxon_key("Louisiana Heron"), taxon_key("Egretta tricolor"),
        taxon_key("Tricolored Heron"), taxon_key("Egretta tricolor"),
    )


def test_matching_scientific_name_is_not_enough_when_absent():
    assert not _matches(taxon_key("Little Blue Heron"), "", taxon_key("Tricolored Heron"), "")


def test_wrong_species_does_not_match_on_either_name():
    assert not _matches(
        taxon_key("Little Blue Heron"), taxon_key("Egretta caerulea"),
        taxon_key("Tricolored Heron"), taxon_key("Egretta tricolor"),
    )


def test_confidence_buckets():
    assert _bucket(0.95) == "0.90-1.00"
    assert _bucket(0.90) == "0.90-1.00"
    assert _bucket(0.85) == "0.80-0.89"
    assert _bucket(0.10) == "<0.50"


# --------------------------------------------------------------------------- #
# end-to-end scoring
# --------------------------------------------------------------------------- #
def _result(file: str, name: str, sci: str, conf: float = 0.9) -> ImageResult:
    return ImageResult(
        file=file, content_hash=file, status="ok",
        identification=Identification(
            taxon=Taxon.BIRD,
            candidates=[Candidate(common_name=name, scientific_name=sci, confidence=conf)],
        ),
    )


def _abstained(file: str) -> ImageResult:
    return ImageResult(
        file=file, content_hash=file, status="ok",
        identification=Identification(taxon=Taxon.BIRD, candidates=[], abstain=True,
                                      abstain_reason="facing away"),
    )


@pytest.fixture()
def labels(tmp_path: Path) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": [
        {"file": "a.jpg", "taxon": "bird", "common_name": "Tricolored Heron",
         "scientific_name": "Egretta tricolor", "expected_outcome": "identify"},
        {"file": "b.jpg", "taxon": "bird", "common_name": "Snowy Egret",
         "scientific_name": "Egretta thula", "expected_outcome": "identify"},
        {"file": "c.jpg", "taxon": "bird", "common_name": "Common Ground Dove",
         "scientific_name": "Columbina passerina", "expected_outcome": "abstain"},
    ]}))
    return path


def test_hyphenation_variant_scores_as_correct(labels: Path):
    s, _ = score([_result("a.jpg", "Tri-colored Heron", "Egretta tricolor")], labels)
    assert s.top1 == 1, "hyphenation variant scored as a miss"
    assert not s.confusions


def test_wrong_prediction_is_recorded_as_a_confusion(labels: Path):
    s, _ = score([_result("a.jpg", "Little Blue Heron", "Egretta caerulea")], labels)
    assert s.top1 == 0
    assert s.confusions[("tricolored heron", "little blue heron")] == 1


def test_abstention_is_not_counted_as_a_wrong_answer(labels: Path):
    """Abstaining is desirable behaviour; it must not be folded into accuracy."""
    s, _ = score([_abstained("c.jpg")], labels)
    assert s.abstain_expected_met == 1
    assert s.n_scored == 0
    assert s.confusions == {}


def test_committed_accuracy_excludes_abstentions(labels: Path):
    s, _ = score(
        [
            _result("a.jpg", "Tricolored Heron", "Egretta tricolor"),
            _abstained("b.jpg"),
        ],
        labels,
    )
    # Two species rows, one answered correctly and one abstained.
    assert s.n_scored == 2
    assert s.top1 == 1
    assert s.n_committed == 1
    assert s.top1_committed == 1


def test_calibration_buckets_are_populated(labels: Path):
    s, _ = score(
        [
            _result("a.jpg", "Tricolored Heron", "Egretta tricolor", conf=0.95),
            _result("b.jpg", "Little Blue Heron", "Egretta caerulea", conf=0.75),
        ],
        labels,
    )
    assert s.calibration["0.90-1.00"] == [1, 1]
    assert s.calibration["0.70-0.79"] == [0, 1]


def test_macro_average_is_not_dominated_by_the_common_species(labels: Path):
    """The whole point of macro-averaging on an imbalanced corpus."""
    results = [_result("a.jpg", "Tricolored Heron", "Egretta tricolor")]
    results += [_result("b.jpg", "Great Egret", "Ardea alba")]
    s, rendered = score(results, labels)
    # One species perfect, one species zero -> macro 50%, overall also 50% here,
    # but per-species must show both rather than a single blended figure.
    assert s.per_species["tricolored heron"] == [1, 1]
    assert s.per_species["snowy egret"] == [0, 1]
    assert "PER-SPECIES" in rendered
