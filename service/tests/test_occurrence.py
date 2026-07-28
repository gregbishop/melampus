"""Occurrence re-ranking tests (CLAUDE.md §4.3). No network required."""

from __future__ import annotations

from pathlib import Path

import pytest

from melampus.occurrence import (
    GBIFClient, Location, OccurrenceCache, rerank,
)
from melampus.schema import Candidate


class FakeClient(GBIFClient):
    """Canned occurrence counts. None means 'lookup failed', which is not zero."""

    def __init__(self, counts: dict[str, int | None]) -> None:
        super().__init__(cache=None)
        self.counts = counts
        self.calls: list[str] = []

    def count(self, scientific_name, location, month):
        self.calls.append(scientific_name)
        return self.counts.get(scientific_name)


FLORIDA = Location(26.45, -82.11, radius_km=50)


def c(common, sci, conf):
    return Candidate(common_name=common, scientific_name=sci, confidence=conf)


def test_out_of_range_top_pick_is_demoted_not_deleted():
    """The headline behaviour: an Asian cuckoo must lose to a Florida grackle."""
    cands = [c("Long-tailed Cuckoo", "Cuculus micropterus", 0.90),
             c("Boat-tailed Grackle", "Quiscalus major", 0.30)]
    outcome = rerank(cands, FLORIDA, 6,
                     FakeClient({"Cuculus micropterus": 0, "Quiscalus major": 2752}))

    assert outcome.applied
    assert cands[0].common_name == "Boat-tailed Grackle"
    assert len(cands) == 2, "an out-of-range candidate was deleted rather than demoted"
    assert any(c_.common_name == "Long-tailed Cuckoo" for c_ in cands)


def test_out_of_range_survivor_still_flags_for_review():
    """If everything is out of range the top pick stays flagged, not silently kept."""
    cands = [c("Great Bowerbird", "Chlamydera nuchalis", 0.8)]
    outcome = rerank(cands, FLORIDA, 6, FakeClient({"Chlamydera nuchalis": 0}))

    assert outcome.applied
    assert outcome.range_flag is True


def test_scarce_species_is_marked_notable_not_absent():
    """Neotropic Cormorant in Florida: rare but real. Demote gently, flag it."""
    cands = [c("Neotropic Cormorant", "Nannopterum brasilianum", 0.9),
             c("Double-crested Cormorant", "Nannopterum auritum", 0.7)]
    outcome = rerank(cands, FLORIDA, 6,
                     FakeClient({"Nannopterum brasilianum": 5, "Nannopterum auritum": 2284}))

    verdict = outcome.verdicts["Nannopterum brasilianum"]
    assert verdict.in_range is True
    assert verdict.notable is True
    assert cands[0].common_name == "Double-crested Cormorant"
    assert outcome.range_flag is False, "the promoted common species is not itself notable"


def test_two_common_species_are_left_alone():
    """Tricolored vs Little Blue are both abundant; range must not invent a winner."""
    cands = [c("Tricolored Heron", "Egretta tricolor", 0.55),
             c("Little Blue Heron", "Egretta caerulea", 0.45)]
    outcome = rerank(cands, FLORIDA, 6,
                     FakeClient({"Egretta tricolor": 3060, "Egretta caerulea": 2954}))

    assert outcome.applied
    assert cands[0].common_name == "Tricolored Heron", "ordering changed with no evidence"
    assert outcome.range_flag is False


# --------------------------------------------------------------------------- #
# graceful degradation — §4.3 item 5
# --------------------------------------------------------------------------- #
def test_no_location_skips_reranking_and_says_so():
    cands = [c("Tricolored Heron", "Egretta tricolor", 0.9)]
    outcome = rerank(cands, None, 6, FakeClient({}))

    assert outcome.applied is False
    assert "location" in outcome.reason
    assert cands[0].common_name == "Tricolored Heron"


def test_failed_lookup_is_not_treated_as_absence():
    """None must never be read as zero, or an outage would demote real species."""
    cands = [c("Tricolored Heron", "Egretta tricolor", 0.9),
             c("Little Blue Heron", "Egretta caerulea", 0.1)]
    outcome = rerank(cands, FLORIDA, 6,
                     FakeClient({"Egretta tricolor": None, "Egretta caerulea": None}))

    assert outcome.applied is False
    assert cands[0].common_name == "Tricolored Heron", "an outage reordered candidates"


def test_partial_outage_still_ranks_on_what_is_known():
    cands = [c("Long-tailed Cuckoo", "Cuculus micropterus", 0.9),
             c("Boat-tailed Grackle", "Quiscalus major", 0.4)]
    outcome = rerank(cands, FLORIDA, 6,
                     FakeClient({"Cuculus micropterus": 0, "Quiscalus major": None}))

    assert outcome.applied
    assert cands[0].common_name == "Boat-tailed Grackle"


def test_candidate_without_scientific_name_is_kept_and_noted():
    cands = [c("Some Bird", "", 0.9)]
    outcome = rerank(cands, FLORIDA, 6, FakeClient({}))

    assert outcome.applied is False
    assert len(cands) == 1
    assert any("no scientific name" in n for n in outcome.notes)


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #
def test_cache_key_groups_coordinates_in_the_same_cell():
    """Shots within a rounding cell share one lookup.

    Grid boundaries are inherent: two points 200 m apart that straddle a boundary
    land in different cells and cost an extra API call. That is benign — it can
    only cause a redundant lookup, never a wrong verdict — so it is not worth a
    more elaborate scheme.
    """
    a = OccurrenceCache.key("Egretta tricolor", Location(26.462, -82.113), 6)  # -> 26.5, -82.1
    b = OccurrenceCache.key("Egretta tricolor", Location(26.539, -82.087), 6)  # -> 26.5, -82.1
    assert a == b, "nearby shoots in one cell should share a cache entry"


def test_cache_key_separates_months_and_distant_places():
    june = OccurrenceCache.key("Egretta tricolor", Location(26.45, -82.11), 6)
    december = OccurrenceCache.key("Egretta tricolor", Location(26.45, -82.11), 12)
    elsewhere = OccurrenceCache.key("Egretta tricolor", Location(40.0, -74.0), 6)
    assert june != december
    assert june != elsewhere


def test_cache_round_trips_to_disk(tmp_path: Path):
    path = tmp_path / "occ.json"
    cache = OccurrenceCache(path)
    key = OccurrenceCache.key("Egretta tricolor", Location(26.45, -82.11), 6)
    cache.put(key, 3060)
    cache.flush()

    assert OccurrenceCache(path).get(key) == 3060


# --------------------------------------------------------------------------- #
# non-organism subjects
# --------------------------------------------------------------------------- #
def test_occurrence_does_not_apply_to_sports():
    """A GBIF lookup for 'CrossFit' returns zero and would flag it out of range."""
    from melampus.occurrence import applies_to

    for taxon in ("football", "fitness", "court_sport", "people", "none"):
        assert not applies_to(taxon), f"{taxon} should be exempt from range checks"
    for taxon in ("bird", "reptile", "plant", "insect"):
        assert applies_to(taxon)


def test_reranking_skips_non_organism_taxa():
    cands = [c("American Football", "", 0.95)]
    outcome = rerank(cands, FLORIDA, 6, FakeClient({}), taxon="football")

    assert outcome.applied is False
    assert "does not apply" in outcome.reason
    assert outcome.range_flag is False, "a sport must never be flagged out of range"
