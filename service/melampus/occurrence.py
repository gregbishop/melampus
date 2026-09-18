"""Occurrence lookups and candidate re-ranking (CLAUDE.md §4.3).

The measured failure mode this exists to fix: the model offers species that do not
occur anywhere near where the photograph was taken. Observed on a Florida corpus —
Long-tailed Cuckoo (Asia), Great Bowerbird (Australia), Rufous-tailed Nightjar
(Neotropics), one at 0.90 confidence.

Design commitments, all from §4.3:

* **Never silently drop an out-of-range candidate.** An improbable identification is
  either a model error or a genuinely notable record, and both deserve a human.
  Candidates are re-scored and flagged, never deleted.
* **Cache aggressively**, keyed on rounded coordinates plus month. A photographer's
  work clusters in a handful of places, so hit rates are high.
* **Degrade gracefully.** No coordinates, no network, or an API outage must not break
  the pipeline — re-ranking is skipped and the result says so.

GBIF needs no API key and covers every taxon. eBird has far denser bird data but
requires a free token; it is optional and additive.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

GBIF_SEARCH = "https://api.gbif.org/v1/occurrence/search"
EBIRD_SPPLIST = "https://api.ebird.org/v2/product/spplist/{region}"
USER_AGENT = "melampus/0.1 (local wildlife photo triage; +https://github.com/gregbishop/melampus)"


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    radius_km: float = 50.0

    def rounded(self, places: int = 1) -> tuple[float, float]:
        """Coordinates rounded for cache keying.

        One decimal place is ~11 km, comfortably inside the default search radius,
        so nearby shoots share cache entries instead of each paying for a lookup.
        """
        return (round(self.latitude, places), round(self.longitude, places))


@dataclass
class RangeVerdict:
    """What occurrence data says about one candidate."""

    scientific_name: str
    records: int | None          # None when the lookup could not be performed
    in_range: bool
    notable: bool = False        # present but scarce — the interesting pile
    source: str = "gbif"

    @property
    def known(self) -> bool:
        return self.records is not None


@dataclass
class RerankOutcome:
    """Result of re-ranking, including why nothing happened when nothing did."""

    applied: bool
    reason: str = ""
    verdicts: dict[str, RangeVerdict] = field(default_factory=dict)
    range_flag: bool = False     # top candidate is improbable for this place and month
    notes: list[str] = field(default_factory=list)


class OccurrenceCache:
    """Tiny JSON-backed cache keyed on species, rounded coordinates and month."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, int] = {}
        if self.path.is_file():
            try:
                self._data = json.loads(self.path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}
        self._dirty = False

    @staticmethod
    def key(name: str, location: Location, month: int | None) -> str:
        lat, lon = location.rounded()
        return f"{name.strip().lower()}|{lat}|{lon}|{location.radius_km:g}|{month or 0}"

    def get(self, key: str) -> int | None:
        return self._data.get(key)

    def put(self, key: str, count: int) -> None:
        self._data[key] = count
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self.path.write_text(json.dumps(self._data, indent=0), encoding="utf-8")
            self._dirty = False


class GBIFClient:
    """Open occurrence counts. No key required, all taxa."""

    def __init__(self, cache: OccurrenceCache | None = None, timeout: float = 20.0,
                 min_interval: float = 0.2) -> None:
        self.cache = cache
        self.timeout = timeout
        self.min_interval = min_interval  # be a polite API citizen
        self._last_call = 0.0

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.monotonic()

    def count(self, scientific_name: str, location: Location, month: int | None) -> int | None:
        """Occurrence records for a species near a place, optionally in one month.

        Returns None when the lookup could not be performed — offline, timeout, or an
        API error. None is not zero, and the caller must not treat it as absence.
        """
        if not scientific_name.strip():
            return None

        key = OccurrenceCache.key(scientific_name, location, month)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        params = {
            "scientificName": scientific_name.strip(),
            "geoDistance": f"{location.latitude},{location.longitude},{location.radius_km:g}km",
            "limit": 0,
        }
        if month:
            params["month"] = str(month)

        url = f"{GBIF_SEARCH}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            self._throttle()
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

        count = payload.get("count")
        if not isinstance(count, int):
            return None
        if self.cache is not None:
            self.cache.put(key, count)
        return count


#: Taxa that are not organisms. Occurrence data says nothing about them, and a
#: GBIF lookup for "CrossFit" would return zero and wrongly flag it out of range.
NON_ORGANISM_TAXA = frozenset({
    "football", "fitness", "field_sport", "court_sport",
    "running", "team_other", "people", "none",
})


def applies_to(taxon: str | None) -> bool:
    """Whether occurrence re-ranking is meaningful for this taxon at all."""
    return (taxon or "").strip().lower() not in NON_ORGANISM_TAXA


def rerank(
    candidates: list,
    location: Location | None,
    month: int | None,
    client: GBIFClient | None,
    *,
    taxon: str | None = None,
    absent_penalty: float = 0.15,
    notable_threshold: int = 25,
    notable_penalty: float = 0.6,
) -> RerankOutcome:
    """Re-score candidates against occurrence data. Never drops any of them.

    A candidate with no regional records is multiplied down hard but kept, and the
    photo is flagged for review. A candidate present but scarce is demoted more
    gently and marked notable — that is the pile worth looking at, because it holds
    both the model's mistakes and any genuinely unusual record.

    Confidence here is an ordinal hint, not a probability, so multiplying it is a
    ranking operation and the results should not be read as calibrated.
    """
    if not candidates:
        return RerankOutcome(applied=False, reason="no candidates to re-rank")
    if taxon is not None and not applies_to(taxon):
        return RerankOutcome(
            applied=False,
            reason=f"occurrence data does not apply to '{taxon}'")
    if location is None:
        return RerankOutcome(applied=False, reason="no location available for this photo")
    if client is None:
        return RerankOutcome(applied=False, reason="occurrence lookups disabled")

    verdicts: dict[str, RangeVerdict] = {}
    notes: list[str] = []
    looked_up = 0

    for cand in candidates:
        name = (getattr(cand, "scientific_name", "") or "").strip()
        if not name:
            notes.append(f"{getattr(cand, 'common_name', '?')}: no scientific name to look up")
            continue
        count = client.count(name, location, month)
        if count is None:
            notes.append(f"{name}: lookup unavailable")
            verdicts[name] = RangeVerdict(name, None, in_range=True)
            continue
        looked_up += 1
        verdicts[name] = RangeVerdict(
            scientific_name=name,
            records=count,
            in_range=count > 0,
            notable=0 < count < notable_threshold,
        )

    if not looked_up:
        return RerankOutcome(
            applied=False,
            reason="no occurrence data could be retrieved",
            verdicts=verdicts,
            notes=notes,
        )

    original_top = candidates[0]

    def adjusted(cand) -> float:
        name = (getattr(cand, "scientific_name", "") or "").strip()
        verdict = verdicts.get(name)
        base = float(getattr(cand, "confidence", 0.0) or 0.0)
        if verdict is None or not verdict.known:
            return base
        if not verdict.in_range:
            return base * absent_penalty
        if verdict.notable:
            return base * notable_penalty
        return base

    # Stable sort: equal scores keep the model's original ordering.
    candidates.sort(key=adjusted, reverse=True)

    top_name = (getattr(candidates[0], "scientific_name", "") or "").strip()
    top_verdict = verdicts.get(top_name)
    range_flag = bool(
        top_verdict and top_verdict.known and (not top_verdict.in_range or top_verdict.notable)
    )
    if candidates[0] is not original_top:
        notes.append(
            f"re-ranked: {getattr(original_top, 'common_name', '?')} -> "
            f"{getattr(candidates[0], 'common_name', '?')}"
        )

    return RerankOutcome(
        applied=True,
        reason="",
        verdicts=verdicts,
        range_flag=range_flag,
        notes=notes,
    )


def range_lookup(settings) -> tuple[GBIFClient, Location] | None:
    """The client and place a range check needs, or None when §4.3 cannot run.

    None when lookups are disabled or no default location is configured. Callers
    treat None as "skip the check", never as "everything is in range".
    """
    if not settings.enabled:
        return None
    if settings.default_latitude is None or settings.default_longitude is None:
        return None
    where = Location(settings.default_latitude, settings.default_longitude, settings.radius_km)
    return GBIFClient(cache=OccurrenceCache(settings.cache_path)), where
