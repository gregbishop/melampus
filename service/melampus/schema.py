"""The strict output contract from CLAUDE.md §4.2.

Local VLMs are markedly less reliable at strict JSON than frontier cloud models, so
everything the model returns is validated against these models before it is allowed
anywhere near a result. Validation failure is a first-class outcome: retry once with
a corrective message, then mark the photo unprocessed rather than write garbage.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Taxon(StrEnum):
    BIRD = "bird"
    FISH = "fish"
    REPTILE = "reptile"
    AMPHIBIAN = "amphibian"
    INSECT = "insect"
    ARACHNID = "arachnid"
    MAMMAL = "mammal"
    PLANT = "plant"
    FUNGUS = "fungus"
    # Human-activity subjects. They share the Identification shape deliberately:
    # candidates carry the sport, behavior carries the action, so gates, keywords,
    # review and correction all work unchanged.
    FOOTBALL = "football"
    FITNESS = "fitness"
    FIELD_SPORT = "field_sport"
    COURT_SPORT = "court_sport"
    RUNNING = "running"
    TEAM_OTHER = "team_other"
    PEOPLE = "people"
    NONE = "none"


class TaxonRouting(BaseModel):
    """Stage A output: cheap coarse routing so Stage B can use a specialised prompt."""

    model_config = ConfigDict(extra="ignore")

    taxon: Taxon
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("taxon", mode="before")
    @classmethod
    def _normalise(cls, value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            aliases = {
                "birds": "bird", "avian": "bird",
                "reptiles": "reptile", "snake": "reptile", "lizard": "reptile",
                "crocodilian": "reptile", "turtle": "reptile", "alligator": "reptile",
                "mammals": "mammal", "insects": "insect", "bug": "insect",
                "plants": "plant", "flower": "plant", "tree": "plant",
                "fungi": "fungus", "mushroom": "fungus",
                "spider": "arachnid", "fishes": "fish",
                "american football": "football", "gridiron": "football",
                "crossfit": "fitness", "gym": "fitness", "weightlifting": "fitness",
                "soccer": "field_sport", "rugby": "field_sport",
                "basketball": "court_sport", "volleyball": "court_sport",
                "person": "people", "portrait": "people",
                "frog": "amphibian", "toad": "amphibian",
                "nothing": "none", "no organism": "none", "n/a": "none", "": "none",
            }
            return aliases.get(cleaned, cleaned)
        return value


class Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    common_name: str
    scientific_name: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class Identification(BaseModel):
    """Stage B output. Mirrors the JSON block in CLAUDE.md §4.2 exactly."""

    model_config = ConfigDict(extra="ignore")

    taxon: Taxon
    candidates: list[Candidate] = Field(default_factory=list)
    age_sex: str = "indeterminate"
    count: int = 1
    behavior: list[str] = Field(default_factory=list)
    diagnostic_features_visible: bool = True
    abstain: bool = False
    abstain_reason: str | None = None

    @field_validator("taxon", mode="before")
    @classmethod
    def _normalise_taxon(cls, value: object) -> object:
        return TaxonRouting._normalise(value)

    @field_validator("behavior", mode="before")
    @classmethod
    def _listify(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value

    @field_validator("count", mode="before")
    @classmethod
    def _coerce_count(cls, value: object) -> object:
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else 1
        return value

    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def ranked(self) -> list[Candidate]:
        return sorted(self.candidates, key=lambda c: c.confidence, reverse=True)


class ImageResult(BaseModel):
    """What the runner persists per image."""

    model_config = ConfigDict(extra="ignore")

    file: str
    content_hash: str
    status: str  # ok | unprocessed | error
    taxon_routing: TaxonRouting | None = None
    identification: Identification | None = None
    error: str | None = None
    retries: int = 0
    seconds: float = 0.0
    model: str = ""
    image_max_edge: int = 0
    # Identifies the model + prompt set + image settings that produced this result,
    # so a prompt edit invalidates it instead of being silently re-served.
    run_fingerprint: str = ""

    # Cloud escalation provenance (CLAUDE.md §6.6). Set only on results produced by
    # the optional Claude API pass. `local_identification` keeps what the local model
    # said, which is what makes local-vs-cloud agreement measurable after the fact —
    # the number that decides whether escalation is worth paying for at all.
    escalated: bool = False
    escalation_model: str = ""
    escalation_reason: str = ""
    local_identification: Identification | None = None
