"""Pipeline tests that need no model weights.

Everything here runs against ScriptedBackend, so parsing, validation, retry, caching
and the no-leak guarantee are all verifiable in milliseconds and in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from melampus.backend import ScriptedBackend
from melampus.cache import ResultCache
from melampus.config import load_config
from melampus.identify import Identifier, extract_json
from melampus.images import NEUTRAL_NAME, content_hash, staged_pixels
from melampus.prompts import PromptError, PromptLibrary
from melampus.runner import run_batch
from melampus.schema import Identification, Taxon

REPO = Path(__file__).resolve().parents[2]

ROUTING_OK = '{"taxon": "bird", "confidence": 0.9, "reasoning": "long-legged wader"}'
ID_OK = json.dumps(
    {
        "taxon": "bird",
        "candidates": [
            {"common_name": "Tricolored Heron", "scientific_name": "Egretta tricolor",
             "confidence": 0.8, "reasoning": "white belly against dark body"},
            {"common_name": "Little Blue Heron", "scientific_name": "Egretta caerulea",
             "confidence": 0.2, "reasoning": "similar structure but lacks white belly"},
        ],
        "age_sex": "adult",
        "count": 1,
        "behavior": ["wading"],
        "diagnostic_features_visible": True,
        "abstain": False,
        "abstain_reason": None,
    }
)


@pytest.fixture()
def photo(tmp_path: Path) -> Path:
    path = tmp_path / "SECRET_SPECIES_NAME.jpg"
    Image.new("RGB", (2400, 1600), (90, 120, 70)).save(path, exif=b"")
    return path


@pytest.fixture()
def config(tmp_path: Path):
    return load_config(
        run={"prompts_dir": str(REPO / "prompts"), "cache_path": str(tmp_path / "c.jsonl")}
    )


# --------------------------------------------------------------------------- #
# the leak guarantee
# --------------------------------------------------------------------------- #
def test_staged_image_is_renamed_and_stripped(photo: Path):
    """The model must never see the filename, and never see EXIF/XMP."""
    with staged_pixels(photo, max_edge=800) as staged:
        assert staged.name == NEUTRAL_NAME, "original filename reached the staged file"
        with Image.open(staged) as img:
            assert max(img.size) <= 800
            assert not img.getexif(), "EXIF survived staging"
            assert "XML:com.adobe.xmp" not in img.info


def test_backend_never_receives_original_filename(photo: Path, config):
    backend = ScriptedBackend([ROUTING_OK, ID_OK])
    Identifier(backend, config).identify(photo)

    assert backend.calls, "backend was never called"
    for image_path, prompt in backend.calls:
        assert image_path.name == NEUTRAL_NAME
        assert "SECRET_SPECIES_NAME" not in str(image_path)
        assert "SECRET_SPECIES_NAME" not in prompt


def test_prompt_rejects_unapproved_context():
    library = PromptLibrary(REPO / "prompts")
    with pytest.raises(PromptError):
        library.render("bird", filename="0A1A2475.jpg")


# --------------------------------------------------------------------------- #
# defensive parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    [
        '{"taxon": "bird", "confidence": 0.5}',
        'Here is the result:\n```json\n{"taxon": "bird", "confidence": 0.5}\n```\nHope that helps!',
        'Sure! {"taxon": "bird", "confidence": 0.5} Let me know if you need more.',
        '```\n{"taxon": "bird", "confidence": 0.5}\n```',
    ],
)
def test_extract_json_survives_model_chatter(raw: str):
    assert extract_json(raw) == {"taxon": "bird", "confidence": 0.5}


def test_extract_json_handles_braces_inside_strings():
    payload = extract_json('{"reasoning": "the shape {like this} is odd", "taxon": "bird"}')
    assert payload["taxon"] == "bird"


def test_extract_json_returns_none_on_garbage():
    assert extract_json("I cannot identify this image.") is None


# --------------------------------------------------------------------------- #
# retry and graceful failure
# --------------------------------------------------------------------------- #
def test_one_corrective_retry_then_success(photo: Path, config):
    backend = ScriptedBackend(["not json at all", ROUTING_OK, ID_OK])
    result = Identifier(backend, config).identify(photo)

    assert result.status == "ok"
    assert result.retries == 1
    assert "could not be parsed" in backend.calls[1][1]


def test_marks_unprocessed_rather_than_writing_garbage(photo: Path, config):
    backend = ScriptedBackend(["garbage", "still garbage", "and again", "and again"])
    result = Identifier(backend, config).identify(photo)

    assert result.status == "unprocessed"
    assert result.identification is None, "must not invent an identification"


def test_none_taxon_short_circuits_before_species_stage(photo: Path, config):
    backend = ScriptedBackend(['{"taxon": "none", "confidence": 0.95, "reasoning": "empty"}'])
    result = Identifier(backend, config).identify(photo)

    assert result.status == "ok"
    assert result.identification.taxon is Taxon.NONE
    assert result.identification.abstain is True
    assert len(backend.calls) == 1, "ran the species stage on a frame with no organism"


def test_abstention_clears_candidates(photo: Path, config):
    contradictory = json.dumps({
        "taxon": "bird", "abstain": True, "abstain_reason": "facing away",
        "candidates": [{"common_name": "Guess", "confidence": 0.4}],
    })
    backend = ScriptedBackend([ROUTING_OK, contradictory])
    result = Identifier(backend, config).identify(photo)

    assert result.identification.abstain is True
    assert result.identification.candidates == []


# --------------------------------------------------------------------------- #
# schema coercion
# --------------------------------------------------------------------------- #
def test_taxon_aliases_are_normalised():
    assert Identification.model_validate({"taxon": "Birds"}).taxon is Taxon.BIRD
    assert Identification.model_validate({"taxon": "alligator"}).taxon is Taxon.REPTILE


def test_behavior_string_is_coerced_to_list():
    assert Identification.model_validate({"taxon": "bird", "behavior": "perched"}).behavior == ["perched"]


def test_count_from_prose_is_coerced():
    assert Identification.model_validate({"taxon": "bird", "count": "2 birds"}).count == 2


# --------------------------------------------------------------------------- #
# cache, resume, idempotency
# --------------------------------------------------------------------------- #
def test_second_run_is_a_no_op(tmp_path: Path, config):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(3):
        Image.new("RGB", (400, 300), (i * 40, 100, 100)).save(folder / f"f{i}.jpg")
    paths = sorted(folder.glob("*.jpg"))
    cache = ResultCache(config.run.cache_path)

    first = run_batch(
        paths,
        Identifier(ScriptedBackend([ROUTING_OK, ID_OK] * 3), config),
        cache, log=lambda _: None,
    )
    assert first.processed == 3 and first.skipped == 0

    second = run_batch(
        paths, Identifier(ScriptedBackend([]), config), cache, log=lambda _: None
    )
    assert second.processed == 0
    assert second.skipped == 3


def test_cache_is_keyed_on_content_not_path(tmp_path: Path, config):
    original = tmp_path / "a.jpg"
    Image.new("RGB", (400, 300), (10, 20, 30)).save(original)
    cache = ResultCache(config.run.cache_path)
    identifier = Identifier(ScriptedBackend([ROUTING_OK, ID_OK]), config)

    run_batch([original], identifier, cache, log=lambda _: None)
    renamed = tmp_path / "b.jpg"
    original.rename(renamed)

    again = run_batch([renamed], identifier, cache, log=lambda _: None)
    assert again.skipped == 1, "renaming a file forced needless reprocessing"


def test_batch_survives_a_bad_file(tmp_path: Path, config):
    folder = tmp_path / "imgs"
    folder.mkdir()
    Image.new("RGB", (400, 300), (10, 20, 30)).save(folder / "good.jpg")
    (folder / "broken.jpg").write_bytes(b"this is not a JPEG")

    stats = run_batch(
        sorted(folder.glob("*.jpg")),
        Identifier(ScriptedBackend([ROUTING_OK, ID_OK]), config),
        ResultCache(config.run.cache_path),
        log=lambda _: None,
    )
    assert stats.total == 2
    assert stats.errors == 1
    assert stats.ok == 1, "one corrupt file aborted the batch"


def test_content_hash_is_stable(tmp_path: Path):
    path = tmp_path / "x.jpg"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(path)
    assert content_hash(path) == content_hash(path)
