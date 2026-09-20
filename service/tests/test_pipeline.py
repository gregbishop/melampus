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


def _metadata_laden(path: Path) -> dict[str, bytes | str]:
    """An image carrying every metadata channel a real camera file would.

    The previous version of the stripping test used a freshly-constructed image
    with no metadata at all, so it asserted the absence of something that was
    never there. It would have passed unchanged if staging had been rewritten to
    copy EXIF straight through. This builds the adversarial case instead.
    """
    exif = Image.Exif()
    exif[0x010F] = "Canon"                       # Make
    exif[0x0110] = "Canon EOS R3"                # Model
    exif[0x013B] = "SECRET_PHOTOGRAPHER_NAME"    # Artist
    exif[0x010E] = "SECRET_CAPTION_TEXT"         # ImageDescription
    exif[0x0132] = "2026:06:14 05:23:37"         # DateTime
    exif[0x8825] = {1: "N", 2: (28.0, 39.0, 0.0), 3: "W", 4: (80.0, 43.0, 0.0)}  # GPS IFD

    markers: dict[str, bytes | str] = {
        "exif": exif.tobytes(),
        "xmp": b'<?xpacket?><x:xmpmeta xmlns:x="adobe:ns:meta/">'
               b"<dc:subject>SECRET_KEYWORD</dc:subject></x:xmpmeta>",
        "comment": b"SECRET_JFIF_COMMENT",
        "icc_profile": b"\x00\x00\x02\x0cSECRET_ICC_PROFILE" + b"\x00" * 500,
    }
    Image.new("RGB", (1200, 800), (70, 100, 60)).save(path, format="JPEG", **markers)
    return markers


def test_staging_strips_every_metadata_channel(tmp_path: Path):
    """The project's single most important rule, tested adversarially.

    Only pixels may reach the model. This builds a file carrying EXIF (including
    maker, model, artist, caption and a GPS IFD), an XMP packet with a keyword, a
    JFIF comment and an ICC profile, then asserts none of it survives staging.
    """
    source = tmp_path / "SECRET_SPECIES_NAME.jpg"
    _metadata_laden(source)

    # Sanity: the fixture must actually carry the metadata, or the test is vacuous
    # in exactly the way the old one was.
    raw_before = source.read_bytes()
    for secret in (b"SECRET_PHOTOGRAPHER_NAME", b"SECRET_CAPTION_TEXT",
                   b"SECRET_KEYWORD", b"SECRET_JFIF_COMMENT", b"SECRET_ICC_PROFILE"):
        assert secret in raw_before, f"fixture never carried {secret!r} — test would be vacuous"

    with staged_pixels(source, max_edge=800) as staged:
        raw_after = staged.read_bytes()

        assert staged.name == NEUTRAL_NAME, "original filename reached the staged file"
        for secret in (b"SECRET_PHOTOGRAPHER_NAME", b"SECRET_CAPTION_TEXT",
                       b"SECRET_KEYWORD", b"SECRET_JFIF_COMMENT", b"SECRET_ICC_PROFILE"):
            assert secret not in raw_after, f"{secret!r} survived staging"

        # And nothing camera-identifying by name, in case a channel is added later.
        for token in (b"Canon", b"EOS R3", b"2026:06:14"):
            assert token not in raw_after, f"{token!r} survived staging"

        with Image.open(staged) as img:
            assert not img.getexif(), "EXIF survived staging"
            assert "XML:com.adobe.xmp" not in img.info
            assert "icc_profile" not in img.info
            assert "comment" not in img.info
            assert max(img.size) <= 800


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


def test_the_cache_exports_the_records_it_holds_as_one_list_of_dicts(tmp_path: Path, config):
    """--json-out and --plugin-out read the cache through one method, so a
    record looks the same in both files."""
    photo = tmp_path / "a.jpg"
    Image.new("RGB", (400, 300), (10, 20, 30)).save(photo)
    cache = ResultCache(config.run.cache_path)
    run_batch([photo], Identifier(ScriptedBackend([ROUTING_OK, ID_OK]), config), cache,
              log=lambda _: None)
    out = tmp_path / "out.json"

    cache.export_json(out)

    records = cache.records()
    assert records == json.loads(out.read_text(encoding="utf-8"))
    assert [r["file"] for r in records] == ["a.jpg"]
    assert records[0]["identification"]["candidates"][0]["common_name"] == "Tricolored Heron"


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


# --------------------------------------------------------------------------- #
# cache invalidation on prompt / model change
# --------------------------------------------------------------------------- #
def test_editing_a_prompt_invalidates_cached_results(tmp_path: Path, config):
    """A tuning pass must actually re-run, not silently re-serve old answers."""
    import shutil

    prompts = tmp_path / "prompts"
    shutil.copytree(REPO / "prompts", prompts)
    cfg = config.model_copy(update={"run": config.run.model_copy(update={"prompts_dir": prompts})})

    folder = tmp_path / "imgs"
    folder.mkdir()
    Image.new("RGB", (400, 300), (10, 20, 30)).save(folder / "a.jpg")
    paths = sorted(folder.glob("*.jpg"))
    cache = ResultCache(cfg.run.cache_path)

    first = run_batch(paths, Identifier(ScriptedBackend([ROUTING_OK, ID_OK]), cfg), cache,
                      log=lambda _: None)
    assert first.processed == 1

    again = run_batch(paths, Identifier(ScriptedBackend([]), cfg), cache, log=lambda _: None)
    assert again.skipped == 1, "unchanged prompts should still hit the cache"

    (prompts / "bird.md").write_text("A different prompt entirely.\n$season_context\n")
    after = run_batch(paths, Identifier(ScriptedBackend([ROUTING_OK, ID_OK]), cfg), cache,
                      log=lambda _: None)
    assert after.processed == 1, "prompt edit did not invalidate the cached result"
    assert after.skipped == 0


def test_changing_the_model_invalidates_cached_results(tmp_path: Path, config):
    folder = tmp_path / "imgs"
    folder.mkdir()
    Image.new("RGB", (400, 300), (10, 20, 30)).save(folder / "a.jpg")
    paths = sorted(folder.glob("*.jpg"))
    cache = ResultCache(config.run.cache_path)

    run_batch(paths, Identifier(ScriptedBackend([ROUTING_OK, ID_OK], name="model-a"), config),
              cache, log=lambda _: None)
    other = run_batch(paths, Identifier(ScriptedBackend([ROUTING_OK, ID_OK], name="model-b"), config),
                      cache, log=lambda _: None)
    assert other.processed == 1, "a different model reused another model's answer"


def test_results_predating_fingerprinting_are_still_honoured(tmp_path: Path, config):
    """Introducing the fingerprint must not discard hours of prior work."""
    folder = tmp_path / "imgs"
    folder.mkdir()
    Image.new("RGB", (400, 300), (10, 20, 30)).save(folder / "a.jpg")
    paths = sorted(folder.glob("*.jpg"))
    cache = ResultCache(config.run.cache_path)
    identifier = Identifier(ScriptedBackend([ROUTING_OK, ID_OK]), config)

    run_batch(paths, identifier, cache, log=lambda _: None)
    # Simulate a record written before fingerprints existed.
    for record in cache.results():
        record.run_fingerprint = ""
    assert cache.legacy_count() == 1
    assert cache.has_success(cache.results()[0].content_hash, "some-other-fingerprint")


# --------------------------------------------------------------------------- #
# adaptive downscale ladder (mlx-vlm prompt-token ceiling workaround)
# --------------------------------------------------------------------------- #
class _SizeSensitiveBackend(ScriptedBackend):
    """Mimics the mlx-vlm 0.6.7 defect: empty generation above a size threshold.

    The real runtime returns a single EOS token once the prompt (dominated by vision
    tokens) passes ~2.1k. Reproducing that as a size cutoff lets the fallback ladder
    be tested without weights.
    """

    def __init__(self, fails_above: int, responses: list[str]) -> None:
        super().__init__(responses, name="size-sensitive")
        self.fails_above = fails_above
        self.seen_edges: list[int] = []

    def complete(self, image_path: Path, prompt: str, max_tokens: int):
        with Image.open(image_path) as img:
            edge = max(img.size)
        self.seen_edges.append(edge)
        if edge > self.fails_above:
            self.calls.append((image_path, prompt))
            from melampus.backend import Completion

            return Completion(text="", seconds=0.0)
        return super().complete(image_path, prompt, max_tokens)


def test_downscale_ladder_recovers_when_the_full_size_returns_nothing(photo: Path, config):
    """An empty generation at 1280 must be retried smaller, not reported as failure."""
    cfg = config.model_copy(
        update={"image": config.image.model_copy(
            update={"max_edge": 1280, "fallback_edges": [1024, 768]})}
    )
    backend = _SizeSensitiveBackend(fails_above=900, responses=[ROUTING_OK, ID_OK])
    result = Identifier(backend, cfg).identify(photo)

    assert result.status == "ok", "ladder did not recover from an empty generation"
    assert result.image_max_edge == 768, f"recovered at {result.image_max_edge}, expected 768"
    assert 1280 in backend.seen_edges, "never attempted the configured size first"
    assert backend.seen_edges[0] == 1280, "did not try largest first"


def test_ladder_records_the_size_that_worked(photo: Path, config):
    """Silent degradation is worse than degradation you can see."""
    cfg = config.model_copy(
        update={"image": config.image.model_copy(
            update={"max_edge": 1280, "fallback_edges": [1024]})}
    )
    backend = _SizeSensitiveBackend(fails_above=1100, responses=[ROUTING_OK, ID_OK])
    result = Identifier(backend, cfg).identify(photo)

    assert result.status == "ok"
    assert result.image_max_edge == 1024


def test_ladder_gives_up_rather_than_looping(photo: Path, config):
    """When every size fails the image is unprocessed, not retried forever."""
    cfg = config.model_copy(
        update={"image": config.image.model_copy(
            update={"max_edge": 1280, "fallback_edges": [1024, 768]})}
    )
    backend = _SizeSensitiveBackend(fails_above=1, responses=[])
    result = Identifier(backend, cfg).identify(photo)

    assert result.status == "unprocessed"
    assert result.identification is None
    assert set(backend.seen_edges) == {1280, 1024, 768}
