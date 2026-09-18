"""Cloud escalation for the low-confidence tail (CLAUDE.md §6.6).

Every test here runs offline. The selection logic is pure, and the Anthropic
backend takes an injected client, so the decision of *what* gets sent to a paid
API — and the guarantee that only pixels go — is verifiable without a key, a
network, or a bill.

The rules being pinned down:

* off unless explicitly enabled, and refused without a key;
* a confident "no organism here" is never escalated, because it is an answer,
  not an uncertainty, and 800 empty frames would be an expensive way to
  rediscover that;
* when the cap bites, the most uncertain frames go first and the dropped ones
  are reported rather than silently truncated;
* the cloud sees the same metadata-free staged pixels the local model sees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from melampus.backend import AnthropicBackend, ScriptedBackend
from melampus.cache import ResultCache
from melampus.config import load_config
from melampus.escalate import (
    EscalationRefused,
    build_cloud_identifier,
    escalate,
    estimate_cost_usd,
    resolve_api_key,
    resolve_model,
    select_for_escalation,
    should_escalate,
)
from melampus.identify import Identifier
from melampus.images import NEUTRAL_NAME
from melampus.schema import Candidate, Identification, ImageResult, Taxon, TaxonRouting

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def make_result(
    file: str = "a.jpg",
    *,
    status: str = "ok",
    abstain: bool = False,
    taxon: Taxon = Taxon.BIRD,
    confidences: tuple[float, ...] = (0.95,),
    content_hash: str = "hash-a",
) -> ImageResult:
    candidates = [
        Candidate(common_name=f"Species {i}", scientific_name=f"Genus sp{i}", confidence=c)
        for i, c in enumerate(confidences)
    ]
    return ImageResult(
        file=file,
        content_hash=content_hash,
        status=status,
        taxon_routing=TaxonRouting(taxon=taxon, confidence=0.9),
        identification=Identification(
            taxon=taxon,
            candidates=[] if abstain else candidates,
            abstain=abstain,
            abstain_reason="not visible" if abstain else None,
        ),
    )


@pytest.fixture()
def cfg():
    # use_local=False throughout: melampus.local.toml is git-ignored, so reading it
    # here would assert on the developer's machine rather than on the defaults, and
    # would fail for anyone who follows the docs and puts a key in it.
    return load_config(use_local=False).escalation


# --------------------------------------------------------------------------- #
# the default posture
# --------------------------------------------------------------------------- #

def test_escalation_is_off_by_default():
    """§6.6: keep it off by default. Nothing leaves the machine unasked.

    Asserted on the class, not on a loaded config, so a local override cannot make
    this pass or fail for the wrong reason.
    """
    from melampus.config import EscalationConfig

    assert EscalationConfig().enabled is False


def test_escalation_refuses_when_disabled(tmp_path):
    config = load_config(use_local=False, escalation={"enabled": False, "api_key": "sk-test"})
    with pytest.raises(EscalationRefused, match="not enabled"):
        escalate(
            [],
            ResultCache(tmp_path / "local.jsonl"),
            ResultCache(tmp_path / "cloud.jsonl"),
            identifier=None,
            config=config,
        )


def test_escalation_refuses_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MELAMPUS_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config(use_local=False, escalation={"enabled": True})
    with pytest.raises(EscalationRefused, match="API key"):
        escalate(
            [],
            ResultCache(tmp_path / "local.jsonl"),
            ResultCache(tmp_path / "cloud.jsonl"),
            identifier=None,
            config=config,
        )


def test_dry_run_needs_no_key(tmp_path, monkeypatch):
    """You should be able to see the count and the cost before deciding whether to
    sign up for an API key at all."""
    monkeypatch.delenv("MELAMPUS_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config(use_local=False, escalation={"enabled": True})
    run = escalate(
        [],
        ResultCache(tmp_path / "local.jsonl"),
        ResultCache(tmp_path / "cloud.jsonl"),
        identifier=None,
        config=config,
        dry_run=True,
    )
    assert run.selected == 0
    assert run.processed == 0


def test_key_resolution_order(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-generic-env")
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "from-melampus-env")

    explicit = load_config(use_local=False, escalation={"api_key": "from-config"}).escalation
    assert resolve_api_key(explicit) == "from-config"

    assert resolve_api_key(load_config(use_local=False).escalation) == "from-melampus-env"

    monkeypatch.delenv("MELAMPUS_ANTHROPIC_KEY")
    assert resolve_api_key(load_config(use_local=False).escalation) == "from-generic-env"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert resolve_api_key(load_config(use_local=False).escalation) is None


def test_api_key_is_never_written_into_tracked_config():
    """The repo is public. A key must only ever arrive from the environment or the
    git-ignored local file — never from anything committed."""
    from melampus.config import EscalationConfig

    assert EscalationConfig().api_key is None
    assert load_config(use_local=False).escalation.api_key is None


# --------------------------------------------------------------------------- #
# what gets escalated
# --------------------------------------------------------------------------- #

def test_confident_in_range_identification_is_left_alone(cfg):
    assert should_escalate(make_result(confidences=(0.95, 0.2)), config=cfg) is None


def test_low_confidence_is_escalated(cfg):
    reason = should_escalate(make_result(confidences=(0.55, 0.3)), config=cfg)
    assert reason is not None and "confidence" in reason


def test_abstention_is_escalated(cfg):
    reason = should_escalate(make_result(abstain=True), config=cfg)
    assert reason is not None and "abstain" in reason.lower()


def test_unprocessed_is_escalated(cfg):
    result = ImageResult(file="a.jpg", content_hash="h", status="unprocessed",
                         error="identification failed")
    reason = should_escalate(result, config=cfg)
    assert reason is not None and "unprocessed" in reason.lower()


def test_unreadable_file_is_not_escalated(cfg):
    """A corrupt file is not a hard identification. A better model cannot read it either."""
    result = ImageResult(file="a.jpg", content_hash="h", status="error", error="unreadable")
    assert should_escalate(result, config=cfg) is None


def test_range_flag_is_escalated(cfg):
    """§4.3 routes range-flagged frames to human review; this offers a second opinion first."""
    confident = make_result(confidences=(0.95,))
    assert should_escalate(confident, config=cfg) is None
    reason = should_escalate(confident, config=cfg, range_flagged=True)
    assert reason is not None and "range" in reason.lower()


def test_confident_empty_frame_is_not_escalated(cfg):
    """`taxon: none` abstains by construction. Escalating those would burn the budget
    on frames the pipeline is already certain about."""
    empty = ImageResult(
        file="a.jpg", content_hash="h", status="ok",
        identification=Identification(
            taxon=Taxon.NONE, candidates=[], abstain=True,
            abstain_reason="no organism present in frame",
        ),
    )
    assert should_escalate(empty, config=cfg) is None


def test_already_escalated_is_not_escalated_again(cfg):
    """Idempotency (§5.3). A second run must not re-bill the same frame."""
    result = make_result(confidences=(0.4,))
    assert should_escalate(result, config=cfg) is not None
    result.escalated = True
    assert should_escalate(result, config=cfg) is None


def test_toggles_can_disable_each_trigger():
    off = load_config(use_local=False, escalation={"on_abstain": False, "on_range_flag": False}).escalation
    assert should_escalate(make_result(abstain=True), config=off) is None
    assert should_escalate(make_result(confidences=(0.95,)), config=off, range_flagged=True) is None
    # Low confidence still escalates; it is the primary trigger.
    assert should_escalate(make_result(confidences=(0.4,)), config=off) is not None


# --------------------------------------------------------------------------- #
# selection, ordering and the cap
# --------------------------------------------------------------------------- #

def test_selection_orders_most_uncertain_first(cfg):
    results = [
        make_result("mid.jpg", confidences=(0.70,), content_hash="h-mid"),
        make_result("worst.jpg", abstain=True, content_hash="h-worst"),
        make_result("near.jpg", confidences=(0.79,), content_hash="h-near"),
    ]
    selected, dropped = select_for_escalation(results, config=cfg)
    assert [r.file for r, _ in selected] == ["worst.jpg", "mid.jpg", "near.jpg"]
    assert dropped == []


def test_cap_keeps_the_worst_and_reports_the_rest():
    """No silent truncation: what was left out has to be visible."""
    config = load_config(use_local=False, escalation={"max_images": 2}).escalation
    results = [
        make_result("a.jpg", confidences=(0.70,), content_hash="h-a"),
        make_result("b.jpg", confidences=(0.50,), content_hash="h-b"),
        make_result("c.jpg", confidences=(0.60,), content_hash="h-c"),
    ]
    selected, dropped = select_for_escalation(results, config=config)
    assert [r.file for r, _ in selected] == ["b.jpg", "c.jpg"]
    assert [r.file for r, _ in dropped] == ["a.jpg"]


def test_cost_estimate_scales_and_is_never_free():
    one = estimate_cost_usd(1)
    ten = estimate_cost_usd(10)
    assert one > 0
    assert ten == pytest.approx(one * 10)
    assert estimate_cost_usd(0) == 0


# --------------------------------------------------------------------------- #
# running it
# --------------------------------------------------------------------------- #

CLOUD_ROUTING = '{"taxon": "bird", "confidence": 0.95, "reasoning": "wader"}'
CLOUD_ID = json.dumps({
    "taxon": "bird",
    "candidates": [
        {"common_name": "Tricolored Heron", "scientific_name": "Egretta tricolor",
         "confidence": 0.93, "reasoning": "white belly visible against dark body"},
        {"common_name": "Little Blue Heron", "scientific_name": "Egretta caerulea",
         "confidence": 0.05, "reasoning": "no white belly, so ruled out"},
    ],
    "age_sex": "adult", "count": 1, "behavior": ["wading"],
    "diagnostic_features_visible": True, "abstain": False, "abstain_reason": None,
})


@pytest.fixture()
def photo(tmp_path: Path) -> Path:
    path = tmp_path / "SECRET_SPECIES_NAME.jpg"
    Image.new("RGB", (1800, 1200), (80, 110, 65)).save(path)
    return path


def _cloud_identifier(tmp_path: Path, responses: list[str]) -> tuple[Identifier, ScriptedBackend]:
    backend = ScriptedBackend(responses, name="claude-opus-5")
    config = load_config(use_local=False, run={"prompts_dir": str(REPO / "prompts"),
                              "cache_path": str(tmp_path / "unused.jsonl")})
    return Identifier(backend, config), backend


def test_escalation_replaces_a_weak_answer_and_records_provenance(tmp_path, photo, monkeypatch):
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    config = load_config(use_local=False, escalation={"enabled": True})

    local_cache = ResultCache(tmp_path / "local.jsonl")
    cloud_cache = ResultCache(tmp_path / "cloud.jsonl")

    from melampus.images import content_hash

    digest = content_hash(photo)
    local_cache.put(make_result(photo.name, confidences=(0.42,), content_hash=digest))

    identifier, _ = _cloud_identifier(tmp_path, [CLOUD_ROUTING, CLOUD_ID])
    run = escalate([photo], local_cache, cloud_cache, identifier=identifier, config=config)

    assert run.processed == 1
    assert run.dropped == 0

    record = cloud_cache.get(digest)
    assert record is not None
    assert record.status == "ok"
    assert record.identification.top().common_name == "Tricolored Heron"

    # Provenance: which model answered, why it was asked, and what the local model
    # had said. The last one is what makes local-vs-cloud agreement measurable.
    assert record.escalated is True
    assert record.escalation_model == "claude-opus-5"
    assert "confidence" in record.escalation_reason
    assert record.local_identification is not None
    assert record.local_identification.top().common_name == "Species 0"


def test_escalation_sends_pixels_only(tmp_path, photo, monkeypatch):
    """The CRITICAL constraint, and it matters more here than anywhere else: this is
    the one path where an image leaves the machine."""
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    config = load_config(use_local=False, escalation={"enabled": True})

    local_cache = ResultCache(tmp_path / "local.jsonl")
    cloud_cache = ResultCache(tmp_path / "cloud.jsonl")
    from melampus.images import content_hash

    local_cache.put(make_result(photo.name, abstain=True, content_hash=content_hash(photo)))

    identifier, backend = _cloud_identifier(tmp_path, [CLOUD_ROUTING, CLOUD_ID])
    escalate([photo], local_cache, cloud_cache, identifier=identifier, config=config)

    assert backend.calls, "the cloud backend was never called"
    for sent_path, prompt in backend.calls:
        assert sent_path.name == NEUTRAL_NAME
        assert "SECRET_SPECIES_NAME" not in str(sent_path)
        assert "SECRET_SPECIES_NAME" not in prompt


def test_escalation_leaves_the_local_cache_untouched(tmp_path, photo, monkeypatch):
    """Cloud answers go in their own file. Otherwise the next local run sees a
    foreign fingerprint, decides the frame is stale, and quietly overwrites a
    result that was paid for."""
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    config = load_config(use_local=False, escalation={"enabled": True})

    local_cache = ResultCache(tmp_path / "local.jsonl")
    cloud_cache = ResultCache(tmp_path / "cloud.jsonl")
    from melampus.images import content_hash

    digest = content_hash(photo)
    local_cache.put(make_result(photo.name, confidences=(0.30,), content_hash=digest))

    identifier, _ = _cloud_identifier(tmp_path, [CLOUD_ROUTING, CLOUD_ID])
    escalate([photo], local_cache, cloud_cache, identifier=identifier, config=config)

    assert local_cache.get(digest).identification.top().common_name == "Species 0"
    assert cloud_cache.get(digest).identification.top().common_name == "Tricolored Heron"


def test_second_run_is_a_no_op(tmp_path, photo, monkeypatch):
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    config = load_config(use_local=False, escalation={"enabled": True})
    local_cache = ResultCache(tmp_path / "local.jsonl")
    cloud_cache = ResultCache(tmp_path / "cloud.jsonl")
    from melampus.images import content_hash

    local_cache.put(make_result(photo.name, abstain=True, content_hash=content_hash(photo)))

    identifier, backend = _cloud_identifier(tmp_path, [CLOUD_ROUTING, CLOUD_ID])
    escalate([photo], local_cache, cloud_cache, identifier=identifier, config=config)
    calls_after_first = len(backend.calls)

    second = escalate([photo], local_cache, cloud_cache, identifier=identifier, config=config)
    assert second.processed == 0
    assert len(backend.calls) == calls_after_first


def test_run_estimate_uses_the_configured_rates(tmp_path, photo, monkeypatch):
    """Regression: the estimate function took rates but the run path ignored them,
    so switching provider printed a confidently wrong number."""
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    local_cache = ResultCache(tmp_path / "local.jsonl")
    from melampus.images import content_hash

    local_cache.put(make_result(photo.name, abstain=True, content_hash=content_hash(photo)))

    def run_with(rate: float) -> float:
        config = load_config(use_local=False, escalation={
            "enabled": True, "input_usd_per_mtok": rate, "output_usd_per_mtok": rate})
        run = escalate([photo], local_cache, ResultCache(tmp_path / f"c{rate}.jsonl"),
                       identifier=None, config=config, dry_run=True)
        assert run.selected == 1
        return run.estimated_cost_usd

    assert run_with(10.0) == pytest.approx(run_with(1.0) * 10)


def test_escalation_rejects_an_unknown_provider(tmp_path):
    config = load_config(use_local=False, escalation={"enabled": True, "provider": "gemini"})
    with pytest.raises(EscalationRefused, match="provider"):
        escalate([], ResultCache(tmp_path / "l.jsonl"), ResultCache(tmp_path / "c.jsonl"),
                 identifier=None, config=config, dry_run=True)


def test_dry_run_selects_but_spends_nothing(tmp_path, photo, monkeypatch):
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    config = load_config(use_local=False, escalation={"enabled": True})
    local_cache = ResultCache(tmp_path / "local.jsonl")
    cloud_cache = ResultCache(tmp_path / "cloud.jsonl")
    from melampus.images import content_hash

    local_cache.put(make_result(photo.name, abstain=True, content_hash=content_hash(photo)))

    identifier, backend = _cloud_identifier(tmp_path, [CLOUD_ROUTING, CLOUD_ID])
    run = escalate([photo], local_cache, cloud_cache, identifier=identifier,
                   config=config, dry_run=True)

    assert run.selected == 1
    assert run.processed == 0
    assert backend.calls == []
    assert run.estimated_cost_usd > 0


# --------------------------------------------------------------------------- #
# failures: what is permanent, what is worth retrying
# --------------------------------------------------------------------------- #

class _FailingBackend:
    """Raises on every call, like a network outage or a bad key."""

    def __init__(self, exc: Exception) -> None:
        self.name = "claude-opus-5"
        self.exc = exc
        self.calls = 0

    def complete(self, image_path, prompt, max_tokens):
        self.calls += 1
        raise self.exc

    def warmup(self) -> None:
        return None


class _RefusingBackend:
    """Returns a refusal, the way a safety classifier does: HTTP 200, no content."""

    def __init__(self) -> None:
        self.name = "claude-opus-5"
        self.calls = 0

    def complete(self, image_path, prompt, max_tokens):
        from melampus.backend import Completion

        self.calls += 1
        return Completion(text="", seconds=0.0, refused=True)

    def warmup(self) -> None:
        return None


def _prepare(tmp_path: Path, photo: Path, backend):
    from melampus.images import content_hash

    local = ResultCache(tmp_path / "local.jsonl")
    cloud = ResultCache(tmp_path / "cloud.jsonl")
    local.put(make_result(photo.name, abstain=True, content_hash=content_hash(photo)))
    config = load_config(
        run={"prompts_dir": str(REPO / "prompts"), "cache_path": str(tmp_path / "u.jsonl")},
        escalation={"enabled": True},
    )
    return local, cloud, Identifier(backend, config), config


def test_transient_failure_is_not_cached_and_is_retried(tmp_path, photo, monkeypatch):
    """The worst bug in the original: `identify()` returns rather than raises on
    failure, so caching every result wrote outages into the paid-work file and the
    next run skipped those frames forever. One 529 window burned the whole tail."""
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    backend = _FailingBackend(RuntimeError("529 overloaded_error"))
    local, cloud, identifier, config = _prepare(tmp_path, photo, backend)

    first = escalate([photo], local, cloud, identifier=identifier, config=config)
    assert first.selected == 1
    assert first.processed == 0, "a failed frame must not count as processed"
    assert first.errors == 1, "transient failure must be reported as an error"
    assert first.refused == 0, "an outage is not a refusal"

    from melampus.images import content_hash
    assert cloud.get(content_hash(photo)) is None, "failure was cached; frame is now stuck"

    # The API recovers; the same frame must be retried.
    good = ScriptedBackend([CLOUD_ROUTING, CLOUD_ID], name="claude-opus-5")
    identifier.backend = good
    second = escalate([photo], local, cloud, identifier=identifier, config=config)
    assert second.selected == 1, "frame was not retried after the outage cleared"
    assert second.processed == 1
    assert cloud.get(content_hash(photo)).identification.top().common_name == "Tricolored Heron"


def test_a_refusal_is_permanent_and_is_not_retried(tmp_path, photo, monkeypatch):
    """A refusal is a decision about the image. Re-asking spends money to be told
    no again, so unlike a transient failure it IS cached."""
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    backend = _RefusingBackend()
    local, cloud, identifier, config = _prepare(tmp_path, photo, backend)

    first = escalate([photo], local, cloud, identifier=identifier, config=config)
    assert first.refused == 1
    assert first.errors == 0, "a refusal is not an error"
    assert first.processed == 1

    from melampus.images import content_hash
    cached = cloud.get(content_hash(photo))
    assert cached is not None and cached.refused is True

    calls_after_first = backend.calls
    second = escalate([photo], local, cloud, identifier=identifier, config=config)
    assert second.selected == 0, "a refused frame was re-billed"
    assert backend.calls == calls_after_first


def test_a_refusal_does_not_trigger_the_corrective_retry(tmp_path, photo, monkeypatch):
    """`Completion.refused` existed but nothing consumed it, so every refusal paid
    for a second call guaranteed to be refused as well."""
    monkeypatch.setenv("MELAMPUS_ANTHROPIC_KEY", "sk-test")
    backend = _RefusingBackend()
    local, cloud, identifier, config = _prepare(tmp_path, photo, backend)

    escalate([photo], local, cloud, identifier=identifier, config=config)
    # One call for Stage A. No corrective retry, and no descent down the
    # downscale ladder — a refusal will not change at a smaller size.
    assert backend.calls == 1, f"refusal cost {backend.calls} calls; expected 1"


def test_both_reasons_are_recorded_when_a_frame_is_uncertain_and_out_of_range(cfg):
    """Returning early on confidence used to erase the range flag from provenance,
    losing exactly the signal that makes a frame interesting."""
    result = make_result(confidences=(0.55,))
    reason = should_escalate(result, config=cfg, range_flagged=True)
    assert reason is not None
    assert "confidence" in reason
    assert "range" in reason


def test_range_flags_are_keyed_on_content_not_filename(cfg):
    """Everything else in this module identifies a frame by content hash. Keying
    range flags on the filename means two IMG_0042 in different folders collide,
    and a rename silently unflags."""
    result = make_result("IMG_0042.jpg", confidences=(0.95,), content_hash="hash-xyz")

    by_name, _ = select_for_escalation([result], config=cfg,
                                       range_flagged={"IMG_0042.jpg"})
    assert by_name == [], "matched on filename; a rename would change the outcome"

    by_hash, _ = select_for_escalation([result], config=cfg, range_flagged={"hash-xyz"})
    assert len(by_hash) == 1


# --------------------------------------------------------------------------- #
# the Anthropic backend itself
# --------------------------------------------------------------------------- #

class _StubMessages:
    """Records what the SDK would have been asked to send."""

    def __init__(self, text: str = "{}", stop_reason: str = "end_turn") -> None:
        self.text = text
        self.stop_reason = stop_reason
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        block = type("Block", (), {"type": "text", "text": self.text})()
        usage = type("Usage", (), {"input_tokens": 1500, "output_tokens": 300})()
        return type("Message", (), {
            "content": [block],
            "stop_reason": self.stop_reason,
            "stop_details": None,
            "usage": usage,
        })()


class _StubClient:
    def __init__(self, messages: _StubMessages) -> None:
        self.messages = messages
        self.beta = type("Beta", (), {"messages": messages})()


def test_default_provider_is_claude(cfg):
    """Card #403: the Anthropic provider is named `claude`, the same word the
    engine setting uses, so there is one name for it."""
    assert cfg.provider == "claude"


def test_openai_key_resolution(monkeypatch):
    monkeypatch.delenv("MELAMPUS_ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "from-generic-env")
    monkeypatch.setenv("MELAMPUS_OPENAI_KEY", "from-melampus-env")

    openai = load_config(use_local=False, escalation={"provider": "openai"}).escalation
    assert resolve_api_key(openai) == "from-melampus-env"

    monkeypatch.delenv("MELAMPUS_OPENAI_KEY")
    assert resolve_api_key(openai) == "from-generic-env"

    # Provider keys must not cross over: an Anthropic key does not authorise OpenAI.
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-provider")
    assert resolve_api_key(openai) is None


def test_each_provider_gets_its_own_default_model():
    assert resolve_model(load_config(use_local=False).escalation) == "claude-opus-5"
    assert "gpt" in resolve_model(load_config(use_local=False, escalation={"provider": "openai"}).escalation)
    explicit = load_config(use_local=False, escalation={"provider": "openai", "model": "some-other"}).escalation
    assert resolve_model(explicit) == "some-other"


def test_unknown_provider_is_rejected_loudly():
    config = load_config(use_local=False, escalation={"provider": "gemini", "enabled": True})
    with pytest.raises(EscalationRefused, match="provider"):
        build_cloud_identifier(config, api_key="sk-test")


def test_cost_estimate_follows_configured_rates():
    """Providers price differently; a hardcoded rate would quietly mislead."""
    cheap = load_config(
        escalation={"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0}).escalation
    dear = load_config(
        escalation={"input_usd_per_mtok": 10.0, "output_usd_per_mtok": 20.0}).escalation
    assert estimate_cost_usd(10, dear) == pytest.approx(estimate_cost_usd(10, cheap) * 10)


def _sdk_installed(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


@pytest.mark.parametrize("provider,module", [("claude", "anthropic"), ("openai", "openai")])
def test_missing_sdk_fails_before_any_frame_is_selected(provider, module):
    """The backends import their SDK lazily, so without an eager check the failure
    surfaced once per frame, mid-run, after selection — and every frame then got a
    permanent error cached against it."""
    if _sdk_installed(module):
        pytest.skip(f"{module} is installed; nothing to assert about its absence")
    config = load_config(use_local=False, escalation={"enabled": True, "provider": provider})
    with pytest.raises(ImportError):
        build_cloud_identifier(config, api_key="sk-test")


@pytest.mark.skipif(not _sdk_installed("anthropic"), reason="anthropic SDK not installed")
def test_anthropic_provider_builds_an_anthropic_backend():
    identifier = build_cloud_identifier(
        load_config(use_local=False, escalation={"enabled": True}), api_key="sk-test")
    assert isinstance(identifier.backend, AnthropicBackend)


@pytest.mark.skipif(not _sdk_installed("openai"), reason="openai SDK not installed")
def test_openai_provider_builds_an_openai_backend():
    from melampus.backend import OpenAIBackend

    identifier = build_cloud_identifier(
        load_config(use_local=False, escalation={"enabled": True, "provider": "openai"}), api_key="sk-test")
    assert isinstance(identifier.backend, OpenAIBackend)


def test_cloud_identifier_overrides_the_routing_budget():
    """The local routing default is 200 tokens, sized for a runtime that does not
    think before answering. On a thinking-on model that is consumed before any JSON
    appears, so every frame would fail twice and be billed for both. This is the
    single defect that would have made the whole feature not work."""
    from melampus.config import MelampusConfig

    config = MelampusConfig()
    config.escalation.enabled = True
    # Build the cloud config the same way build_cloud_identifier does, without
    # needing a provider SDK installed.
    cloud = config.model_copy(deep=True)
    cloud.model.routing_max_tokens = config.escalation.routing_max_tokens
    assert config.model.routing_max_tokens == 200, "local default changed; retune this"
    assert cloud.model.routing_max_tokens >= 600, (
        "cloud routing budget too small for a model that thinks before answering"
    )


# --------------------------------------------------------------------------- #
# the OpenAI-compatible backend
# --------------------------------------------------------------------------- #

class _StubCompletions:
    def __init__(self, text: str = "{}", finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.requests: list[dict] = []
        self.reject_max_completion_tokens = False

    def create(self, **kwargs):
        if self.reject_max_completion_tokens and "max_completion_tokens" in kwargs:
            raise TypeError("Unsupported parameter: 'max_completion_tokens'")
        self.requests.append(kwargs)
        message = type("Msg", (), {"content": self.text})()
        choice = type("Choice", (), {"message": message, "finish_reason": self.finish_reason})()
        usage = type("Usage", (), {"prompt_tokens": 1500, "completion_tokens": 300})()
        return type("Response", (), {"choices": [choice], "usage": usage})()


class _StubOpenAIClient:
    def __init__(self, completions: _StubCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def test_openai_backend_requires_a_key():
    from melampus.backend import OpenAIBackend

    with pytest.raises(ValueError, match="API key"):
        OpenAIBackend(api_key=None)


def test_openai_backend_sends_a_data_uri_and_no_filename(tmp_path):
    from melampus.backend import OpenAIBackend

    staged = tmp_path / NEUTRAL_NAME
    Image.new("RGB", (64, 48), (10, 20, 30)).save(staged)

    stub = _StubCompletions(text='{"taxon": "bird", "confidence": 0.9, "reasoning": "x"}')
    backend = OpenAIBackend(api_key="sk-test", model="gpt-5",
                            client=_StubOpenAIClient(stub))

    completion = backend.complete(staged, "Identify this bird.", max_tokens=200)
    assert "bird" in completion.text

    sent = stub.requests[0]
    assert sent["model"] == "gpt-5"
    blocks = sent["messages"][0]["content"]
    image_block = next(b for b in blocks if b["type"] == "image_url")
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert NEUTRAL_NAME not in json.dumps(sent["messages"])
    assert str(staged) not in json.dumps(sent["messages"])


def test_openai_backend_falls_back_to_max_tokens(tmp_path):
    """Reasoning models take max_completion_tokens; older ones only take max_tokens.
    Learn which once, rather than failing every frame in the batch."""
    from melampus.backend import OpenAIBackend

    staged = tmp_path / NEUTRAL_NAME
    Image.new("RGB", (64, 48), (10, 20, 30)).save(staged)

    stub = _StubCompletions(text="{}")
    stub.reject_max_completion_tokens = True
    backend = OpenAIBackend(api_key="sk-test", client=_StubOpenAIClient(stub))

    backend.complete(staged, "Identify.", max_tokens=200)
    backend.complete(staged, "Identify.", max_tokens=200)

    assert len(stub.requests) == 2
    assert all("max_tokens" in r and "max_completion_tokens" not in r for r in stub.requests)


def test_openai_backend_reports_a_content_filter_as_a_refusal(tmp_path):
    from melampus.backend import OpenAIBackend

    staged = tmp_path / NEUTRAL_NAME
    Image.new("RGB", (64, 48), (10, 20, 30)).save(staged)

    # Partial text, not None: a filtered response can carry some content, and with
    # text=None this test passed even with the guard deleted, because the `or ""`
    # already produced an empty string.
    stub = _StubCompletions(text="partial answer before the filter fired",
                            finish_reason="content_filter")
    backend = OpenAIBackend(api_key="sk-test", client=_StubOpenAIClient(stub))

    completion = backend.complete(staged, "Identify.", max_tokens=200)
    assert completion.refused is True
    assert completion.text == "", "filtered content was passed through as an answer"


def test_anthropic_backend_requires_a_key():
    with pytest.raises(ValueError, match="API key"):
        AnthropicBackend(api_key=None)


def test_anthropic_backend_sends_base64_pixels_and_no_filename(tmp_path):
    staged = tmp_path / NEUTRAL_NAME
    Image.new("RGB", (64, 48), (10, 20, 30)).save(staged)

    stub = _StubMessages(text='{"taxon": "bird", "confidence": 0.9, "reasoning": "x"}')
    backend = AnthropicBackend(api_key="sk-test", client=_StubClient(stub))

    completion = backend.complete(staged, "Identify this bird.", max_tokens=200)
    assert "bird" in completion.text

    sent = stub.requests[0]
    assert sent["model"] == "claude-opus-5"
    blocks = sent["messages"][0]["content"]
    image_block = next(b for b in blocks if b["type"] == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/jpeg"
    # The wire format carries bytes, not a path — but assert it anyway, because the
    # whole accuracy claim rests on the model seeing no metadata.
    assert NEUTRAL_NAME not in json.dumps(sent["messages"])
    assert str(staged) not in json.dumps(sent["messages"])


def test_anthropic_request_carries_the_exact_parameters_the_api_needs(tmp_path):
    """The stub accepts any kwargs, so a misspelled key or a wrong beta string
    would pass green here and 400 in production. Assert the wire shape."""
    staged = tmp_path / NEUTRAL_NAME
    Image.new("RGB", (64, 48), (10, 20, 30)).save(staged)

    stub = _StubMessages(text="{}")
    backend = AnthropicBackend(api_key="sk-test", effort="high",
                               client=_StubClient(stub))
    backend.complete(staged, "Identify.", max_tokens=900)

    sent = stub.requests[0]
    assert sent["max_tokens"] == 900
    # effort lives inside output_config, not at the top level
    assert sent["output_config"] == {"effort": "high"}
    assert "effort" not in sent
    # The scalar `fallbacks: "default"` form pairs with the -07-01 beta; crossing
    # it with the array form's -06-01 header is a 400.
    assert sent["fallbacks"] == "default"
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]
    # Sampling params were removed on this model tier and are rejected outright.
    for banned in ("temperature", "top_p", "top_k", "thinking"):
        assert banned not in sent, f"{banned} is rejected on this model tier"


class _BetaRejectingMessages(_StubMessages):
    """Beta endpoint refuses the fallback parameter; the plain endpoint works."""

    def __init__(self, text: str = "{}", error: Exception | None = None) -> None:
        super().__init__(text)
        self.beta_calls = 0
        self.plain_calls = 0
        self.error = error or TypeError("unexpected keyword argument 'fallbacks'")

    def beta_create(self, **kwargs):
        self.beta_calls += 1
        raise self.error

    def create(self, **kwargs):
        self.plain_calls += 1
        return super().create(**kwargs)


class _SplitClient:
    def __init__(self, messages: _BetaRejectingMessages) -> None:
        self.messages = messages
        self.beta = type("Beta", (), {
            "messages": type("M", (), {"create": messages.beta_create})()
        })()


def test_anthropic_falls_back_to_the_plain_endpoint_and_latches(tmp_path):
    """An org without the fallback beta must still work — once, not per frame."""
    staged = tmp_path / NEUTRAL_NAME
    Image.new("RGB", (64, 48), (10, 20, 30)).save(staged)

    stub = _BetaRejectingMessages(text='{"taxon":"bird","confidence":0.9,"reasoning":"x"}')
    backend = AnthropicBackend(api_key="sk-test", client=_SplitClient(stub))

    assert "bird" in backend.complete(staged, "Identify.", 200).text
    assert "bird" in backend.complete(staged, "Identify.", 200).text

    assert stub.beta_calls == 1, "re-probed the unavailable beta on every frame"
    assert stub.plain_calls == 2


def test_an_unrelated_error_does_not_silently_disable_fallbacks(tmp_path):
    """The matcher used to accept any message containing 'beta', so one unrelated
    failure switched refusal recovery off for the whole batch and re-raised a
    misleading second error."""
    staged = tmp_path / NEUTRAL_NAME
    Image.new("RGB", (64, 48), (10, 20, 30)).save(staged)

    unrelated = RuntimeError("overloaded_error: the model is in beta capacity issues")
    unrelated.status_code = 529
    stub = _BetaRejectingMessages(error=unrelated)
    backend = AnthropicBackend(api_key="sk-test", client=_SplitClient(stub))

    with pytest.raises(RuntimeError, match="overloaded"):
        backend.complete(staged, "Identify.", 200)
    assert backend._use_fallbacks is True, "an unrelated error disabled refusal recovery"
    assert stub.plain_calls == 0, "masked the real error by retrying without fallbacks"


def test_anthropic_backend_reports_a_refusal_instead_of_crashing(tmp_path):
    """Opus 5 returns HTTP 200 with stop_reason 'refusal'. Reading content[0]
    unconditionally would raise; the frame should just come back unprocessed."""
    staged = tmp_path / NEUTRAL_NAME
    Image.new("RGB", (64, 48), (10, 20, 30)).save(staged)

    stub = _StubMessages(text="", stop_reason="refusal")
    stub.create = lambda **kwargs: type("Message", (), {  # noqa: E731
        "content": [],
        "stop_reason": "refusal",
        "stop_details": type("D", (), {"category": "cyber", "explanation": "no"})(),
        "usage": type("U", (), {"input_tokens": 10, "output_tokens": 0})(),
    })()
    backend = AnthropicBackend(api_key="sk-test", client=_StubClient(stub))

    completion = backend.complete(staged, "Identify this bird.", max_tokens=200)
    assert completion.text == ""
    assert completion.refused is True
