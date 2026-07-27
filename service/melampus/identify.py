"""Two-stage identification with defensive parsing.

Stage A routes to a coarse taxon; Stage B runs a taxon-specialised prompt. Both stages
validate against the schema, retry once with a corrective message, and then give up
gracefully — an unprocessed photo is a far better outcome than a wrong keyword
(CLAUDE.md §4.4: a wrong keyword is worse than no keyword).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from .backend import VLMBackend
from .config import MelampusConfig
from .images import staged_pixels
from .prompts import ROUTING_PROMPT, PromptLibrary
from .schema import Identification, ImageResult, Taxon, TaxonRouting

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

CORRECTIVE = (
    "Your previous reply could not be parsed. Reply with a SINGLE valid JSON object "
    "and nothing else — no prose, no markdown fences, no trailing commentary. "
    "Error was: {error}"
)


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a model reply.

    Local VLMs wrap JSON in fences, prefix it with commentary, or append a summary
    sentence. All three are recoverable without another round-trip.
    """
    if not text:
        return None

    fenced = _FENCE.search(text)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)

    for blob in candidates:
        start = blob.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for index in range(start, len(blob)):
                char = blob[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(blob[start : index + 1])
                        except json.JSONDecodeError:
                            break
                        if isinstance(parsed, dict):
                            return parsed
                        break
            start = blob.find("{", start + 1)
    return None


class Identifier:
    def __init__(
        self,
        backend: VLMBackend,
        config: MelampusConfig,
        prompts: PromptLibrary | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.prompts = prompts or PromptLibrary(config.run.prompts_dir)
        self._fingerprint: str | None = None

    @property
    def fingerprint(self) -> str:
        """Everything that can change an answer without the image changing."""
        if self._fingerprint is None:
            import hashlib

            parts = "|".join([
                self.backend.name,
                self.prompts.fingerprint(),
                str(self.config.image.max_edge),
                str(self.config.model.max_tokens),
                str(self.config.model.temperature),
            ])
            self._fingerprint = hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]
        return self._fingerprint

    def _ask(self, image: Path, prompt: str, max_tokens: int, model_cls):
        """Call the model, validate, retry once with a corrective message."""
        attempt = 0
        current = prompt
        last_error = "no response"
        seconds = 0.0

        while attempt <= self.config.run.max_retries:
            completion = self.backend.complete(image, current, max_tokens)
            seconds += completion.seconds
            payload = extract_json(completion.text)
            if payload is None:
                last_error = "no JSON object found in reply"
            else:
                try:
                    return model_cls.model_validate(payload), attempt, seconds, None
                except ValidationError as exc:
                    last_error = "; ".join(
                        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                        for e in exc.errors()[:4]
                    )
            attempt += 1
            current = f"{prompt}\n\n{CORRECTIVE.format(error=last_error)}"

        return None, attempt, seconds, last_error

    def identify(self, path: Path) -> ImageResult:
        content = ""
        try:
            from .images import content_hash

            content = content_hash(path)
        except OSError as exc:
            return ImageResult(
                file=path.name, content_hash="", status="error",
                error=f"unreadable: {exc}", model=self.backend.name,
            )

        # Try progressively smaller images. See ImageConfig.max_edge: the runtime
        # returns an empty generation once the prompt grows past ~2.1k tokens, and
        # vision tokens are the dominant term, so shrinking the image is the lever.
        edges = [self.config.image.max_edge, *self.config.image.fallback_edges]
        last: ImageResult | None = None
        for edge in edges:
            try:
                with staged_pixels(path, edge, self.config.image.jpeg_quality) as staged:
                    result = self._identify_staged(staged, path.name, content)
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort a batch
                return ImageResult(
                    file=path.name, content_hash=content, status="error",
                    error=f"{type(exc).__name__}: {exc}", model=self.backend.name,
                    image_max_edge=edge,
                )
            result.image_max_edge = edge
            result.run_fingerprint = self.fingerprint
            if result.status == "ok":
                return result
            last = result

        assert last is not None
        return last

    def _identify_staged(self, staged: Path, display_name: str, content: str) -> ImageResult:
        total = 0.0
        retries = 0

        routing, tries, secs, err = self._ask(
            staged,
            self.prompts.render(ROUTING_PROMPT),
            self.config.model.routing_max_tokens,
            TaxonRouting,
        )
        total += secs
        retries += tries
        if routing is None:
            return ImageResult(
                file=display_name, content_hash=content, status="unprocessed",
                error=f"taxon routing failed: {err}", retries=retries,
                seconds=total, model=self.backend.name,
            )

        # No organism: Stage B would only invent one. Stop here.
        if routing.taxon is Taxon.NONE:
            return ImageResult(
                file=display_name, content_hash=content, status="ok",
                taxon_routing=routing,
                identification=Identification(
                    taxon=Taxon.NONE, candidates=[], diagnostic_features_visible=False,
                    abstain=True, abstain_reason="no organism present in frame",
                ),
                retries=retries, seconds=total, model=self.backend.name,
            )

        identification, tries, secs, err = self._ask(
            staged,
            self.prompts.prompt_for_taxon(routing.taxon.value),
            self.config.model.max_tokens,
            Identification,
        )
        total += secs
        retries += tries
        if identification is None:
            return ImageResult(
                file=display_name, content_hash=content, status="unprocessed",
                taxon_routing=routing, error=f"identification failed: {err}",
                retries=retries, seconds=total, model=self.backend.name,
            )

        # A model that abstains but still lists candidates is contradicting itself;
        # trust the abstention, since bias toward abstaining is the point.
        if identification.abstain:
            identification.candidates = []

        return ImageResult(
            file=display_name, content_hash=content, status="ok",
            taxon_routing=routing, identification=identification,
            retries=retries, seconds=total, model=self.backend.name,
        )
