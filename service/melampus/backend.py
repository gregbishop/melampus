"""The model call, behind one swappable interface (CLAUDE.md §3).

Everything above this module talks to `VLMBackend`. Swapping MLX for vllm-mlx, or for
an Anthropic-compatible endpoint during the Phase-3 cloud-escalation path, means adding
one class here and changing nothing else.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Completion:
    text: str
    seconds: float
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    # Set when a safety classifier declined the request rather than the model failing.
    # Distinguishing the two matters: a refusal is permanent for that image and
    # prompt, so retrying it is only spending money to be told no again.
    refused: bool = False


class VLMBackend(ABC):
    """Takes an image path and a prompt; returns text. Nothing else crosses this line."""

    name: str

    @abstractmethod
    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion: ...

    def warmup(self) -> None:  # pragma: no cover - optional
        return None


class MLXBackend(VLMBackend):
    """mlx-vlm on Apple Silicon. Loads lazily so `--help` doesn't pull 18 GB of weights."""

    def __init__(self, repo: str, temperature: float = 0.0) -> None:
        self.name = repo
        self.repo = repo
        self.temperature = temperature
        self._model = None
        self._processor = None
        self._config = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self._model, self._processor = load(self.repo)
        self._config = load_config(self.repo)

    def warmup(self) -> None:
        self._ensure_loaded()

    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion:
        self._ensure_loaded()
        from mlx_vlm import apply_chat_template, generate

        formatted = apply_chat_template(self._processor, self._config, prompt, num_images=1)
        started = time.perf_counter()
        result = generate(
            self._model,
            self._processor,
            formatted,
            image=[str(image_path)],
            max_tokens=max_tokens,
            temperature=self.temperature,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        text = getattr(result, "text", None)
        if text is None:
            text = str(result)
        return Completion(
            text=text,
            seconds=elapsed,
            prompt_tokens=getattr(result, "prompt_tokens", None),
            generated_tokens=getattr(result, "generation_tokens", None),
        )


class AnthropicBackend(VLMBackend):
    """The Claude API behind the same interface as the local model (CLAUDE.md §6.6).

    Identical request shape to `MLXBackend`: one image, one prompt, text back. That
    is what makes escalation a backend swap rather than a second pipeline — the same
    prompt files, the same schema validation, the same corrective retry.

    The image is sent as base64 bytes. Nothing else goes with it: no filename, no
    EXIF, no path. The caller hands over a staged file from `images.staged_pixels`,
    which is already stripped and named neutrally, and only its bytes are read.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str = "claude-opus-5",
        *,
        effort: str = "high",
        timeout: float = 180.0,
        client: object | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Cloud escalation needs an Anthropic API key. Set MELAMPUS_ANTHROPIC_KEY "
                "in your environment, or put it in melampus.local.toml (git-ignored). "
                "Never commit it."
            )
        self.name = model
        self.model = model
        self.effort = effort
        self._api_key = api_key
        self._timeout = timeout
        self._client = client
        # Server-side refusal fallbacks are a beta. If this organisation does not
        # have it, the first request says so and every later one skips it rather
        # than re-learning the same rejection on every image.
        self._use_fallbacks = True

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # imported lazily: the local path must not need it

            self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
        return self._client

    @staticmethod
    def _is_beta_rejection(exc: Exception) -> bool:
        """Only a genuine "this organisation cannot use the fallback beta".

        This used to match any message containing "beta" or "unexpected keyword",
        which is loose enough to catch failures having nothing to do with
        fallbacks — and because the result is latched for the process, one
        unrelated error silently disabled refusal recovery for the whole batch
        while re-raising a misleading second error. Both halves were bad.
        """
        status = getattr(exc, "status_code", None)
        if status is not None and status not in (400, 403, 404):
            return False
        text = str(exc).lower()
        if "fallback" in text:
            return True
        # An SDK too old to know the parameter names raises TypeError locally.
        return isinstance(exc, TypeError) and (
            "fallbacks" in text or "betas" in text
        )

    def _send(self, blocks: list[dict], max_tokens: int):
        client = self._ensure_client()
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            # Thinking is on by default on this model tier, which is what we want:
            # these are the frames the local model could not resolve.
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": blocks}],
        }
        if self._use_fallbacks:
            try:
                return client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **params,
                )
            except Exception as exc:  # noqa: BLE001 - degrade, never abort the batch
                if not self._is_beta_rejection(exc):
                    raise
                self._use_fallbacks = False
        return client.messages.create(**params)

    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion:
        import base64

        data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")
        blocks = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
            },
            {"type": "text", "text": prompt},
        ]

        started = time.perf_counter()
        message = self._send(blocks, max_tokens)
        elapsed = time.perf_counter() - started

        usage = getattr(message, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None)
        generated = getattr(usage, "output_tokens", None)

        # Check stop_reason before touching content: on a refusal the content list is
        # empty, and indexing it would turn a routine decline into a crashed batch.
        if getattr(message, "stop_reason", None) == "refusal":
            return Completion(
                text="", seconds=elapsed, prompt_tokens=prompt_tokens,
                generated_tokens=generated, refused=True,
            )

        text = "".join(
            getattr(block, "text", "")
            for block in getattr(message, "content", [])
            if getattr(block, "type", "") == "text"
        )
        return Completion(
            text=text, seconds=elapsed,
            prompt_tokens=prompt_tokens, generated_tokens=generated,
        )


class OpenAIBackend(VLMBackend):
    """An OpenAI-compatible chat-completions endpoint behind the same interface.

    Chosen over the newer responses API deliberately: the chat-completions shape
    with a `data:` image URI is what OpenRouter, LM Studio, vLLM and most proxies
    also speak, so `base_url` turns this one class into a general escape hatch
    rather than a single vendor's client.

    As with every other backend, the only thing that travels is the bytes of a
    staged, metadata-free image.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str = "gpt-5",
        *,
        base_url: str | None = None,
        timeout: float = 180.0,
        client: object | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Cloud escalation needs an OpenAI API key. Set MELAMPUS_OPENAI_KEY in "
                "your environment, or put it in melampus.local.toml (git-ignored). "
                "Never commit it."
            )
        self.name = model
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client = client
        # Reasoning models take max_completion_tokens; older ones only accept
        # max_tokens. Learn which once and remember, instead of failing on every
        # frame or paying a rejected request per image. Recorded against the model
        # it was learned for: the latch is one-way, so without this a mid-process
        # model switch would keep sending the wrong parameter with no recovery.
        self._token_param = "max_completion_tokens"
        self._token_param_model = model

    def _ensure_client(self):
        if self._client is None:
            import openai  # imported lazily: the local path must not need it

            kwargs = {"api_key": self._api_key, "timeout": self._timeout}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _is_token_param_rejection(exc: Exception) -> bool:
        text = str(exc).lower()
        return "max_completion_tokens" in text and (
            "unsupported" in text or "unexpected" in text or "unrecognized" in text
        )

    def _send(self, blocks: list[dict], max_tokens: int):
        client = self._ensure_client()
        if self._token_param_model != self.model:
            # Different model: what we learned no longer applies.
            self._token_param = "max_completion_tokens"
            self._token_param_model = self.model
        params = {"model": self.model, "messages": [{"role": "user", "content": blocks}]}
        try:
            return client.chat.completions.create(
                **params, **{self._token_param: max_tokens}
            )
        except Exception as exc:  # noqa: BLE001 - narrowed immediately below
            if self._token_param == "max_tokens" or not self._is_token_param_rejection(exc):
                raise
            self._token_param = "max_tokens"
            return client.chat.completions.create(**params, max_tokens=max_tokens)

    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion:
        import base64

        data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")
        blocks = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
        ]

        started = time.perf_counter()
        response = self._send(blocks, max_tokens)
        elapsed = time.perf_counter() - started

        usage = getattr(response, "usage", None)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        finish = getattr(choice, "finish_reason", None) if choice else None
        text = getattr(getattr(choice, "message", None), "content", None) or ""

        return Completion(
            text="" if finish == "content_filter" else text,
            seconds=elapsed,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            generated_tokens=getattr(usage, "completion_tokens", None),
            refused=finish == "content_filter",
        )


class ScriptedBackend(VLMBackend):
    """Deterministic canned responses. Lets the pipeline be tested without weights."""

    def __init__(self, responses: list[str], name: str = "scripted") -> None:
        self.name = name
        self._responses = list(responses)
        self.calls: list[tuple[Path, str]] = []

    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion:
        self.calls.append((image_path, prompt))
        text = self._responses.pop(0) if self._responses else "{}"
        return Completion(text=text, seconds=0.0)
