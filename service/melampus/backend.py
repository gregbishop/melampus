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
