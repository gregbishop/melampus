"""The model call, behind one swappable interface (CLAUDE.md §3).

Everything above this module talks to `VLMBackend`. Swapping MLX for vllm-mlx, or for
an Anthropic-compatible endpoint during the Phase-3 cloud-escalation path, means adding
one class here and changing nothing else.
"""

from __future__ import annotations

import base64
import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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


def _image_as_base64(image_path: Path) -> str:
    """The staged image's bytes, base64 for a JSON body: the one thing every
    HTTP backend sends of an image (no path, no filename)."""
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")


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
                "The Claude backend needs an API key. Set MELAMPUS_ANTHROPIC_KEY "
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
        data = _image_as_base64(image_path)
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
                "The OpenAI backend needs an API key. Set MELAMPUS_OPENAI_KEY in "
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
        data = _image_as_base64(image_path)
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


def _hang_up(line, expired: threading.Event) -> None:
    """The deadline, from its timer. `line` is whatever holds the exchange's
    socket under the name `sock`: the probe's HTTPConnection, or a
    _Deadline. Not connection.close(): the response
    being read holds the socket's file object, and socket.close() waits for
    that to go before it really closes, so the blocked read would read on.
    shutdown(SHUT_RDWR) ends the stream now. And `expired`, because
    http.client takes end-of-stream as the end of the headers: a status line
    that arrived before the trickle would still parse as a 200, and the
    probe must know the deadline finished the response, not the server. No
    socket yet means the probe is still connecting: the socket timeout bounds
    that, and the probe checks `expired` once connected, since a timer that
    fired before the socket existed had nothing to hang up and the reads
    after a late handshake would otherwise be bounded per byte only. The
    socket is read once: the main thread's close() sets it to None at any
    moment, and a socket it already closed raises OSError, which is
    suppressed; None between two reads would not be."""
    expired.set()
    sock = line.sock
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)


class _Deadline:
    """A wall-clock bound on one HTTP exchange, as a context manager. A
    socket timeout bounds each operation, not the exchange, so a server
    trickling a byte at a time, each within the timeout, could hold the
    caller for as long as it liked. Inside the block a timer counts
    `seconds`; when it fires, `_hang_up` ends the stream on `sock` and sets
    `expired`, which the caller reads after the exchange, since whatever
    arrived by then is not the server's answer. `sock` is the socket the
    exchange is on, given by `on()` once there is one: a deadline that
    fired while the caller was still connecting found nothing to hang up,
    so `on()` hangs up then, and the reads after a late handshake are not
    left bounded per byte only. Leaving the block cancels the timer."""

    def __init__(self, seconds: float) -> None:
        self.sock: socket.socket | None = None
        self.expired = threading.Event()
        self._timer = threading.Timer(seconds, _hang_up, [self, self.expired])

    def __enter__(self) -> _Deadline:
        self._timer.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._timer.cancel()

    def on(self, sock: socket.socket | None) -> None:
        self.sock = sock
        if self.expired.is_set():
            _hang_up(self, self.expired)


class _StayPut(urllib.request.HTTPRedirectHandler):
    """Follows no redirect: a 3xx from the configured address is an answer
    from the wrong place, surfaced as the status it is."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


class OllamaBackend(VLMBackend):
    """A local Ollama server behind the same interface (card #406): local
    inference on Windows and Linux, and on Macs that prefer it, through the
    engine most users already have.

    One request per completion, from Ollama's docs/api.md § Generate a chat
    completion: POST {url}/api/chat with a JSON body of `model`, one user
    message carrying the prompt as `content` and the image as one base64
    string in `images`, `stream` false so a single response object comes
    back, and `options` of `num_predict` (docs/modelfile.mdx: the maximum
    number of tokens to predict) and `temperature`. The reply's text is
    `message.content`; the counts are `prompt_eval_count` and `eval_count`.
    The text goes through the same JSON extraction and schema validation as
    every other backend's (identify.py): nothing here parses candidates.

    The standard library speaks it: the cloud backends each use their vendor's
    SDK, so there is no shared HTTP client to reuse, and a dependency for four
    JSON fields would be a new path for nothing. As with every backend, only
    the bytes of the staged, metadata-free image travel: no path, no filename.

    Whether a server is there at all is the factory's question
    (providers.build_primary_backend probes before building this), so the
    not-running message can name the install and exit 3 before any image is
    read. Here a failure mid-run maps to a plain error naming the address or
    the status, which identify() records on that frame while the batch goes on.
    """

    ENDPOINT = "/api/chat"
    #: The most of a reply that is read: one object, the text of at most
    #: `num_predict` tokens and a dozen counters, so a megabyte is not an
    #: answer, and a server that keeps sending does not fill memory.
    MAX_REPLY_BYTES = 1 << 20
    #: The most of a non-200's body that is read: it lands in the frame's
    #: error record (identify.py), so in the cache and --json-out. Ollama's
    #: own errors are one line; a proxy's error page is cut here.
    MAX_ERROR_BYTES = 1 << 10

    def __init__(
        self,
        model: str,
        url: str,
        *,
        temperature: float = 0.0,
        timeout: float = 180.0,
        client: Callable | None = None,
    ) -> None:
        self.name = model
        self.model = model
        # The address as the factory hands it: providers.ollama_url has
        # already dropped the trailing slash, so ENDPOINT appends cleanly.
        self.url = url
        self.temperature = temperature
        self.timeout = timeout
        # Shaped like urllib.request.urlopen(request, timeout=...): the tests hand
        # in a fake at this edge, the way the cloud backends take a client. Not
        # urlopen itself: its opener honours http_proxy and the system proxy
        # settings, which would send every frame's bytes off the machine and
        # let the proxy's answer stand in for the model's (the probe in
        # providers.ollama_answers keeps off the proxy for the same reason);
        # ProxyHandler({}) consults neither. And it follows a 3xx, so
        # whatever listens on the port when Ollama does not could point a
        # frame at another host and have that host's reply stand in for the
        # model's; _StayPut follows nothing, as the probe follows nothing.
        self._urlopen = client or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _StayPut()
        ).open

    def _request(self, image_path: Path, prompt: str, max_tokens: int) -> urllib.request.Request:
        image = _image_as_base64(image_path)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [image]}],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": self.temperature},
        }
        return urllib.request.Request(
            f"{self.url}{self.ENDPOINT}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _send(self, request: urllib.request.Request) -> bytes:
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                raw = response.read(self.MAX_REPLY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama answered {exc.code}: {self._error_text(exc)}"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise self._timed_out() from exc
            raise ConnectionError(
                f"no Ollama server answering at {self.url} ({exc.reason}); "
                "start Ollama, or set [model] ollama_url to where it listens"
            ) from exc
        except TimeoutError as exc:
            raise self._timed_out() from exc
        if len(raw) > self.MAX_REPLY_BYTES:
            raise RuntimeError(
                f"Ollama's reply from {self.url} ran past {self.MAX_REPLY_BYTES} bytes"
            )
        return raw

    def _timed_out(self) -> TimeoutError:
        return TimeoutError(
            f"Ollama at {self.url} did not answer within {self.timeout:g}s; "
            f"{self.model} may still be loading, or raise [model] timeout_seconds"
        )

    @classmethod
    def _error_text(cls, exc: urllib.error.HTTPError) -> str:
        """Ollama's own words when the body is its {"error": ...} object,
        else the body as it came (a proxy's HTML, say), else the status line;
        at most MAX_ERROR_BYTES of it."""
        body = exc.read(cls.MAX_ERROR_BYTES).decode("utf-8", "replace").strip()
        try:
            error = json.loads(body).get("error")
        except (json.JSONDecodeError, AttributeError):
            error = None
        return error or body or exc.reason

    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion:
        request = self._request(image_path, prompt, max_tokens)
        started = time.perf_counter()
        raw = self._send(request)
        elapsed = time.perf_counter() - started
        try:
            reply = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama's reply from {self.url} was not JSON: {raw[:120]!r}"
            ) from exc
        message = (reply.get("message") or {}) if isinstance(reply, dict) else None
        if not isinstance(message, dict):
            raise RuntimeError(
                f"Ollama's reply from {self.url} was not a JSON object with a "
                f"message object: {raw[:120]!r}"
            )
        return Completion(
            text=message.get("content") or "",
            seconds=elapsed,
            prompt_tokens=reply.get("prompt_eval_count"),
            generated_tokens=reply.get("eval_count"),
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
