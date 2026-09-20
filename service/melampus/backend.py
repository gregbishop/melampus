"""The model call, behind one swappable interface (CLAUDE.md §3).

Everything above this module talks to `VLMBackend`. Swapping MLX for vllm-mlx, or for
an Anthropic-compatible endpoint during the Phase-3 cloud-escalation path, means adding
one class here and changing nothing else.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import http.client
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

        from .download import load_lock

        # `load` fetches what the cache does not hold of the repo, so the load
        # takes the repo's lock the download and the removal take: a removal
        # in the meantime is refused, one already running is waited for.
        with load_lock(self.repo):
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
    """The deadline, from its timer. `line` is the _Deadline holding the
    exchange's socket as `sock`, handed over the moment `_Noted` makes it
    and again, for https, as the wrapped socket before the handshake. Not
    close(): the response being read holds the socket's file object, and
    socket.close() waits for that to go before it really closes, so the
    blocked read would read on. shutdown(SHUT_RDWR) ends the stream now.
    And `expired`, because http.client takes end-of-stream as the end of
    the headers: a status line that arrived before the trickle would still
    parse as a 200, and the caller must know the deadline finished the
    response, not the server. No socket yet means the caller is still
    connecting: the socket timeout bounds that, and `_Deadline.on` hangs
    the socket up as soon as it is given, since a timer that fired before
    the socket existed had nothing to hang up. A socket the main thread
    already closed (the probe's `finally`, or urllib once the headers are
    in) raises OSError, which is suppressed."""
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


class _Noted:
    """Mixed into http.client's connection classes, for the backend (urllib
    opens them) and the probe alike: the socket goes to a _Deadline the
    moment it exists. The moment it exists, because connect() is the TCP
    connection and, for https, then the TLS handshake, each bounded by the
    socket timeout on its own, so a connection that took most of the
    budget and then a handshake that stalls would hold the caller a second
    whole timeout before a deadline given the socket afterwards could
    touch it; http.client makes the socket through the `_create_connection`
    attribute it sets on itself (the seam its own tests use), so that is
    where the socket is caught. And it is the deadline that holds the
    socket, not the connection: urllib's do_open forgets the socket on
    the connection once the headers are in (the response's file holds it
    from then on), while the deadline outlives both. What comes before the
    socket, resolving a hostname, has no timeout to give it: the resolver's
    own applies, and `ollama_url` is an IP literal unless a user names a
    host."""

    def __init__(self, *args, deadline: _Deadline, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.deadline = deadline
        self._create_connection = self._connect

    def _connect(self, address, timeout, source_address) -> socket.socket:
        sock = socket.create_connection(address, timeout, source_address)
        self.deadline.on(sock)
        return sock


class _NotedHTTP(_Noted, http.client.HTTPConnection):
    pass


class _NotedHTTPS(_Noted, http.client.HTTPSConnection):
    """HTTPSConnection.connect wraps the socket and runs the handshake in
    one call, and the wrap detaches the socket the deadline holds (its
    descriptor moves to the new SSLSocket), so a hang-up during the
    handshake would touch nothing: a connection landing just before the
    deadline bought a stalled handshake a whole socket timeout more. So
    connect() here wraps without the handshake, gives the deadline the
    wrapped socket, then shakes hands, with the verifying context the
    connection has: the one `_Bounded` passed in for the backend, or the
    one HTTPSConnection makes for itself when given none, for the probe.
    The server name is the host: nothing here tunnels through a proxy, so
    there is no other."""

    def connect(self) -> None:  # noqa: D102 - http.client's
        http.client.HTTPConnection.connect(self)
        self.sock = self._context.wrap_socket(
            self.sock, server_hostname=self.host, do_handshake_on_connect=False
        )
        self.deadline.on(self.sock)
        self.sock.do_handshake()


class _Bounded(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    """urllib's HTTP and HTTPS handlers, opening _Noted connections for the
    request's deadline (OllamaBackend._send puts it on the request). The
    https side keeps HTTPSHandler's default context, the verifying one it
    builds when given none, and passes it to _NotedHTTPS as HTTPSHandler
    would to HTTPSConnection."""

    def http_open(self, req):  # noqa: D102 - urllib's
        return self.do_open(functools.partial(_NotedHTTP, deadline=req.deadline), req)

    def https_open(self, req):  # noqa: D102 - urllib's
        return self.do_open(
            functools.partial(_NotedHTTPS, deadline=req.deadline), req, context=self._context
        )


def ollama_opener(*handlers: urllib.request.BaseHandler) -> Callable:
    """urllib's `open` for every request to the Ollama address, the backend's
    frames and the model pull alike (card #409): straight to the address,
    never past it. Not urlopen itself: its opener honours http_proxy and
    the system proxy settings, which would send the bytes off the machine
    and let the proxy's answer stand in for Ollama's (the probe in
    providers.ollama_answers keeps off the proxy for the same reason);
    ProxyHandler({}) consults neither. And it follows a 3xx, so whatever
    listens on the port when Ollama does not could point a request at
    another host and have that host's reply stand in for Ollama's;
    _StayPut follows nothing, as the probe follows nothing. `handlers` add
    to those two: the backend passes _Bounded, whose connections hand their
    socket to the request's deadline."""
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _StayPut(), *handlers
    ).open


def ollama_not_running(url: str, reason: object) -> str:
    """The one message for a request Ollama did not answer at `url`: the
    backend's mid-run failure and the model pull (card #409) say the same
    thing about the same condition."""
    return (
        f"no Ollama server answering at {url} ({reason}); "
        "start Ollama, or set [model] ollama_url to where it listens"
    )


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
        # urlopen itself: ollama_opener says why (no proxy, no redirect). And
        # urlopen's `timeout` is the socket's, per operation, so a server
        # trickling bytes could hold a frame past `timeout_seconds`; _Bounded
        # opens connections that hand their socket to the request's deadline.
        self._urlopen = client or ollama_opener(_Bounded())

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
        """The reply's bytes, within `timeout` of wall-clock time from
        connecting to the last byte read, error bodies included: a
        _Deadline hangs up the socket when the time is up, and whatever the
        exchange then looks like (a body cut short, a status line that
        never finished, a reset) is the timeout, not that shape's error."""
        with _Deadline(self.timeout) as deadline:
            request.deadline = deadline
            try:
                raw = self._exchange(request)
            except Exception as exc:
                if deadline.expired.is_set():
                    raise self._timed_out() from exc
                raise
        if deadline.expired.is_set():
            raise self._timed_out()
        if len(raw) > self.MAX_REPLY_BYTES:
            raise RuntimeError(
                f"Ollama's reply from {self.url} ran past {self.MAX_REPLY_BYTES} bytes"
            )
        return raw

    def _exchange(self, request: urllib.request.Request) -> bytes:
        """One request and what came back, every failure a plain error naming
        the address or the status."""
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                return response.read(self.MAX_REPLY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama answered {exc.code}: {self._error_text(exc)}"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise self._timed_out() from exc
            raise ConnectionError(ollama_not_running(self.url, exc.reason)) from exc
        except TimeoutError as exc:
            raise self._timed_out() from exc
        except http.client.HTTPException as exc:
            # What urllib lets through unwrapped: a status line that is not
            # HTTP (BadStatusLine carries it as sent), a server hanging up
            # before one, a body cut short, a line past http.client's limit.
            raise RuntimeError(
                f"Ollama's reply from {self.url} was not HTTP: {self._plain(str(exc))}"
            ) from exc

    def _timed_out(self) -> TimeoutError:
        return TimeoutError(
            f"Ollama at {self.url} did not answer within {self.timeout:g}s; "
            f"{self.model} may still be loading, or raise [model] timeout_seconds"
        )

    @classmethod
    def _plain(cls, text: str) -> str:
        """Text the server wrote, as it may reach the frame's error record,
        the log and the terminal: one line of at most MAX_ERROR_BYTES
        printable characters. An escape sequence in it would move the
        cursor, recolour the terminal or erase a line, and a line break
        would fake a line of the log. Whatever is not printable
        (str.isprintable: the C0 and C1 controls, line and paragraph
        breaks, the unassigned) becomes a space, and runs of whitespace
        collapse to one, so what is left is words. The one rule for every
        message that carries the server's words: an error body, and a
        status line http.client could not parse."""
        words = " ".join("".join(c if c.isprintable() else " " for c in text).split())
        return words[: cls.MAX_ERROR_BYTES]

    @classmethod
    def _error_text(cls, exc: urllib.error.HTTPError) -> str:
        """Ollama's own words when the body is its {"error": ...} object,
        else the body as it came (a proxy's HTML, say), else the status
        line's reason; at most MAX_ERROR_BYTES of it read, and only its
        printable characters (`_plain`), since all three are the server's
        to write."""
        body = exc.read(cls.MAX_ERROR_BYTES).decode("utf-8", "replace").strip()
        try:
            error = json.loads(body).get("error")
        except (json.JSONDecodeError, AttributeError):
            error = None
        return cls._plain(f"{error}" if error else body or exc.reason)

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
