"""The model call, behind one swappable interface (CLAUDE.md §3).

Everything above this module talks to `VLMBackend`. Swapping MLX for vllm-mlx, or for
an Anthropic-compatible endpoint during the Phase-3 cloud-escalation path, means adding
one class here and changing nothing else.
"""

from __future__ import annotations

import base64
import contextlib
import errno
import functools
import http.client
import itertools
import json
import os
import select
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


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
    #: The most of what the model's side wrote that an error message carries:
    #: it lands in the frame's error record (identify.py), so in the cache
    #: and --json-out. Ollama's own errors are one line; a proxy's error
    #: page, or a line of a CLI's usage text, is cut here.
    MAX_ERROR_BYTES = 1 << 10

    @abstractmethod
    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion: ...

    def warmup(self) -> None:  # pragma: no cover - optional
        return None

    @classmethod
    def plain(cls, text: str) -> str:
        """Text the model's side wrote (a server, a program), as it may
        reach the frame's error record, the log and the terminal: one line
        of at most MAX_ERROR_BYTES printable characters. An escape
        sequence in it would move the cursor, recolour the terminal or
        erase a line, and a line break would fake a line of the log.
        Whatever is not printable (str.isprintable: the C0 and C1
        controls, line and paragraph breaks, the unassigned) becomes a
        space, and runs of whitespace collapse to one, so what is left is
        words. The one rule for every message that carries those words:
        Ollama's error body, a status line http.client could not parse,
        and a command's stderr."""
        words = " ".join("".join(c if c.isprintable() else " " for c in text).split())
        return words[: cls.MAX_ERROR_BYTES]


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


def _hang_up(line, event: threading.Event) -> None:
    """The deadline, from its timer, or the cancellation, from its watcher:
    `event` is the one that says which, the _Deadline's `expired` or its
    `cancelled`. `line` is the _Deadline holding the
    exchange's socket as `sock`, handed over the moment `_Noted` makes it
    and again, for https, as the wrapped socket before the handshake. Not
    close(): the response being read holds the socket's file object, and
    socket.close() waits for that to go before it really closes, so the
    blocked read would read on. shutdown(SHUT_RDWR) ends the stream now.
    And `event`, because http.client takes end-of-stream as the end of
    the headers: a status line that arrived before the trickle would still
    parse as a 200, and the caller must know the deadline finished the
    response, not the server. No socket yet means the caller is still
    connecting: the socket timeout bounds that, and `_Deadline.on` hangs
    the socket up as soon as it is given, since a timer that fired before
    the socket existed had nothing to hang up. A socket the main thread
    already closed (the probe's `finally`, or urllib once the headers are
    in) raises OSError, which is suppressed."""
    event.set()
    sock = line.sock
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)


class _Deadline:
    """A wall-clock bound on one HTTP exchange, as a context manager. A
    socket timeout bounds each operation, not the exchange, so a server
    trickling a byte at a time, each within the timeout, could hold the
    caller for as long as it liked. Inside the block one thread, the timer,
    waits for the deadline `seconds` away; when it passes, `_hang_up` ends
    the stream on `sock` and sets `expired`, which the caller reads after
    the exchange, since whatever arrived by then is not the server's
    answer. `sock` is the socket the exchange is on, given by `on()` once
    there is one: a deadline that fired while the caller was still
    connecting found nothing to hang up, so `on()` hangs up then, and the
    reads after a late handshake are not left bounded per byte only.
    Leaving the block ends the timer; `again()` moves the deadline, for a
    stream's next line, and starts nothing: the pull runs under a signal
    handler that raises wherever the main thread is, and an exception
    raised inside `Thread.start()` leaves the thread module's tables and
    locks half-changed (review round 12, backend.py:419), so the block
    starts its threads once, on entry, not once a line.

    `cancel`, when given, is a predicate looked at every WATCH seconds from
    a thread of the timer's kind, for a stream whose caller may need it
    ended while a read blocks (the pull's cancel marker, download.py:
    looked for between lines, it waited on a stalled Ollama for the whole
    timeout). Its turning true hangs the socket up as the timer does and
    sets `cancelled`, which the caller reads as it reads `expired`: the
    read ends within WATCH whatever the server is writing, and whatever
    shape the ended read takes is the cancellation, not that shape's
    error. Leaving the block ends the watcher too."""

    WATCH = 0.25

    def __init__(self, seconds: float, cancel: Callable[[], bool] | None = None) -> None:
        self.seconds = seconds
        self.sock: socket.socket | None = None
        self.expired = threading.Event()
        self.cancelled = threading.Event()
        self._over = threading.Event()
        self._timer = threading.Thread(target=self._counting, daemon=True)
        self._watcher = threading.Thread(target=self._watching, args=[cancel], daemon=True) if cancel else None

    def _counting(self) -> None:
        while not self._over.wait(max(0.0, self._until - time.monotonic())):
            if time.monotonic() >= self._until:
                _hang_up(self, self.expired)
                return

    def _watching(self, cancel: Callable[[], bool]) -> None:
        while not self._over.wait(self.WATCH):
            if cancel():
                _hang_up(self, self.cancelled)
                return

    def __enter__(self) -> _Deadline:
        self.again()
        self._timer.start()
        if self._watcher is not None:
            self._watcher.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._over.set()
        self._timer.join()
        if self._watcher is not None:
            self._watcher.join()

    def again(self) -> None:
        """The bound from now: the deadline the timer waits for is `seconds`
        from now, an assignment the timer reads when the one it waits on
        comes round. `__enter__` arms the block with it, before the timer
        starts, and OllamaBackend.stream calls it before each line: a
        stream has no one exchange to bound, since a model pull runs as
        long as the model is large, so each line gets the bound an
        exchange gets. A deadline that already fired stays fired: its
        socket is hung up, and the caller reads that."""
        self._until = time.monotonic() + self.seconds

    def on(self, sock: socket.socket | None) -> None:
        self.sock = sock
        for event in (self.expired, self.cancelled):
            if event.is_set():
                _hang_up(self, event)


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
    request's deadline (OllamaBackend._bounded puts it on the request). The
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
    frames and the model pull's stream alike (card #409): straight to the address,
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


def ollama_request(
    address: str, path: str, body: dict | None = None, *, method: str = "POST"
) -> urllib.request.Request:
    """The one request to an Ollama endpoint: `path` under `address` (the
    address as providers.ollama_url hands it, its trailing slash already
    dropped, so the path appends cleanly), `body` sent as JSON with its
    content type when given. The backend's frame, the model pull and the
    list and the delete of download.py (card #409) build theirs here."""
    return urllib.request.Request(
        f"{address}{path}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )


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
        return ollama_request(self.url, self.ENDPOINT, body)

    def send(self, request: urllib.request.Request) -> bytes:
        """The reply's bytes, within `timeout` of wall-clock time from
        connecting to the last byte read, error bodies included: a
        _Deadline hangs up the socket when the time is up, and whatever the
        exchange then looks like (a body cut short, a status line that
        never finished, a reset) is the timeout, not that shape's error.
        The one way a request reaches Ollama: a frame's here, the list's
        and the delete's from download.py; the pull's stream through
        `stream`, beside."""
        with self._bounded(request):
            raw = self._exchange(request)
        return self._within_bound(raw)

    def stream(self, request: urllib.request.Request, *,
               cancel: Callable[[], bool] | None = None) -> Iterator[bytes]:
        """The reply's lines as the server writes them, for the model pull
        (card #409, download.pull_model), each within `timeout` of wall-clock
        time: the stream has no one exchange to bound, since a pull runs as
        long as the model is large, so each line gets what `send` gives an
        exchange, the deadline armed again for it. One line is one of
        Ollama's objects, a few hundred bytes, written at once; a listener
        writing one a byte at a time within the socket timeout would
        otherwise hold the pull, and the cancel marker read between lines,
        for as long as it liked. At most MAX_REPLY_BYTES of a line is read,
        a longer one refused by name; the stream is closed on leaving the
        loop, however it is left, which is how Ollama learns to stop. Every
        failure is named as `send` names one (`_naming`): the status with
        Ollama's words, nothing answering, the timeout, a reply that is not
        HTTP. `cancel`, when given, is the caller's predicate (the pull's
        cancel marker), looked at every _Deadline.WATCH seconds while a
        read blocks: its turning true ends the stream, cleanly, within that
        period, whatever the server is writing; the caller, whose predicate
        it is, knows why the stream ended."""
        with self._bounded(request, cancel) as deadline, self._naming(), \
                self._urlopen(request, timeout=self.timeout) as response:
            while True:
                deadline.again()
                line = response.readline(self.MAX_REPLY_BYTES + 1)
                if not line or deadline.expired.is_set() or deadline.cancelled.is_set():
                    return
                yield self._within_bound(line)

    @contextlib.contextmanager
    def _bounded(self, request: urllib.request.Request,
                 cancel: Callable[[], bool] | None = None) -> Iterator[_Deadline]:
        """The block within `timeout` of wall-clock time: a _Deadline on the
        request, for the connection _Bounded opens for it, that hangs up the
        socket when the time is up. Whatever the block then looks like (a
        body cut short, a status line that never finished, a reset, or a
        return with what arrived) is the timeout, not that shape's error.
        With `cancel` (a stream's), the deadline hangs the socket up for
        the predicate too, and whatever the block then looks like is the
        cancellation: the block ends, and nothing is raised for its shape.
        The cancellation is read before the timeout: the two hang-ups are
        the same hang-up, and both may have fired (a cancel in the last
        WATCH before a line's deadline), so once the block has timed out
        the predicate decides which it was (`_cancelled`), asked once more
        for a marker the watcher's tick has not yet seen."""
        with _Deadline(self.timeout, cancel) as deadline:
            request.deadline = deadline
            try:
                yield deadline
            except Exception as exc:
                if self._cancelled(deadline, cancel, exc):
                    return
                if deadline.expired.is_set():
                    raise self._timed_out() from exc
                raise
        if self._cancelled(deadline, cancel):
            return
        if deadline.expired.is_set():
            raise self._timed_out()

    @staticmethod
    def _cancelled(deadline: _Deadline, cancel: Callable[[], bool] | None,
                   exc: Exception | None = None) -> bool:
        """Whether the block ended for the cancellation: the watcher saw the
        predicate true and hung up, or the block timed out and the
        predicate, asked now, is true. The watcher looks every WATCH
        seconds, so a cancel asked in the last WATCH before the deadline
        is one it may not have seen; and the block times out two ways at
        the same moment, the timer's hang-up (`expired`) or the socket's
        own timeout on the read, `exc` as `_naming` names it, since the
        one `timeout` arms both. Never with no `cancel`: `send` has no
        cancellation."""
        if deadline.cancelled.is_set():
            return True
        timed_out = deadline.expired.is_set() or isinstance(exc, TimeoutError)
        return timed_out and cancel is not None and cancel()

    def _exchange(self, request: urllib.request.Request) -> bytes:
        """One request and what came back, every failure a plain error naming
        the address or the status."""
        with self._naming(), self._urlopen(request, timeout=self.timeout) as response:
            return response.read(self.MAX_REPLY_BYTES + 1)

    @contextlib.contextmanager
    def _naming(self) -> Iterator[None]:
        """Every failure of the block a plain error naming the address or the
        status: what urllib raises for an HTTP status, for nothing answering
        and for the socket timeout, and what it lets through unwrapped: an
        http.client protocol error, or the raw socket error of a connection
        that ended while the reply was being read (a reset, a broken pipe:
        Ollama killed, or a listener hanging up)."""
        try:
            yield
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Ollama answered {exc.code}: {self.error_text(exc)}"
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
                f"Ollama's reply from {self.url} was not HTTP: {self.plain(str(exc))}"
            ) from exc
        except OSError as exc:
            # The raw socket error of a connection that ended mid-reply
            # (ConnectionResetError, BrokenPipeError: a ConnectionError, so
            # otherwise taken by a caller's clause for the not-running
            # failure above, which is the only ConnectionError raised here).
            raise RuntimeError(f"the connection to Ollama at {self.url} ended: {exc}") from exc

    def _within_bound(self, raw: bytes) -> bytes:
        """`raw` (a reply, or one line of a stream) when it is at most
        MAX_REPLY_BYTES; the refusal naming the bound past it, the one
        for `send` and `stream` alike."""
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
    def error_text(cls, exc: urllib.error.HTTPError) -> str:
        """Ollama's own words when the body is its {"error": ...} object,
        else the body as it came (a proxy's HTML, say), else the status
        line's reason; at most MAX_ERROR_BYTES of it read, and only its
        printable characters (`plain`), since all three are the server's
        to write."""
        body = exc.read(cls.MAX_ERROR_BYTES).decode("utf-8", "replace").strip()
        try:
            error = json.loads(body).get("error")
        except (json.JSONDecodeError, AttributeError):
            error = None
        return cls.plain(f"{error}" if error else body or exc.reason)

    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion:
        request = self._request(image_path, prompt, max_tokens)
        started = time.perf_counter()
        raw = self.send(request)
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


class CommandFailed(RuntimeError):
    """The command exited non-zero: the engine is broken (not signed in, wrong
    flags), not the frame. The CLI surfaces it at exit 3 like the other
    backend failures instead of recording it on every frame in turn."""


class CommandBackend(VLMBackend):
    """An installed command-line program behind the same interface (card
    #420): one run per completion, the reply on stdout. Claude Code and Codex
    CLI bill to a subscription rather than per call, so a command that takes
    an image and a prompt is vision with no API key; the templates for those
    two are cards #421 and #422. This class knows no program: `command` is
    the config's argv template, one element per argument, with `{image}` and
    `{prompt}` placeholders replaced wherever they sit. An argv list, never a
    shell: the prompt is one argument however many spaces, quotes or newlines
    it holds, and nothing is quoted or escaped.

    `executable` is what shutil.which resolved the template's first element
    to (providers.build_primary_backend does that before any image is read,
    so a missing program is refused up front, and so is one that resolves to
    a `.cmd` or `.bat` file, which Windows would run through cmd.exe): it
    replaces the bare name in the argv, so what was checked is what runs.
    The same factory refuses the engine when the process that started
    melampus ignores SIGCHLD (`SIG_IGN` is inherited across exec), because
    the kernel would then reap the program the moment it exits; that
    refusal is what `_stop_tree`'s precondition rests on: the command stays
    unreaped until `_wait` reaps it, so the pid its tree is stopped by is
    still its own and never a number given since to someone else's process.
    The child gets the parent's environment as it is, so the
    program finds its own sign-in; nothing is added to it and no secret
    crosses the command line. As with every backend, the image is the staged,
    metadata-free file and only its path travels. `max_tokens` has no
    placeholder: the program's own limits apply.

    stdout goes through the same JSON extraction and schema validation as
    every other backend's text (identify.py); nothing here parses
    candidates. stderr is kept for error messages only. Both are read as
    they come, at most MAX_OUTPUT_BYTES each, and a program that streams
    past that is stopped and the frame refused by name. `timeout` is the
    one ceiling on a run's time; past it the command and every process it
    started are stopped (OWN_GROUP) before the frame's TimeoutError is
    raised. The command's exit ends its answer: whatever it started is
    stopped then too, so a helper it leaves holding stdout or stderr is
    stopped rather than waited on, and what was read is the reply.
    """

    #: How much of stderr an error message carries: enough to say what went
    #: wrong, not a CLI's whole usage text.
    STDERR_LINES = 3
    #: The most of stdout, and of stderr, that is kept: a JSON reply of
    #: candidates is kilobytes, and a CLI's progress chatter over a whole
    #: run is far less than this, so a program that streams megabytes is
    #: broken, not answering. It is stopped at the ceiling rather than read
    #: into memory until the timeout (the reviewer's probe held gigabytes
    #: within seconds), and the frame records the refusal by name.
    MAX_OUTPUT_BYTES = 4 << 20
    #: One read from a pipe.
    CHUNK_BYTES = 1 << 16
    @property
    def OWN_GROUP(self) -> dict:
        """The Popen arguments that give the command its own session (POSIX:
        setsid, so its process group id is its pid and os.killpg reaches
        every worker it forked) or its own process group (Windows, where
        `taskkill /T` walks the tree), so a stop reaches everything it
        started and not just the first process: a CLI that hands the work
        to a worker would otherwise leave that worker running, one per
        timed-out frame, while the batch goes on. Read at each start, so
        the Windows shape can be asserted from any platform."""
        if sys.platform == "win32":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def __init__(
        self,
        command: list[str],
        *,
        executable: str | None = None,
        timeout: float = 180.0,
        run: Callable | None = None,
    ) -> None:
        self.command = list(command)
        # The template, so a changed flag is a changed run fingerprint and the
        # cache cannot re-serve the old template's answers. shlex.join keeps
        # the argument boundaries: `--label 'bird --mode precise'` (one
        # argument) is not `--label bird --mode precise` (three), and they
        # run the program differently, so they must not share a fingerprint.
        self.name = shlex.join(self.command)
        self.executable = executable or self.command[0]
        self.timeout = timeout
        # Shaped like subprocess.Popen(argv, **kwargs): the tests hand in a
        # fake at this edge, the way the other backends take a client.
        self._run = run or subprocess.Popen

    @property
    def program(self) -> str:
        """The name the user knows the program by, for messages."""
        return self.command[0]

    def _argv(self, image_path: Path, prompt: str) -> list[str]:
        expanded = [
            argument.replace("{image}", str(image_path)).replace("{prompt}", prompt)
            for argument in self.command
        ]
        return [self.executable, *expanded[1:]]

    def _stderr_lines(self, stderr: str) -> str:
        """The first STDERR_LINES lines the program wrote that are words
        once read through `plain`, joined with " / ": they land in the
        frame's error record, the log and the terminal, so an escape
        sequence in them would clear the screen or recolour it, and a
        control would fake a line of the log. A line that is only
        controls is not a line, so it spends none of the STDERR_LINES;
        `islice` stops the reading at the cap."""
        words = (self.plain(line) for line in stderr.splitlines())
        return " / ".join(itertools.islice(filter(None, words), self.STDERR_LINES))

    def _stop_tree(self, pid: int) -> None:
        """Stop the process tree the command with `pid` heads (OWN_GROUP):
        every process in its group on POSIX, the tree under it on Windows.
        Called only while the command is unreaped (_wait reaps it last), so
        the pid is still the command's own and its group's, never a
        number given since to a process of someone else's. A tree already
        gone is nothing to do; so is one holding nothing but the command's
        own exited process, which macOS answers with EPERM."""
        if sys.platform == "win32":
            # taskkill by its absolute path: run by bare name, CreateProcess
            # would look in the current directory before System32, and a
            # taskkill.exe planted there would run with this process's
            # rights the first time a command timed out.
            taskkill = os.environ.get("SystemRoot", r"C:\Windows") + r"\System32\taskkill.exe"
            subprocess.run(
                [taskkill, "/T", "/F", "/PID", str(pid)],
                stdin=subprocess.DEVNULL, capture_output=True, check=False,
            )
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)

    def _stop(self, process) -> None:
        """Stop the command and everything it started."""
        self._stop_tree(process.pid)
        process.kill()

    def _drain(self, process, name: str, sink: bytearray, overflowed: list[str]) -> None:
        """Read the pipe `name` of `process` to its end into `sink`, keeping
        at most MAX_OUTPUT_BYTES. Past that the program is a runaway: its
        name goes on `overflowed`, which _wait watches for and stops the
        tree on (every stop is made by the thread that reaps, before it
        reaps), and the rest is read and dropped so the pipe still ends."""
        stream = getattr(process, name)
        while chunk := stream.read1(self.CHUNK_BYTES):
            if overflowed:
                continue
            if len(sink) + len(chunk) > self.MAX_OUTPUT_BYTES:
                overflowed.append(name)
                continue
            sink += chunk

    def _exited(self, process, within: float) -> bool:
        """Whether the command has exited, seen within `within` seconds and
        without reaping it: an exited process that is not reaped keeps its
        pid, and so its group's id, until _wait reaps it last, so a tree
        stopped by that pid is the command's and never a process given the
        number since. macOS's Python has no waitid, and kqueue is how it
        sees an exit: a NOTE_EXIT event is the exit, seen during the wait,
        and a registration refused with ESRCH is the command already
        exited before the look (XNU does not see an exited process; for
        this process's own unreaped child that can only mean it has
        exited); waitid with WNOWAIT elsewhere on POSIX, a look before the
        step and one at its end, so an exit already there is seen at once
        and one during the step at its end; on Windows the Popen handle
        keeps the pid reserved, so wait itself is safe there and the order
        does not matter."""
        if sys.platform == "win32":
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=within)
                return True
            return False
        if hasattr(select, "kqueue"):
            exit_event = select.kevent(
                process.pid, select.KQ_FILTER_PROC,
                select.KQ_EV_ADD | select.KQ_EV_ONESHOT, select.KQ_NOTE_EXIT,
            )
            queue = select.kqueue()
            try:
                events = queue.control([exit_event], 1, within)
            finally:
                queue.close()
            if not events:
                return False
            (event,) = events
            if event.flags & select.KQ_EV_ERROR:
                # The registration was refused, and with one event asked
                # for the refusal comes back as an event. ESRCH is the
                # command already exited (before this look, or between two
                # looks); any other error is the error it is, not an exit.
                if event.data == errno.ESRCH:
                    return True
                raise OSError(event.data, os.strerror(event.data))
            return True  # NOTE_EXIT: the exit, seen during the wait.
        # Look first, so a command already exited is seen at once as the
        # other two branches see it; sleep the step and look once more only
        # when it has not, so an exit during the step is seen at its end.
        look = os.WEXITED | os.WNOWAIT | os.WNOHANG
        if os.waitid(os.P_PID, process.pid, look) is not None:
            return True
        time.sleep(within)
        return os.waitid(os.P_PID, process.pid, look) is not None

    def _wait(self, process, readers: list[threading.Thread], overflowed: list[str]) -> bool:
        """Whether the command exited within `timeout`. Its exit ends its
        answer: the moment it is seen (_exited, without reaping) the tree
        it heads is stopped, so a worker it left holding stdout or stderr
        dies and the pipe ends, the readers are given a short bound to
        reach those ends, and what they read is the reply. Past the
        timeout, at the ceiling (`overflowed`, which the readers raise and
        this loop sees within a step), or on any other interruption
        (Ctrl+C), the tree and the command are stopped the same way, as
        subprocess.run kills its child: the command does not outlive the
        run that started it, and neither does anything it started. Every
        stop is made before the command is reaped, which is the last thing
        done here, so the pid the tree is stopped by is still the
        command's own; on the interruption path it is stopped, not reaped,
        and the interrupt propagates."""
        deadline = time.monotonic() + self.timeout
        step = 0.0005
        try:
            while not (in_time := self._exited(process, step)) and not overflowed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                step = min(step * 2, remaining, 0.05)
        except BaseException:
            self._stop(process)
            raise
        if in_time:
            self._stop_tree(process.pid)
        else:
            self._stop(process)
        ends = time.monotonic() + 5.0
        for reader in readers:
            # The pipes end when the tree is gone; one still open is held by
            # something that left the group (a double-forked daemon) and is
            # left to it rather than waited on.
            reader.join(timeout=max(0.0, ends - time.monotonic()))
        for reader, name in zip(readers, ("stdout", "stderr")):
            # A pipe read to its end is closed here, not by the garbage
            # collector; one whose reader is still in read1 is left to it,
            # since closing a BufferedReader from another thread blocks
            # until that read returns.
            if not reader.is_alive():
                getattr(process, name).close()
        process.wait()
        return in_time

    def complete(self, image_path: Path, prompt: str, max_tokens: int) -> Completion:
        argv = self._argv(image_path, prompt)
        started = time.perf_counter()
        try:
            process = self._run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                **self.OWN_GROUP,
            )
        except OSError as exc:
            raise RuntimeError(f"{self.program} could not be run: {exc}") from exc
        sinks = {"stdout": bytearray(), "stderr": bytearray()}
        overflowed: list[str] = []
        readers = [
            threading.Thread(target=self._drain, args=(process, name, sink, overflowed), daemon=True)
            for name, sink in sinks.items()
        ]
        for reader in readers:
            reader.start()
        in_time = self._wait(process, readers, overflowed)
        elapsed = time.perf_counter() - started
        stdout, stderr = (sinks[name].decode("utf-8", "replace") for name in ("stdout", "stderr"))

        if overflowed:
            raise RuntimeError(
                f"{self.program} wrote more than {self.MAX_OUTPUT_BYTES} bytes on "
                f"{overflowed[0]} and was stopped: a reply is kilobytes"
            )
        if not in_time:
            raise TimeoutError(
                f"{self.program} did not answer within {self.timeout:g}s; "
                "raise [model] timeout_seconds if it needs longer"
            )
        if process.returncode != 0:
            said = self._stderr_lines(stderr)
            raise CommandFailed(
                f"{self.program} exited {process.returncode}"
                + (f": {said}" if said else " with nothing on stderr")
            )
        if not stdout.strip():
            said = self._stderr_lines(stderr)
            raise RuntimeError(
                f"{self.program} printed nothing on stdout"
                + (f": {said}" if said else "")
            )
        return Completion(text=stdout, seconds=elapsed)


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
