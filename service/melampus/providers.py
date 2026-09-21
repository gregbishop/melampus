"""Cloud provider registry, and the primary-backend factory.

Two places build a cloud backend: escalation (the second-opinion tail on a Mac)
and the primary pipeline on machines with no local runtime (CLAUDE.md §3 made
the backend a seam; Windows is why the seam is now also a setting). Key lookup,
default model names and provider validation must not fork between them, so they
live here and both callers import them.
"""

from __future__ import annotations

import json
import platform
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import SecretStr

from .backend import CommandFailed, VLMBackend, _Deadline, _NotedHTTP, _NotedHTTPS, stderr_lines
from .config import MelampusConfig

#: Where each provider's key is looked for, in order, when the config has none.
#: Keys never cross providers: an Anthropic key must not silently authorise a
#: request to OpenAI, or the "which cloud am I using" question has no answer.
#: The providers carry the engine names the user chooses between (card #403):
#: `claude` is the Anthropic backend everywhere a user names it; the key
#: variables keep the vendor's name.
KEY_VARIABLES = {
    "claude": ("MELAMPUS_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("MELAMPUS_OPENAI_KEY", "OPENAI_API_KEY"),
}

#: Starting points only. Vision model names move faster than this file does —
#: check the provider's current listing and override in config when they age.
DEFAULT_MODELS = {
    "claude": "claude-opus-5",
    "openai": "gpt-5",
}

#: The fake the unit tests run against, reachable from the CLI so the shipped
#: executable can be smoke-tested on a machine with no weights (card #399). It
#: answers nothing useful; it is here to prove the pipeline around it runs.
SCRIPTED = "scripted"

#: The local Ollama server (card #403 named it, card #406 built it): the local
#: engine on Windows and Linux, and on Macs that prefer it. Refused below, the
#: way mlx is refused off Apple Silicon, when no server answers.
OLLAMA = "ollama"

#: The engines the user chooses between, in the owner's order, then the fake.
#: `detect_engines` tries them in this order for a default (card #404): the
#: first that can run on this machine.
BACKEND_CHOICES = ("mlx", OLLAMA, "openai", "claude", SCRIPTED)

#: Where the local Ollama server listens. Ollama's docs/faq.mdx: "Ollama binds
#: 127.0.0.1 port 11434 by default." One constant: the default of the
#: `[model] ollama_url` setting (card #406), which is unset until a user
#: names another address, so the probe and the backend share it.
OLLAMA_URL = "http://127.0.0.1:11434"

#: How long the probe waits for the local server. Loopback answers in
#: milliseconds or not at all; a second is a firewall's silence, not Ollama's.
OLLAMA_PROBE_SECONDS = 1.0

#: Where to get Ollama when nothing answers at OLLAMA_URL.
OLLAMA_INSTALL = "https://ollama.com/download"

#: An installed command-line program driven per frame (card #420): a
#: subscription CLI is vision with no API key. Selected by `[model] backend`
#: or --backend, not offered by the plugin's picker until card #423 teaches
#: detection about it, so it is not in BACKEND_CHOICES.
COMMAND = "command"

#: What a `command` may not resolve to: Windows launches a batch file through
#: cmd.exe regardless of what subprocess is told (Python's subprocess docs,
#: Security Considerations), and cmd.exe would parse the prompt, newlines,
#: quotes and braces included, instead of passing it as one argument. An
#: npm-installed CLI is such a shim; its real entry is the fix.
BATCH_SUFFIXES = (".cmd", ".bat")

#: Claude Code as an engine (card #421): the command seam configured for
#: one program, in the owner's words. Not a class: `claude-code` resolves to
#: a CommandBackend on CLAUDE_CODE_COMMAND, or on `[model] command` when the
#: user sets one. Runs bill to the Claude subscription Claude Code is signed
#: in to, never to an API key here.
CLAUDE_CODE = "claude-code"

#: The program, as shutil.which looks for it: `claude` on PATH.
CLAUDE_CODE_PROGRAM = "claude"

#: The one copy of the template. Every flag is from `claude --help` (2.1.277)
#: and code.claude.com/docs/en/headless: `-p` prints one reply and exits;
#: `--output-format json` puts the reply in the result object's `result`
#: field (claude_code_reply unwraps it); `--tools Read` leaves Claude Code
#: only the tool that reads files, which returns "PNG, JPG, and other image
#: formats ... as visual content that Claude can see" (tools-reference);
#: `--allowedTools Read` pre-approves that tool everywhere, so the staged
#: image in its temporary folder, outside any working directory, is read
#: without a permission prompt; `--permission-prompts none` denies anything
#: else that would wait for a person; `--no-session-persistence` writes no
#: transcript per frame; `--strict-mcp-config` connects no MCP server;
#: `--setting-sources user` loads no project or local settings from wherever
#: melampus was launched. The prompt is the positional argument, last: the
#: staged image's path for the Read tool, then the pipeline's prompt in full,
#: on one line up to the placeholder: the template is printed in the
#: `loading` line and in every message that names the program, so the
#: config refuses an element with a line break in it (card #420), and the
#: prompt's own line breaks arrive through the placeholder, not the template.
#: Not `--bare`: bare mode never reads the subscription login (headless docs:
#: "bare mode doesn't use your subscription login").
CLAUDE_CODE_COMMAND = [
    CLAUDE_CODE_PROGRAM, "-p", "--output-format", "json", "--tools", "Read",
    "--allowedTools", "Read", "--permission-prompts", "none", "--no-session-persistence",
    "--strict-mcp-config", "--setting-sources", "user",
    "The photograph is the file {image}. Read it with the Read tool, then answer this "
    "about it: {prompt}",
]

#: Where to get Claude Code when nothing on PATH is called `claude`.
CLAUDE_CODE_INSTALL = "https://code.claude.com/docs/en/setup"

#: How to sign in from a shell (`claude auth login --help`: "Sign in to your
#: Anthropic account"; `--claudeai`, the default, is the subscription). The
#: CLI's own not-signed-in result says "Please run /login", the interactive
#: session's command, so the refusal names this one instead.
CLAUDE_CODE_SIGN_IN = "claude auth login"

#: The documented, cheap sign-in check (cli-reference: "Show authentication
#: status as JSON ... Exits with code 0 if logged in, 1 if not"): no model
#: call, so detection and the up-front refusal spend nothing.
CLAUDE_CODE_STATUS = ("auth", "status", "--json")

#: How long the status check may take. A Node CLI answers it in a fraction
#: of a second (measured: 0.1 s); ten seconds is a broken install, and
#: --detect-engines must never hang the settings dialog.
CLAUDE_CODE_PROBE_SECONDS = 10.0

#: The backends that run on this machine and bill nobody per call.
LOCAL_BACKENDS = ("mlx", OLLAMA, COMMAND, CLAUDE_CODE, SCRIPTED)


class BackendUnavailable(RuntimeError):
    """This machine cannot run the configured backend; the message says what to do."""


def on_apple_silicon() -> bool:
    """The pyproject marker for mlx-vlm, as a predicate: the one place the
    runtime check, the build script and the tests' skips ask whether MLX
    exists here. Both halves, or an Intel Mac passes the OS check and then
    dies on a raw ModuleNotFoundError at warmup instead of the message."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _refusal(reason: str, *, works_here: tuple[str, ...]) -> BackendUnavailable:
    """One shape for every refusal: what is wrong, what works here, how to switch."""
    return BackendUnavailable(
        f"{reason} The backends that work on this machine are: "
        f"{', '.join(works_here)}. Set [model] backend in the config (with the "
        "matching API key for a cloud provider), or pass --backend. See "
        "readme.md § Windows."
    )


def ollama_url(configured: str | None = None) -> str:
    """The address Ollama is looked for at: `[model] ollama_url` when set,
    else OLLAMA_URL. Trailing slash dropped so the endpoints append cleanly."""
    return (configured or OLLAMA_URL).rstrip("/")


def ollama_answers(url: str | None = None) -> bool:
    """Whether an Ollama server answers at `url` (default OLLAMA_URL):
    GET /api/version (Ollama's docs/api.md § Version) within
    OLLAMA_PROBE_SECONDS, status 200. Connection refused, a timeout, a
    non-200: unavailable. Never raises; a probe reports. The address is
    read the way the backend reads it for every frame (OllamaBackend
    builds `{url}/api/chat` and hands it to urllib): the endpoint goes on
    the end of the address as typed, so a path in front of it (a reverse
    proxy's `/ollama`) stays; the scheme picks the connection, the
    backend's own _Noted ones, https spoken as TLS with the certificate
    verified (http.client's default context, as urllib's), so an https
    address is never asked in the clear and never on port 80; the host and
    port are the address's own. Read
    any other way, the probe would refuse a server every frame would reach,
    or find one no frame would. Straight to the
    address, never through a proxy: urlopen honours http_proxy and the
    system proxy settings, which would send a loopback probe off the machine
    and let the proxy's answer stand in for Ollama's; http.client consults
    neither. And never past the address:
    http.client follows no redirect, and a 3xx is a non-200, so whatever
    listens on the port when Ollama does not cannot point the probe at another
    host and have that host's 200 stand in for Ollama's. And never past the
    deadline: the socket timeout bounds each operation, not the probe, so
    a listener trickling headers a byte at a time, or a slow connection
    and then a handshake that stalls, could hold detection for as long as
    it liked; a _Deadline, holding the socket from the moment the
    connection makes it, hangs up at OLLAMA_PROBE_SECONDS, and whatever
    was read by then, the probe reports unavailable."""
    with _Deadline(OLLAMA_PROBE_SECONDS) as deadline:
        try:
            address = urlsplit(f"{ollama_url(url)}/api/version")
            connect = {"http": _NotedHTTP, "https": _NotedHTTPS}
            connection = connect[address.scheme](
                address.hostname, address.port, timeout=OLLAMA_PROBE_SECONDS, deadline=deadline
            )
        except Exception:  # noqa: BLE001 - an address that cannot be asked (no scheme, no host, a port out of range) is one nobody answers at
            return False
        try:
            connection.request("GET", address.path)
            # A deadline that fired before the socket existed hung it up as
            # soon as it did; asked anyway, the answer is not the server's.
            if deadline.expired.is_set():
                return False
            answered = connection.getresponse().status == 200
        except Exception:  # noqa: BLE001 - every failure means the same thing: not here
            return False
        finally:
            connection.close()
    return answered and not deadline.expired.is_set()


@dataclass(frozen=True, slots=True)
class EngineVerdict:
    """Whether one engine can run on this machine, and why or why not, in the
    words a user sees: the reason is what makes an unavailable engine a
    greyed-out choice rather than a mystery (card #404)."""

    engine: str
    available: bool
    reason: str
    #: Where the program that decides the verdict was found, when one does
    #: (claude-code: what shutil.which resolved `claude` to), so the factory
    #: runs what detection checked and never resolves it again. None for
    #: the engines no program decides.
    executable: str | None = None


def _key_required(engine: str) -> str:
    specific, generic = KEY_VARIABLES[engine]
    return f"API key required: set {specific} (or {generic})"


def claude_code_verdict(program: str | None = None) -> EngineVerdict:
    """Whether Claude Code can be the engine here (card #421): `program`
    (CLAUDE_CODE_PROGRAM unless a template names another) must be on PATH,
    and its status check must say signed in. Never raises; a verdict
    reports. The reasons are the words the user sees: not installed with
    where to get it, not signed in with the command that signs in, a check
    that did not answer or failed some other way (in the CLI's own words),
    or available and billing to the subscription."""
    program = program or CLAUDE_CODE_PROGRAM
    executable = shutil.which(program)
    if executable is None:
        return EngineVerdict(
            CLAUDE_CODE, False,
            f"Claude Code is not installed: nothing on PATH is called '{program}'; "
            f"install it from {CLAUDE_CODE_INSTALL}, then sign in with `{CLAUDE_CODE_SIGN_IN}`",
        )
    try:
        status = subprocess.run(
            [executable, *CLAUDE_CODE_STATUS],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, timeout=CLAUDE_CODE_PROBE_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return EngineVerdict(
            CLAUDE_CODE, False,
            f"`{program} {' '.join(CLAUDE_CODE_STATUS)}` did not answer within "
            f"{CLAUDE_CODE_PROBE_SECONDS:g}s",
        )
    except OSError as exc:
        return EngineVerdict(CLAUDE_CODE, False, f"'{program}' could not be run: {exc}")
    try:
        account = json.loads(status.stdout)
    except ValueError:
        account = {}
    if not isinstance(account, dict):
        account = {}
    if status.returncode != 0:
        # Not signed in is what the status object says (`loggedIn` false) or,
        # without one, the documented exit alone: "Exits with code 0 if
        # logged in, 1 if not" (cli-reference), nothing on stderr. Any other
        # failure (an older CLI with no `auth` subcommand, a usage error, a
        # crash) is reported in the CLI's own words, since signing in would
        # not help.
        said = stderr_lines(status.stderr)
        if account.get("loggedIn") is False or (status.returncode == 1 and not said):
            return EngineVerdict(
                CLAUDE_CODE, False,
                f"Claude Code is installed but not signed in; run `{CLAUDE_CODE_SIGN_IN}`",
            )
        return EngineVerdict(
            CLAUDE_CODE, False,
            f"`{program} {' '.join(CLAUDE_CODE_STATUS)}` exited {status.returncode}"
            + (f": {said}" if said else " with nothing on stderr"),
        )
    signed_in_as = ", ".join(
        str(account[key]) for key in ("authMethod", "subscriptionType") if account.get(key)
    )
    return EngineVerdict(
        CLAUDE_CODE, True,
        "Claude Code is signed in" + (f" ({signed_in_as})" if signed_in_as else "")
        + "; every frame bills to that subscription, not to an API key",
        executable=executable,
    )


def detect_engines(
    ollama_at: str | None = None, claude_code_program: str | None = None
) -> list[EngineVerdict]:
    """One verdict per engine, in the owner's order (BACKEND_CHOICES without the
    test fake), then claude-code (card #421; the picker learns it in #423).
    This is the one place that knows whether an engine can run here: the
    refusals' "what works" list and the CLI's default both come from it, so
    they cannot disagree with what the dialog (card #405) shows. `ollama_at`
    is the configured address, if any (`[model] ollama_url`);
    `claude_code_program` the configured program, if any (a `[model] command`
    under claude-code naming its own), else the built-in `claude`."""
    apple_silicon = on_apple_silicon()
    url = ollama_url(ollama_at)
    ollama = ollama_answers(url)
    return [
        EngineVerdict(
            "mlx", apple_silicon,
            "runs locally on this Apple Silicon Mac" if apple_silicon else "needs Apple Silicon",
        ),
        EngineVerdict(
            OLLAMA, ollama,
            f"Ollama is answering at {url}" if ollama
            else f"no Ollama server at {url}; install it from {OLLAMA_INSTALL}",
        ),
        EngineVerdict("openai", True, _key_required("openai")),
        EngineVerdict("claude", True, _key_required("claude")),
        claude_code_verdict(claude_code_program),
    ]


def claude_code_reply(stdout: str) -> str:
    """The reply text out of Claude Code's `--output-format json` stdout: the
    result object's `result` field (headless docs: "the text result in the
    `result` field"). Measured on 2.1.277 with no credentials: exit 1,
    nothing on stderr, and on stdout the result object with `is_error` true,
    `subtype` still "success" and the result "Not logged in · Please run
    /login", so `is_error` is the field trusted and the not-signed-in case
    is CommandFailed naming CLAUDE_CODE_SIGN_IN; any other error result is
    CommandFailed in Claude Code's own words. A stdout that is not the
    result object (a user's own `[model] command` with `--output-format
    text`, or a bare reply that happens to be JSON without a `result` key)
    passes through untouched."""
    try:
        reply = json.loads(stdout)
    except ValueError:
        return stdout
    if not isinstance(reply, dict) or "result" not in reply:
        return stdout
    result = str(reply.get("result") or "")
    if reply.get("is_error"):
        if "not logged in" in result.lower():
            raise CommandFailed(
                f"Claude Code is not signed in; run `{CLAUDE_CODE_SIGN_IN}` and try again "
                f"(it said: {result})"
            )
        raise CommandFailed(f"Claude Code reported an error: {result}")
    return result


def _works_here(verdicts: list[EngineVerdict]) -> tuple[str, ...]:
    """The backends this machine can run, as the verdicts say, plus the fake."""
    return (*(v.engine for v in verdicts if v.available), SCRIPTED)


def _refuse_here(reason: str, ollama_url: str | None) -> BackendUnavailable:
    """A refusal whose "what works" list comes from a fresh detection: for the
    branches that refuse on their own grounds (no Apple Silicon, no program)
    and have not probed the engines yet. Each refusal is terminal, so the
    probe runs once."""
    return _refusal(reason, works_here=_works_here(detect_engines(ollama_url)))


def default_engine(ollama_at: str | None = None) -> str:
    """What runs when nothing names an engine: the first detection says is
    available, in the owner's order. There is always one, because the cloud
    engines are available everywhere; no fallback, so if the list ever
    changes that invariant breaks loudly here rather than naming mlx."""
    return next(v.engine for v in detect_engines(ollama_at) if v.available)


def normalise_provider(provider: str | None) -> str:
    name = (provider or "").strip().lower()
    if name not in KEY_VARIABLES:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {', '.join(sorted(KEY_VARIABLES))}. "
            "For any other OpenAI-compatible endpoint, use 'openai' with base_url set."
        )
    return name


def resolve_provider_key(provider: str, explicit: SecretStr | None = None) -> str | None:
    """Find the key without ever letting it live in tracked source.

    Order: explicit config (from the git-ignored local file or an override), then
    the provider's own environment variables.
    """
    import os

    if explicit:
        # SecretStr keeps it out of reprs and tracebacks; unwrap only here.
        return explicit.get_secret_value()
    for name in KEY_VARIABLES.get((provider or "").strip().lower(), ()):
        value = os.environ.get(name)
        if value:
            return value
    return None


def is_cloud_primary(config: MelampusConfig) -> bool:
    return (config.model.backend or "mlx").strip().lower() not in LOCAL_BACKENDS


def build_primary_backend(config: MelampusConfig) -> VLMBackend:
    """The backend the main pipeline talks to, per `[model] backend`.

    Nothing set, the CLI has already written `default_engine()` here: the
    first engine detection says can run on this machine (card #404). `mlx` is
    the local-first path and exists only on Apple Silicon. The cloud choices
    are for machines without a local runtime — they reuse the exact classes
    escalation uses, so prompts, schema validation and the corrective retry
    are identical wherever the answer comes from.
    """
    kind = (config.model.backend or "mlx").strip().lower()
    settings = config.model

    if kind == "mlx":
        if not on_apple_silicon():
            raise _refuse_here(
                "The local MLX backend only runs on Apple Silicon Macs.", settings.ollama_url
            )
        from .backend import MLXBackend

        return MLXBackend(settings.repo, settings.temperature)

    if kind == OLLAMA:
        # Detection, run once here, before any image is read: a server that is
        # not there fails up front, with the fix, rather than once per frame
        # mid-run. The refusal's sentence is the ollama verdict's own words
        # (what --detect-engines and the dialog say), and its "what works"
        # list comes from the same verdicts: one probe, one sentence.
        verdicts = detect_engines(settings.ollama_url)
        ollama = next(v for v in verdicts if v.engine == OLLAMA)
        if not ollama.available:
            raise _refusal(
                f"{ollama.reason[0].upper()}{ollama.reason[1:]}.",
                works_here=_works_here(verdicts),
            )
        from .backend import OllamaBackend

        return OllamaBackend(
            settings.ollama_model, ollama_url(settings.ollama_url),
            temperature=settings.temperature, timeout=settings.timeout_seconds,
        )

    if kind == COMMAND:
        # Resolved here, before any image is read: a program that is not
        # there fails once, up front, with the fix, rather than once per
        # frame mid-run. The resolved path is what runs, so the check and
        # the run agree on the program.
        if not settings.command:
            raise _refuse_here(
                "The command backend needs [model] command: the program to run, as "
                "a list of arguments with {image} and {prompt} placeholders "
                "(docs/config.md § [model]).",
                settings.ollama_url,
            )
        from .backend import CommandBackend

        program = settings.command[0]
        executable = shutil.which(program)
        if executable is None:
            raise _refuse_here(
                f"The command '{program}' is not installed or not on PATH. Install "
                "it, make sure the shell melampus runs from can find it, or name "
                "its full path in [model] command.",
                settings.ollama_url,
            )
        if executable.lower().endswith(BATCH_SUFFIXES):
            # The path is the program (printable: the config refuses one
            # that is not) under a PATH directory, and PATH came from
            # whatever launched melampus, so it is shown through plain,
            # the way a program's stderr is.
            raise _refuse_here(
                f"The command '{program}' resolves to "
                f"{CommandBackend.plain(executable)}, a batch file "
                "that Windows runs through cmd.exe whatever it is told, so the "
                "prompt would be parsed as shell text rather than passed as one "
                "argument. Name the program's real entry in [model] command "
                "instead: its .exe, or node and the script the shim wraps.",
                settings.ollama_url,
            )
        # The backend stops the tree the program heads by its pid while the
        # program is exited but unreaped, so the pid is still its own. A
        # launcher that ignores SIGCHLD (inherited across exec) has the
        # kernel reap the program the moment it exits, so every such stop
        # would signal a number that may be someone else's: refused here,
        # once, rather than once per frame.
        if hasattr(signal, "SIGCHLD") and signal.getsignal(signal.SIGCHLD) is signal.SIG_IGN:
            raise _refuse_here(
                "The process that started melampus ignores SIGCHLD, so the "
                "command's exit cannot be seen without losing its pid: the kernel "
                "reaps the program the moment it exits, and what it started could "
                "not be stopped safely. Start melampus from a shell, or restore "
                "SIGCHLD's default disposition in the launcher.",
                settings.ollama_url,
            )
        return CommandBackend(
            settings.command, executable=executable, timeout=settings.timeout_seconds
        )

    if kind == CLAUDE_CODE:
        # The seam configured for Claude Code: the built-in template unless
        # the user set [model] command, and the reply unwrapped from the
        # result object. Resolved before any image is read, like `command`,
        # and as `ollama` does it: one detection, on the template's program,
        # whose claude-code verdict is the refusal's sentence and whose list
        # is its "what works", so Claude Code is asked its status once,
        # refused or built, and what runs is the executable that verdict
        # resolved.
        command = list(settings.command or CLAUDE_CODE_COMMAND)
        verdicts = detect_engines(settings.ollama_url, command[0])
        verdict = next(v for v in verdicts if v.engine == CLAUDE_CODE)
        if not verdict.available:
            raise _refusal(f"{verdict.reason}.", works_here=_works_here(verdicts))
        from .backend import CommandBackend

        return CommandBackend(
            command, executable=verdict.executable, timeout=settings.timeout_seconds,
            decode=claude_code_reply,
        )

    if kind == SCRIPTED:
        from .backend import ScriptedBackend

        return ScriptedBackend([])

    provider = normalise_provider(kind)

    # Import now, not on first request: a missing SDK should fail once, up front,
    # with an install hint — not once per frame mid-run.
    if provider == "claude":
        import anthropic  # noqa: F401
    else:
        import openai  # noqa: F401

    key = resolve_provider_key(provider, settings.api_key)
    model = settings.name or DEFAULT_MODELS[provider]

    from .backend import AnthropicBackend, OpenAIBackend

    if provider == "claude":
        return AnthropicBackend(
            key, model, effort=settings.effort, timeout=settings.timeout_seconds
        )
    return OpenAIBackend(
        key, model, base_url=settings.base_url, timeout=settings.timeout_seconds
    )


def apply_cloud_primary_defaults(config: MelampusConfig) -> list[str]:
    """Retune MLX-shaped defaults for a cloud primary, respecting explicit settings.

    Several defaults encode workarounds for the local runtime — the 1280 px ceiling
    exists because of an mlx-vlm token-window bug, the fallback ladder exists for
    its empty-generation failure, and the tight token caps assume a runtime that
    does not think before answering. None of that applies to a cloud model, and
    escalation already retunes them (escalate.build_cloud_identifier); a cloud
    primary deserves the same treatment.

    Only fields the user did not set are touched: `model_fields_set` distinguishes
    "the default" from "deliberately configured to the same number". Returns a
    description of each change, for the startup log.
    """
    changed: list[str] = []
    image, model = config.image, config.model

    if "max_edge" not in image.model_fields_set:
        image.max_edge = 2048
        changed.append("image.max_edge -> 2048 (the 1280 ceiling is an mlx-vlm bug)")
    if "fallback_edges" not in image.model_fields_set:
        image.fallback_edges = []
        changed.append("image.fallback_edges -> [] (retry ladder is for an mlx-vlm bug)")
    if "max_tokens" not in model.model_fields_set:
        model.max_tokens = 1200
        changed.append("model.max_tokens -> 1200 (thinking models spend tokens before JSON)")
    if "routing_max_tokens" not in model.model_fields_set:
        model.routing_max_tokens = 900
        changed.append("model.routing_max_tokens -> 900 (200 starves a thinking model)")
    if "cache_path" not in config.run.model_fields_set:
        # The reason escalation has its own cache file (config.py, EscalationConfig
        # .cache_path) applies with more force to a cloud primary: results carry
        # this backend's fingerprint, so sharing the local file would let the next
        # mlx pass silently overwrite answers that were paid for — and flipping
        # back would re-bill every one of them.
        config.run.cache_path = config.run.cache_path.with_name("identifications-cloud.jsonl")
        changed.append(
            "run.cache_path -> identifications-cloud.jsonl "
            "(cloud answers must not overwrite local ones)"
        )
    return changed
