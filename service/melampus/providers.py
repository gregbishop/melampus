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
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn
from urllib.parse import urlsplit

from pydantic import SecretStr

from .backend import CommandFailed, VLMBackend, _Deadline, _NotedHTTP, _NotedHTTPS, stderr_lines
from .config import MelampusConfig, ModelConfig

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

#: What the plugin's picker calls the four engines (card #423): the one
#: copy, carried on each verdict so the dialog holds no title table of its
#: own. The reason detection gives says the rest; a title only has to be
#: recognisable. The subscription CLIs' titles are composed from
#: CliEngine.title in _cli_verdict.
ENGINE_TITLES = {
    "mlx": "MLX — local, Apple Silicon",
    OLLAMA: "Ollama — local",
    "openai": "OpenAI — cloud, needs an API key",
    "claude": "Claude — cloud, needs an API key",
}

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
#: or --backend; detection knows no program to check for, so the plugin's
#: picker does not offer it (the two named CLIs below it does, card #423)
#: and it is not in BACKEND_CHOICES.
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

#: The flag that loads no settings file: `--restricted` (cli-reference,
#: 2.1.248 and later: "loads only managed settings and `--settings`", for
#: "an evaluation harness [that] drives `claude` on a shared machine and
#: Claude Code must not run commands or read that machine's user and
#: project settings"; it "also confines the built-in file tools to the
#: working directories", the staged image's own folder). A user's own
#: settings file can allow Read everywhere and open more folders, and
#: rules from every loaded settings file merge with --allowedTools
#: (permissions § Settings precedence: only a deny wins), so loaded, that
#: grant would let text rendered in a photograph reach files outside the
#: staged folder (Codex round 1, S1); `--setting-sources user`, the flag
#: this replaces, loaded it. The keychain login is not a settings file
#: and stays: measured on 2.1.278, `claude --restricted auth status
#: --json` reports the claude.ai login and its subscription; `--bare` is
#: the mode that skips keychain reads. The status check carries this
#: flag too, read off the template (CLAUDE_CODE_SETTINGS_FLAGS says how),
#: so the credential it reports is read under the settings the run
#: loads, in the same inherited environment (Codex round 1, S2).
CLAUDE_CODE_ISOLATION = "--restricted"

#: The one copy of the template. Every flag is from `claude --help` (2.1.277)
#: and code.claude.com/docs/en/headless: `-p` prints one reply and exits;
#: `--output-format json` puts the reply in the result object's `result`
#: field (claude_code_reply unwraps it); `--tools Read` leaves Claude Code
#: only the tool that reads files, which returns "PNG, JPG, and other image
#: formats ... as visual content that Claude can see" (tools-reference);
#: `--allowedTools Read(/{image})` pre-approves reading the one staged
#: file and nothing else (the run's working directory is that file's own
#: temporary folder, backend.py: CommandBackend.complete, so the reads
#: Claude Code allows without a rule, those inside the working directory,
#: reach the same one file): per
#: code.claude.com/docs/en/permissions § Read and Edit, `//path` is
#: "Absolute path from filesystem root" (`Edit(//tmp/scratch.txt)` "edits
#: the absolute path /tmp/scratch.txt"), and the staged path begins with
#: `/`. A photograph is untrusted input; rendered text in one asking for
#: ~/.ssh or .env gets that read denied, not answered (security review,
#: round 1). The path is the real one (backend.py: CommandBackend._argv),
#: since an allow rule "applies only when both the symlink path and its
#: target match"; `//c/...` is the documented form on Windows, card #424's
#: run. `--permission-prompts none` denies anything else that would wait
#: for a person; `--no-session-persistence` writes no transcript per frame;
#: `--strict-mcp-config` connects no MCP server; CLAUDE_CODE_ISOLATION
#: (`--restricted`) loads no settings file at all (Codex round 1, S1). The
#: prompt is the positional argument, last: the staged image's path for
#: the Read tool, then the pipeline's prompt in full, on one line up to the
#: placeholder: the template is printed in the `loading` line and in every
#: message that names the program, so the config refuses an element with a
#: line break in it (card #420), and the prompt's own line breaks arrive
#: through the placeholder, not the template.
#: Not `--bare`: bare mode never reads the subscription login (headless docs:
#: "bare mode doesn't use your subscription login").
CLAUDE_CODE_COMMAND = [
    CLAUDE_CODE_PROGRAM, "-p", "--output-format", "json", "--tools", "Read",
    "--allowedTools", "Read(/{image})", "--permission-prompts", "none", "--no-session-persistence",
    "--strict-mcp-config", CLAUDE_CODE_ISOLATION,
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
#: call, so detection and the up-front refusal spend nothing. Run under
#: the settings flags of the template that will run (claude_code_status).
CLAUDE_CODE_STATUS = ("auth", "status", "--json")

#: The flag under which no sign-in can help: `--bare` (`claude --help`,
#: 2.1.278: "OAuth and keychain are never read"; headless: "bare mode
#: doesn't use your subscription login"). A template carrying it is never
#: signed in to the subscription, and signing in changes nothing, so its
#: verdicts say to remove the flag instead of naming the sign-in (review
#: round 6, 1), the way CLAUDE_CODE_CREDENTIAL_FIX names what to remove.
CLAUDE_CODE_BARE = "--bare"
CLAUDE_CODE_BARE_FIX = (
    f"remove `{CLAUDE_CODE_BARE}` from `[model] command`, since bare mode never reads the "
    "subscription login"
)

#: The global flags that decide which settings files a run loads, and so
#: which credential it uses, with how many values each takes: the status
#: check carries them off the template that will run, values and order
#: kept, before the subcommand (the CLI takes its global flags there:
#: measured on 2.1.278, `claude --setting-sources bogus auth status
#: --json` is refused as an invalid setting source, and `claude
#: --restricted --settings '{"apiKeyHelper": ...}' auth status --json`
#: reports authMethod api_key_helper, apiKeySource apiKeyHelper). A
#: `[model] command` of the user's own can leave CLAUDE_CODE_ISOLATION
#: out, name a settings file or a source, or run bare, and the credential
#: differs each way: without `--restricted` the user's settings file is
#: loaded (measured: an apiKeyHelper there is reported with no flag and
#: not under `--restricted`); `--settings` "still appl[ies]" under
#: `--restricted` (`claude --help`); `--setting-sources` picks the files;
#: `--bare` never reads "OAuth and keychain" (`claude --help`), so the
#: check under it says not signed in (measured: loggedIn false, exit 1,
#: on the signed-in Mac). A check under fixed flags would approve the
#: subscription while the run billed an apiKeyHelper's key (Codex round
#: 2, C1); one read off the template is the run's own configuration,
#: whatever the template, in whichever spelling it uses (`--settings
#: file` or `--settings=file`: measured on 2.1.278, `claude --restricted
#: --settings=helper.json auth status --json` reports the helper exactly
#: as the two-argument form does, and a check that dropped the `=` form
#: approved the subscription for a run billing the helper's key; Codex
#: round 3, C1 and S1). Flags that decide nothing about credentials
#: (`--add-dir`, the tools, the prompt) are not carried.
CLAUDE_CODE_SETTINGS_FLAGS = {
    CLAUDE_CODE_ISOLATION: 0, CLAUDE_CODE_BARE: 0, "--settings": 1, "--setting-sources": 1,
}

#: What the status object calls the subscription sign-in: `authMethod`
#: "claude.ai" (measured on 2.1.278, with `subscriptionType` "max"). The
#: program's other words for it, read from its own source since the docs
#: list none: none, api_key, api_key_helper, oauth_token, third_party.
#: Only claude.ai is the subscription, and even then the login can be set
#: aside for a key: a print-mode run uses whatever credential Claude
#: Code's precedence puts first, the environment melampus runs from
#: included (authentication § Authentication precedence: "In
#: non-interactive mode (-p), the key is always used when present"), and
#: the status object then names it in `apiKeySource` (measured: the login
#: plus ANTHROPIC_API_KEY reports authMethod claude.ai, apiKeySource
#: ANTHROPIC_API_KEY, subscriptionType null, and `--text` says "Auth
#: token: claude.ai · not in use"). So the engine is available on
#: authMethod claude.ai with no apiKeySource, and on nothing else (Codex
#: round 1, C1 and S2): "signed in" alone would let a whole batch bill a
#: key while the cloud guards, off for a local engine, ask nothing.
CLAUDE_CODE_SUBSCRIPTION = "claude.ai"

#: What to remove for each credential that is not the subscription, by
#: the status object's name for it (apiKeySource first, then authMethod),
#: in the docs' own variable names (authentication § Authentication
#: precedence; `claude auth login --help`: "--console  Use Anthropic
#: Console (API usage billing) instead of Claude subscription", the
#: sign-in the program reports as apiKeySource "/login managed key").
CLAUDE_CODE_CREDENTIAL_FIX = {
    "ANTHROPIC_API_KEY": "unset ANTHROPIC_API_KEY",
    "api_key": "unset ANTHROPIC_API_KEY",
    "apiKeyHelper": "remove apiKeyHelper from the settings",
    "api_key_helper": "remove apiKeyHelper from the settings",
    "oauth_token": "unset CLAUDE_CODE_OAUTH_TOKEN and ANTHROPIC_AUTH_TOKEN",
    "third_party": "unset CLAUDE_CODE_USE_BEDROCK, CLAUDE_CODE_USE_VERTEX and CLAUDE_CODE_USE_FOUNDRY",
    "/login managed key": "that is the Console sign-in (API usage billing), so run `claude auth logout`",
}

#: Where the precedence is documented, for the refusal.
CLAUDE_CODE_AUTH_DOCS = "https://code.claude.com/docs/en/authentication#authentication-precedence"

#: How long the status check may take. A Node CLI answers it in a fraction
#: of a second (measured: 0.1 s); ten seconds is a broken install, and
#: --detect-engines must never hang the settings dialog.
CLAUDE_CODE_PROBE_SECONDS = 10.0

#: Codex CLI as an engine (card #422): the second CLI behind the same seam,
#: for when Claude is at its limit or the owner prefers it. `codex` resolves
#: to a CommandBackend on CODEX_COMMAND, or on `[model] command` when the
#: user sets one. Runs bill to the ChatGPT plan Codex is signed in to; a
#: Codex signed in with an API key is refused (CODEX_CLI).
CODEX = "codex"

#: The program, as shutil.which looks for it: `codex` on PATH.
CODEX_PROGRAM = "codex"

#: The one copy of the template. Every flag is from `codex exec --help`
#: (0.154.0) and developers.openai.com/codex (non-interactive-mode,
#: developer-commands, image-inputs): `exec` runs "non-interactively";
#: `--image {image}` attaches the staged JPEG ("Attach images to the first
#: message"; "PNG and JPEG" accepted), first, because the flag is variadic
#: (`-i, --image <FILE>...`) and takes a prompt right after it for a second
#: file (measured); `--json` makes stdout a JSONL stream, the reply the
#: agent_message's text and a failure the turn.failed's message
#: (codex_reply reads both); `--ephemeral` writes no session per frame;
#: `--skip-git-repo-check` runs from wherever melampus was launched;
#: `--ignore-user-config` loads no ~/.codex/config.toml ("Authentication
#: still uses CODEX_HOME"), so no MCP server starts per frame and the run is
#: the same on every machine; `--sandbox read-only` and `-c
#: approval_policy="never"` (the documented approval_policy value; exec
#: 0.154.0 rejects --ask-for-approval, measured) let the run proceed with
#: nobody to approve and nothing writable; `-c project_doc_max_bytes=0`
#: ("Maximum bytes read from AGENTS.md") keeps the launch directory's
#: instructions out of the prompt; `--color never` keeps ANSI out of the
#: stderr the error messages quote. The prompt is the positional argument,
#: last: the pipeline's prompt in full; the image needs no mention, it is
#: attached.
CODEX_COMMAND = [
    CODEX_PROGRAM, "exec", "--image", "{image}", "--json", "--ephemeral",
    "--skip-git-repo-check", "--ignore-user-config", "--sandbox", "read-only",
    "-c", 'approval_policy="never"', "-c", "project_doc_max_bytes=0", "--color", "never",
    "{prompt}",
]

#: Where to get Codex CLI when nothing on PATH is called `codex`.
CODEX_INSTALL = "https://developers.openai.com/codex/cli"

#: How to sign in from a shell (`codex login --help`: "Manage login"; with
#: no flags "Codex opens a browser for the ChatGPT OAuth flow", the plan).
CODEX_SIGN_IN = "codex login"

#: The documented, cheap sign-in check (developer-commands: "Print the
#: active authentication mode and exit with 0 when logged in"; measured on
#: 0.154.0: exit 0 and "Logged in using ChatGPT" on stderr, or exit 1 and
#: "Not logged in"): no model call, so detection spends nothing. It does
#: not know the plan's usage limit; only a run does (codex_reply).
CODEX_STATUS = ("login", "status")

#: How long the status check may take. A native binary answers it in
#: milliseconds (measured: 0.01 s); ten seconds is a broken install.
CODEX_PROBE_SECONDS = 10.0

#: The backends that run on this machine and bill nobody per call.
LOCAL_BACKENDS = ("mlx", OLLAMA, COMMAND, CLAUDE_CODE, CODEX, SCRIPTED)


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
    greyed-out choice rather than a mystery (card #404), and the title is
    what the picker calls it (card #423)."""

    engine: str
    title: str
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


def claude_code_status(command: list[str]) -> list[str]:
    """The status check for `command`, the template that will run: its
    settings-deciding global flags (CLAUDE_CODE_SETTINGS_FLAGS, values
    included, in the template's order and spelling) before
    CLAUDE_CODE_STATUS, so the credential the check reports is the one
    that template's run would bill (Codex round 2, C1). A value the
    template attaches with `=` (`--settings=file`, the CLI's other
    spelling: measured on 2.1.278, it reports the same status as
    `--settings file`) is carried as that one argument (Codex round 3, C1
    and S1). The argv after the executable."""
    carried: list[str] = []
    arguments = iter(command[1:])
    for argument in arguments:
        flag, attached, _ = argument.partition("=")
        values = CLAUDE_CODE_SETTINGS_FLAGS.get(flag)
        if values is None:
            continue
        carried.append(argument)
        if not attached:
            carried.extend(next(arguments, "") for _ in range(values))
    return [*carried, *CLAUDE_CODE_STATUS]


@dataclass(frozen=True, slots=True)
class Credential:
    """What a passed status check says the CLI is signed in with, as
    `CliEngine.account` reads it: `kind` is the account kind in the words
    `CliEngine.subscriptions` and `bills_per_call` use ("" when the check
    names nothing the reader can place); `signed_in_as` what the available
    verdict shows, the kind and, when named, the plan; `said` what the
    check said about the credential, safe to quote in a refusal ("" quotes
    nothing); `fix` what to remove before signing in, when known."""

    kind: str
    signed_in_as: str = ""
    said: str = ""
    fix: str = ""


@dataclass(frozen=True, slots=True)
class CliEngine:
    """A subscription CLI behind the command seam, in the owner's words: the
    engine's name, the program shutil.which looks for, where to get it, how
    to sign in, the documented cheap status check, the built-in template,
    the decoder that turns its stdout into the reply, and how its status
    check is derived from a template and read: which of its words are "not
    signed in", and what credential a passed check reports. What differs
    between Claude Code and Codex is data here; the verdict and the
    factory are one function each."""

    engine: str
    title: str
    program: str
    install: str
    sign_in: str
    status: tuple[str, ...]
    command: list[str]
    decode: Callable[[str], str]
    #: The check's argv after the executable, for the template that will
    #: run: `status` behind whatever of the template decides the credential
    #: (claude_code_status carries Claude Code's settings flags).
    status_check: Callable[[list[str]], list[str]]
    #: Whether a failed check says not signed in, in the CLI's own words;
    #: a failure that does not is reported as what ran and what it said.
    signed_out: Callable[[subprocess.CompletedProcess], bool]
    #: The credential a passed check reports.
    account: Callable[[subprocess.CompletedProcess], Credential]
    #: The subscription this engine runs on, for the refusal's sentence.
    subscription: str
    #: The account kinds that bill to that subscription, as `account` names
    #: them. Set, the guard fails closed: a passed check naming any other
    #: kind, or one `account` reads nothing from, is refused, and the words
    #: it could not place are not quoted (a status line is an unversioned
    #: CLI's prose; a wording melampus has not measured may carry key
    #: material, and an account it cannot place may bill per call). Empty,
    #: any signed-in account is accepted.
    subscriptions: tuple[str, ...] = ()
    #: The account kinds that bill per call rather than to a subscription,
    #: as `account` names them; signed in with one, the refusal says so.
    bills_per_call: tuple[str, ...] = ()
    #: Where the billing precedence is documented, for the refusal.
    billing_docs: str = ""
    #: The flag under which no sign-in can help (Claude Code's `--bare`), and
    #: what a verdict says to do instead of naming the sign-in when the
    #: check carries it.
    bare: str = ""
    bare_fix: str = ""


def _claude_code_status_object(status: subprocess.CompletedProcess) -> dict:
    """The status object `claude auth status --json` printed, or {}."""
    try:
        account = json.loads(status.stdout)
    except ValueError:
        return {}
    return account if isinstance(account, dict) else {}


def _claude_code_signed_out(status: subprocess.CompletedProcess) -> bool:
    """Not signed in is what the status object says: `loggedIn` false."""
    return _claude_code_status_object(status).get("loggedIn") is False


def _claude_code_account(status: subprocess.CompletedProcess) -> Credential:
    """The credential `claude auth status --json` reports: authMethod, set
    aside for a key when apiKeySource names one (CLAUDE_CODE_SUBSCRIPTION
    says why that is not the subscription), with the fix
    CLAUDE_CODE_CREDENTIAL_FIX knows for it."""
    account = _claude_code_status_object(status)
    method = str(account.get("authMethod") or "")
    key_source = str(account.get("apiKeySource") or "")
    return Credential(
        kind="" if key_source else method,
        signed_in_as=", ".join(
            str(account[key]) for key in ("authMethod", "subscriptionType") if account.get(key)
        ),
        said=", ".join(
            f"{key} {account[key]}" for key in ("authMethod", "apiKeySource") if account.get(key)
        ),
        fix=(
            CLAUDE_CODE_CREDENTIAL_FIX.get(key_source) or CLAUDE_CODE_CREDENTIAL_FIX.get(method)
            or "remove that credential from the environment melampus runs from"
        ),
    )


def _cli_verdict(cli: CliEngine, command: list[str] | None, probe_seconds: float) -> EngineVerdict:
    """Whether a subscription CLI can be the engine here: the program
    `command` (`cli.command` unless the user set a template) names must be
    on PATH, and its status check, derived from that template
    (`cli.status_check`), must say signed in within `probe_seconds`, to an
    account that bills to the subscription (`cli.subscriptions`). Never
    raises; a verdict reports. The reasons are the words the user sees:
    not installed with where to get it, not signed in with the check as
    run and the command that signs in, signed in but not to the
    subscription with the check as run, what to remove and the sign-in, a
    check that did not answer or failed some other way (in the CLI's own
    words), or available and billing to the subscription. A template
    carrying `cli.bare` is told to remove it in place of the sign-in,
    which cannot help it, in every verdict that would name the sign-in,
    the not-installed one included (review round 7, 1)."""
    command = command or cli.command
    program = command[0]
    check = cli.status_check(command)
    next_step = cli.bare_fix if cli.bare and cli.bare in check else f"sign in with `{cli.sign_in}`"
    title = f"{cli.title} — subscription, no API key"
    executable = shutil.which(program)
    if executable is None:
        return EngineVerdict(
            cli.engine, title, False,
            f"{cli.title} is not installed: nothing on PATH is called '{program}'; "
            f"install it from {cli.install}, then {next_step}",
        )
    try:
        status = subprocess.run(
            [executable, *check],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, timeout=probe_seconds,
        )
    except subprocess.TimeoutExpired:
        return EngineVerdict(
            cli.engine, title, False,
            f"`{program} {' '.join(check)}` did not answer within {probe_seconds:g}s",
        )
    except OSError as exc:
        return EngineVerdict(cli.engine, title, False, f"'{program}' could not be run: {exc}")
    if status.returncode != 0:
        # Not signed in is what the CLI says (`cli.signed_out`) or, without
        # a word, the documented exit alone: "Exits with code 0 if logged
        # in, 1 if not" (cli-reference), nothing on stderr. Any other
        # failure (an older CLI with no status subcommand, a usage error, a
        # crash) is reported in the CLI's own words, since signing in would
        # not help.
        said = stderr_lines(status.stderr)
        if cli.signed_out(status) or (status.returncode == 1 and not said):
            return EngineVerdict(
                cli.engine, title, False,
                f"{cli.title} is installed but not signed in: `{program} {' '.join(check)}` "
                f"says so; {next_step}",
            )
        return EngineVerdict(
            cli.engine, title, False,
            f"`{program} {' '.join(check)}` exited {status.returncode}"
            + (f": {said}" if said else " with nothing on stderr"),
        )
    credential = cli.account(status)
    if credential.kind in cli.bills_per_call or (
        cli.subscriptions and credential.kind not in cli.subscriptions
    ):
        said = credential.said or (
            f"{credential.kind}, which bills per call" if credential.kind in cli.bills_per_call
            else "nothing about the account this engine can place, and one it cannot place may bill per call"
        )
        return EngineVerdict(
            cli.engine, title, False,
            f"{cli.title} is signed in, but not to {cli.subscription}: `{program} "
            f"{' '.join(check)}` says {said}, and every frame would bill that "
            f"credential instead{f' ({cli.billing_docs})' if cli.billing_docs else ''}; "
            f"{f'{credential.fix}, then ' if credential.fix else ''}{next_step}",
        )
    return EngineVerdict(
        cli.engine, title, True,
        f"{cli.title} is signed in"
        + (f" ({credential.signed_in_as})" if credential.signed_in_as else "")
        + "; every frame bills to that subscription, not to an API key",
        executable=executable,
    )


def claude_code_verdict(command: list[str] | None = None) -> EngineVerdict:
    """Whether Claude Code can be the engine here (card #421): the program
    `command` (CLAUDE_CODE_COMMAND unless the user set a template) names on
    PATH, and its status check, under that template's own settings flags
    (claude_code_status), saying signed in to the subscription
    (CLAUDE_CODE_SUBSCRIPTION says why "signed in" alone is not enough),
    within CLAUDE_CODE_PROBE_SECONDS. Never raises; a verdict reports."""
    return _cli_verdict(CLAUDE_CODE_CLI, command, CLAUDE_CODE_PROBE_SECONDS)


def cli_commands(settings: ModelConfig) -> dict[str, list[str]]:
    """The template a subscription CLI's run would ask and run, by engine:
    `[model] command` when the engine is that CLI's and a command is set,
    else nothing, for the CLI's built-in template. Read here, once, by the
    factory and by --detect-engines, so the verdict the dialog shows is
    the verdict the run gets (Done-when 2), as the ollama address is: the
    program it names is the one asked, under the settings flags it carries
    (Codex round 2, C1)."""
    engine = (settings.backend or "").strip().lower()
    return {
        cli.engine: list(settings.command)
        for cli in CLI_ENGINES if engine == cli.engine and settings.command
    }


def _codex_signed_out(status: subprocess.CompletedProcess) -> bool:
    """Not signed in is what `codex login status` says: "Not logged in" on
    stderr, exit 1 (measured on 0.154.0)."""
    return "not logged in" in (status.stderr or "").lower()


def _codex_account(status: subprocess.CompletedProcess) -> Credential:
    """`codex login status`: "Logged in using ChatGPT" on stderr (measured
    on 0.154.0); the words after "using" are the account kind. An API-key
    sign-in (`codex login --with-api-key`) says "Logged in using an API key
    - " and a masked fragment of the key (measured on 0.155.1): the kind
    stops at that " - ", so no key material reaches a verdict, which goes
    to stdout, the plugin's engines file and the settings dialog."""
    _, using, kind = (status.stderr or "").strip().partition("Logged in using ")
    kind = kind.splitlines()[0].partition(" - ")[0].strip() if using else ""
    return Credential(kind=kind, signed_in_as=kind)


def codex_verdict(command: list[str] | None = None) -> EngineVerdict:
    """Whether Codex CLI can be the engine here (card #422): the program
    `command` (CODEX_COMMAND unless the user set a template) names on PATH
    and `codex login status` saying signed in to the ChatGPT plan (an
    API-key sign-in bills per call and is refused; so is any account the
    check does not name as ChatGPT), under CODEX_PROBE_SECONDS. Never raises; a verdict reports. Whether the plan
    is at its usage limit is not knowable here without a model call; the
    first run says, and the batch stops on it naming the reset time
    (codex_reply)."""
    return _cli_verdict(CODEX_CLI, command, CODEX_PROBE_SECONDS)


def detect_engines(
    ollama_at: str | None = None, commands: dict[str, list[str]] | None = None
) -> list[EngineVerdict]:
    """One verdict per engine, in the owner's order (BACKEND_CHOICES without the
    test fake), then claude-code and codex (cards #421, #422): the plugin's
    picker is built from this list, in this order (card #423).
    This is the one place that knows whether an engine can run here: the
    refusals' "what works" list and the CLI's default both come from it, so
    they cannot disagree with what the dialog (card #405) shows. `ollama_at`
    is the configured address, if any (`[model] ollama_url`); `commands`
    the configured template for a CLI engine, if any (a `[model] command`
    under that engine, read by `cli_commands(settings)`, by engine), else
    the CLI's built-in template."""
    apple_silicon = on_apple_silicon()
    url = ollama_url(ollama_at)
    ollama = ollama_answers(url)
    commands = commands or {}
    return [
        EngineVerdict(
            "mlx", ENGINE_TITLES["mlx"], apple_silicon,
            "runs locally on this Apple Silicon Mac" if apple_silicon else "needs Apple Silicon",
        ),
        EngineVerdict(
            OLLAMA, ENGINE_TITLES[OLLAMA], ollama,
            f"Ollama is answering at {url}" if ollama
            else f"no Ollama server at {url}; install it from {OLLAMA_INSTALL}",
        ),
        EngineVerdict("openai", ENGINE_TITLES["openai"], True, _key_required("openai")),
        EngineVerdict("claude", ENGINE_TITLES["claude"], True, _key_required("claude")),
        claude_code_verdict(commands.get(CLAUDE_CODE)),
        codex_verdict(commands.get(CODEX)),
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
        _cli_refuse(CLAUDE_CODE_CLI, result, signed_out="not logged in" in result.lower())
    return result


def _cli_refuse(cli: CliEngine, message: str, *, signed_out: bool) -> NoReturn:
    """A CLI's run failed in its own words, `message`: CommandFailed (the
    engine is broken, not the frame; the batch stops) in the user's terms.
    `signed_out` is the decoder's reading of those words: then the refusal
    names the command that signs in; otherwise it carries the CLI's words
    alone. What differs between the CLIs is the title and the sign-in
    command, the CliEngine's data."""
    if signed_out:
        raise CommandFailed(
            f"{cli.title} is not signed in; run `{cli.sign_in}` and try again (it said: {message})"
        )
    raise CommandFailed(f"{cli.title} reported an error: {message}")


#: Claude Code, as the one verdict and the one factory branch see it.
CLAUDE_CODE_CLI = CliEngine(
    CLAUDE_CODE, "Claude Code", CLAUDE_CODE_PROGRAM, CLAUDE_CODE_INSTALL, CLAUDE_CODE_SIGN_IN,
    CLAUDE_CODE_STATUS, CLAUDE_CODE_COMMAND, claude_code_reply,
    status_check=claude_code_status, signed_out=_claude_code_signed_out,
    account=_claude_code_account, subscription="a Claude subscription",
    subscriptions=(CLAUDE_CODE_SUBSCRIPTION,), billing_docs=CLAUDE_CODE_AUTH_DOCS,
    bare=CLAUDE_CODE_BARE, bare_fix=CLAUDE_CODE_BARE_FIX,
)


def codex_reply(stdout: str) -> str:
    """The reply text out of Codex CLI's `--json` stdout: the JSONL stream's
    last `item.completed` agent_message (docs: the sample stream; `-o`
    writes "the final message"). A `turn.failed` is the engine refusing,
    not the frame: measured on 0.154.0, a plan at its usage limit fails the
    turn with "You've hit your usage limit ... try again at <time>", and no
    credentials fail it with "401 Unauthorized", both exit 1 with the
    stream on stdout; each is CommandFailed in the user's terms (the reset
    time as the CLI said it; the sign-in command) plus Codex's own words,
    and any other failure is CommandFailed in Codex's words alone. `error`
    events before a completed turn are the CLI's own retries, not
    failures. A stdout that is not an event stream (a user's own `[model]
    command` without `--json`, or a bare reply that happens to be JSON
    without a `type`) passes through untouched."""
    events = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            return stdout
        if not isinstance(event, dict) or "type" not in event:
            return stdout
        events.append(event)
    if not events:
        return stdout
    reply = ""
    for event in events:
        if event["type"] == "turn.failed":
            _codex_refuse(str((event.get("error") or {}).get("message") or "the turn failed"))
        item = event.get("item") or {}
        if event["type"] == "item.completed" and item.get("type") == "agent_message":
            reply = str(item.get("text") or "")
    return reply


def _codex_refuse(message: str) -> NoReturn:
    """A failed turn's `message`, as CommandFailed: the usage limit is
    Codex's own refusal, named with the reset time as the CLI said it
    (measured: "... try again at Sep 19th, 2026 7:46 AM."); the measured
    "401 Unauthorized" is not signed in, read by its word, since Codex's
    failures end in a hex request id whose digits may contain 401; and
    anything else is Codex's words, both through the refusal shared with
    Claude Code."""
    lowered = message.lower()
    if "usage limit" in lowered:
        _, _, when = message.partition("try again at ")
        raise CommandFailed(
            "Codex CLI is at its usage limit"
            + (f", until {when.strip().rstrip('.')}" if when.strip() else "")
            + f"; wait for it to reset or switch engines (it said: {message})"
        )
    _cli_refuse(CODEX_CLI, message, signed_out="unauthorized" in lowered)


#: Codex CLI, as the one verdict and the one factory branch see it. Signed
#: in with an API key (`codex login --with-api-key`; the status check says
#: "Logged in using an API key", measured on 0.155.1), every frame would
#: bill the OpenAI API per token with none of the cloud guards, so that
#: account kind is refused by name: this engine runs on the ChatGPT plan
#: only ("Logged in using ChatGPT", measured on 0.154.0 and 0.155.1), and
#: a status line naming anything else, or nothing _codex_account reads, is
#: refused too, since the wording is one version's and the guard is
#: about money.
CODEX_CLI = CliEngine(
    CODEX, "Codex CLI", CODEX_PROGRAM, CODEX_INSTALL, CODEX_SIGN_IN,
    CODEX_STATUS, CODEX_COMMAND, codex_reply,
    status_check=lambda command: list(CODEX_STATUS), signed_out=_codex_signed_out,
    account=_codex_account, subscription="the ChatGPT plan", bills_per_call=("an API key",),
    subscriptions=("ChatGPT",),
)

#: The subscription CLIs, in the owner's order: the verdicts after the
#: four engines, and the templates cli_commands reads.
CLI_ENGINES = (CLAUDE_CODE_CLI, CODEX_CLI)


def _cli_backend(cli: CliEngine, settings: ModelConfig) -> VLMBackend:
    """The seam configured for one subscription CLI: the built-in template
    unless the user set [model] command, and the reply decoded from the
    CLI's stdout. Resolved before any image is read, like `command`, and
    as `ollama` does it: one detection, on the template's program, whose
    verdict for this CLI is the refusal's sentence and whose list is its
    "what works", so the CLI is asked its status once, under the
    template's own settings flags, refused or built, and what runs is the
    executable that verdict resolved. The template is read once, by the
    reader --detect-engines uses."""
    commands = cli_commands(settings)
    command = commands.get(cli.engine) or list(cli.command)
    verdicts = detect_engines(settings.ollama_url, commands)
    verdict = next(v for v in verdicts if v.engine == cli.engine)
    if not verdict.available:
        raise _refusal(f"{verdict.reason}.", works_here=_works_here(verdicts))
    from .backend import CommandBackend

    return CommandBackend(
        command, executable=verdict.executable, timeout=settings.timeout_seconds,
        decode=cli.decode,
    )


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
        return _cli_backend(CLAUDE_CODE_CLI, settings)

    if kind == CODEX:
        return _cli_backend(CODEX_CLI, settings)

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
