"""Docs drift gates.

Each test here makes one promise a doc carries mechanical, so the doc cannot
silently fall behind the code or the repo again (as happened when [occurrence] and
[quality] shipped undocumented). The promises: docs/config.md names every
implemented setting; AGENTS.md, and not .gitignore, names the install command for
the recorded plugins, and on a clone where an installer has run, that documented
command runs; no doc names a file by an uppercase name it does not have;
docs/brief.md names the pytest command CI actually runs and explains it as
installing from the lockfile; every doc block that installs the service, and CI, install
from the lockfile (card #425, Done-when 3 and 2); AGENTS.md points at
docs/brief.md without restating its values; AGENTS.md points at the standard
and names the tracker (card #410, Done-when 3); the Windows job runs the
plugin tests, so the command built for cmd.exe is run by cmd.exe (card #401,
Done-when 3); every `uv sync` that builds the executable, in CI and in
readme.md's build section, installs the extras the executable carries (card
#434, Done-when 1 and 3); CI packages one plugin zip per platform through the
script on every run and, on a pushed v* tag, its release job attaches both to
the GitHub release, which the install docs name (card #402); no doc names a
workflow file that does not exist; no doc states a test count, because the
suite grows with every card and CI checks no such number (card #437,
Done-when 1); readme.md's opening lists exactly the engines providers.py
offers, with what each bills, and names the build specification (card #491,
Done-when 1 and 3), and it is the only opening that lists them — the brief's
and architecture's cite that table instead of copying it; and a page whose
opening says this runs on macOS and Windows does not still offer Linux
further down.

The checks are deliberately dumb — substring presence of the backticked name — so
they never argue with prose style, only with absence. The one exception runs the
documented command against the checkout this clone was installed from, because
a command that only read well was itself the drift.
"""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from melampus import providers
from melampus.config import MelampusConfig

REPO = Path(__file__).resolve().parents[2]
CONFIG_DOC = REPO / "docs" / "config.md"
AGENTS_MD = REPO / "AGENTS.md"
PLUGIN_CHOICE = REPO / ".agents" / "on-purpose.json"
INSTALLED_SKILLS = REPO / ".agents" / "skills"
BRIEF = REPO / "docs" / "brief.md"
WORKFLOWS = REPO / ".github" / "workflows"
WORKFLOW_SUFFIXES = (".yml", ".yaml")  # GitHub runs both
CI_WORKFLOW = WORKFLOWS / "ci.yml"
LUA_PLUGIN_TESTS = REPO / "service" / "tests" / "test_lua_plugin.py"
PLUGIN_DOC = REPO / "docs" / "plugin.md"
GITIGNORE = REPO / ".gitignore"
README = REPO / "readme.md"
DOCS = [README, AGENTS_MD, *sorted((REPO / "docs").glob("*.md"))]


LOCKFILE_FLAGS = ("--locked", "--frozen")
# What a `uv sync` that builds the executable must install: PyInstaller (the
# `build` extra) and the SDKs the executable carries (card #434). PyInstaller
# bundles what the build venv has, so a sync missing one ships without it.
BUILD_EXTRAS = ("--extra build", "--extra cloud", "--extra openai")


def _lacking_build_extras(commands: list[str]) -> list[str]:
    """The commands among `commands` that do not name every build extra."""
    return [c for c in commands if any(extra not in c for extra in BUILD_EXTRAS)]


def _installs_from_the_lockfile(command: str, flags: tuple[str, ...] = LOCKFILE_FLAGS) -> bool:
    """`uv sync` with one of `flags` installs exactly uv.lock; anything else
    re-resolves from pyproject.toml's bounds. `--locked` also fails when the lock
    has drifted from pyproject.toml; `--frozen` installs the stale lock anyway."""
    accepted = "|".join(re.escape(flag) for flag in flags)
    return bool(re.search(rf"\buv sync\b[^&|;]*(?:{accepted})\b", command))


def test_every_config_field_is_documented():
    doc = CONFIG_DOC.read_text(encoding="utf-8")
    missing = []
    for section_name, section_field in MelampusConfig.model_fields.items():
        section_cls = section_field.default_factory
        if f"[{section_name}]" not in doc:
            missing.append(f"[{section_name}] (whole section)")
            continue
        for key in section_cls.model_fields:
            if f"`{key}`" not in doc:
                missing.append(f"{section_name}.{key}")
    assert not missing, (
        "settings implemented in config.py but absent from docs/config.md "
        f"(document them, including a Why): {missing}"
    )


def _documented_installer() -> str:
    """The installer path in the one `node <checkout>/bin/install.mjs <plugins>`
    command AGENTS.md carries, checked against the plugins .agents/on-purpose.json
    records."""
    plugins = json.loads(PLUGIN_CHOICE.read_text(encoding="utf-8"))["plugins"]
    commands = re.findall(r"`node (\S+/install\.mjs) ([^`]*)`", AGENTS_MD.read_text(encoding="utf-8"))
    assert commands, (
        "AGENTS.md must tell a fresh clone to run `node <on-purpose checkout>/bin/install.mjs "
        f"{' '.join(plugins)}` (the plugins recorded in .agents/on-purpose.json)"
    )
    assert len(commands) == 1, f"AGENTS.md names {len(commands)} install commands; one, for the recorded plugins"
    [(installer, named_plugins)] = commands
    assert named_plugins.split() == plugins, (
        f"AGENTS.md's install command names {named_plugins.split()}, "
        f".agents/on-purpose.json records {plugins}"
    )
    return installer


def test_agents_md_names_the_install_command_and_gitignore_does_not_restate_it():
    """The install outputs (.claude/settings.json, .agents/skills, .codex/agents)
    are machine-local and untracked; a fresh clone must be told how to regenerate
    them, with the same plugins .agents/on-purpose.json records. AGENTS.md is the
    one place that says so: .gitignore, which lists those outputs, points there
    rather than restating the command, so a plugin added later moves one file.
    This gate reads only the repository, so it holds on any clone and in CI."""
    _documented_installer()
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    assert "install.mjs" not in gitignore, (
        ".gitignore restates the install command that AGENTS.md is gated for; "
        "say the installer regenerates the ignored outputs and point at AGENTS.md"
    )


def test_installer_this_clone_was_installed_from_runs():
    """A command that merely reads well left a fresh clone without the plugins
    and the secret hook. Where the installer has run, .agents/skills holds its
    symlinks into the on-purpose checkout it ran from, so the `<checkout>` the
    documented command needs is known without naming it (a checkout can be
    anywhere; a tracked file holds one value). The documented command, with that
    checkout filled in, must run: with no plugins the installer prints usage and
    exits 2 before touching git or the repo. A clone without those outputs, CI
    included, has no installation to check and skips; the gate above still holds."""
    links = [p for p in INSTALLED_SKILLS.iterdir() if p.is_symlink()] if INSTALLED_SKILLS.is_dir() else []
    if not links:
        pytest.skip("on-purpose is not installed in this clone (no links in .agents/skills)")
    # Each link targets <checkout>/plugins/<plugin>/skills/<skill>.
    checkout = Path(os.readlink(links[0])).parents[3]
    installer = Path(_documented_installer().replace("<checkout>", str(checkout))).expanduser()
    run = subprocess.run(["node", str(installer)], cwd=REPO, capture_output=True, text=True)
    assert run.returncode == 2 and "usage: install.mjs" in run.stderr, (
        f"`node {installer}` is not the on-purpose installer: "
        f"exit {run.returncode}, stderr {run.stderr.strip()!r}"
    )


def test_docs_name_only_the_lowercase_files():
    """The real files are readme.md and docs/config.md. A doc that still says
    README.md or docs/CONFIG.md, or claims another doc does, is stale."""
    stale = []
    for doc in DOCS:
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if "README.md" in line or "CONFIG.md" in line:
                stale.append(f"{doc.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not stale, f"docs name uppercase files that do not exist: {stale}"


def _workflows() -> list[Path]:
    """Every workflow file in .github/workflows, read at call time."""
    return sorted(path for path in WORKFLOWS.iterdir() if path.suffix in WORKFLOW_SUFFIXES)


def test_docs_name_only_workflows_that_exist():
    """Card #402, round 2: the release steps were folded into ci.yml and
    release.yml removed. A doc that still names a workflow file that is not
    in .github/workflows is stale."""
    suffixes = "|".join(re.escape(suffix) for suffix in WORKFLOW_SUFFIXES)
    stale = []
    for doc in DOCS:
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for name in re.findall(rf"\.github/workflows/([\w.-]+(?:{suffixes}))", line):
                if not (WORKFLOWS / name).is_file():
                    stale.append(f"{doc.relative_to(REPO)}:{lineno}: {name}")
    assert not stale, f"docs name workflow files that do not exist: {stale}"


def _ci_pytest_commands() -> list[str]:
    """The `run:` line of every ci.yml step that invokes pytest."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    commands = [
        command
        for command in re.findall(r"^\s*run:\s*(.+?)\s*$", text, re.MULTILINE)
        if "pytest" in command
    ]
    assert commands, f"{CI_WORKFLOW.name} runs no pytest step"
    return commands


def test_brief_names_the_test_command_ci_runs():
    """Rule 11: the repo's own commands are the truth. CI gates merges with its
    own pytest invocation, so the stack contract must name that command too,
    not only the local one."""
    ci_commands = [f"`{command}`" for command in _ci_pytest_commands()]
    brief = BRIEF.read_text(encoding="utf-8")
    missing = [c for c in ci_commands if c not in brief]
    assert not missing, f"docs/brief.md's stack contract does not name what CI runs: {missing}"


def _fenced_commands(text: str) -> list[str]:
    """Every non-blank, non-comment line inside a fenced code block of a doc."""
    return [
        line
        for block in re.findall(r"^```\w*\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
        for line in block.splitlines()
        if line and not line.startswith("#")
    ]


def _section(text: str, heading: str) -> str | None:
    """The body of a doc's `## heading` section, up to the next `## ` heading
    or the end of the doc; None when the doc has no such section."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else None


def _raw_opening(text: str) -> str:
    """A doc's opening — everything before its first `## ` heading — as
    written, lines and all. The one place an opening's end is defined:
    `_opening` joins this, and the gate that reads readme.md's engine table
    reads it as is, because it matches line-anchored rows."""
    return text.split("\n## ", 1)[0]


def _opening(text: str) -> str:
    """A doc's opening, `_raw_opening`, as one line, its wraps normalized to
    single spaces. Every gate that looks for a phrase in an opening reads it
    through here: round 2's finding was a gate matching "macOS and Windows"
    against the raw opening, which skipped docs/brief.md because the phrase
    is wrapped there, so the Linux claim the gate exists to catch would have
    passed."""
    return " ".join(_raw_opening(text).split())


def test_the_opening_reader_joins_the_lines_a_phrase_is_wrapped_across():
    """Round 2, Codex finding: docs/brief.md's opening wraps "macOS and
    Windows" across two lines, so a gate that read the raw opening for that
    phrase skipped the file and the Linux claim below it. The opening reader
    hands back one line, so a phrase is found however the paragraph happens
    to be filled, and it still stops at the first `## ` heading."""
    text = "# Title\n\nruns on macOS and\nWindows.\n\n## Requirements\n\nLinux too.\n"
    assert _opening(text) == "# Title runs on macOS and Windows."


def test_the_section_reader_takes_the_heading_literally():
    """Round 7, finding 1: `_section` promises the body under a literal
    `## heading`, and readme.md has `## Windows (cloud inference)` today, so a
    heading with regex metacharacters must find its section rather than
    quietly reporting the doc has none."""
    text = "## Windows (cloud inference)\ncloud body\n## macOS\nmac body\n"
    assert _section(text, "Windows (cloud inference)") == "cloud body\n"


def test_the_section_reader_reads_a_docs_last_section():
    """Round 8, finding 1: `_section` promises None only when the doc has no
    such section, and every doc ends in a section with no `## ` heading after
    it (readme.md's `## License`, docs/plugin.md's `## Safety`), so the body
    of the last heading must be read up to the end of the text rather than
    reported as missing."""
    assert _section("## A\na\n## B\nb\n", "B") == "b\n"


def _row(text: str, key: str) -> str | None:
    """The line of a doc's config table whose first cell is `key` (the
    `| `key` |` row, the whole key); None when the text has no such row."""
    return next((line for line in text.splitlines() if line.startswith(f"| `{key}` |")), None)


def test_the_row_reader_takes_the_key_whole():
    """Round 8, finding 1: `_row` promises the row whose first cell is the
    key, whole: docs/config.md's keys share prefixes (`max_edge`,
    `max_tokens`; `ollama_model`, `ollama_url`), so a key must find its own
    row and not the first whose key starts the same way, and None only when
    the text has no such row."""
    text = "| `timeout_seconds` | `180` | seconds |\n| `timeout` | `3` | plain |\n"
    assert _row(text, "timeout") == "| `timeout` | `3` | plain |"
    assert _row(text, "timeouts") is None


def test_install_blocks_install_from_the_lockfile():
    """Card #425, Done-when 3: given a fresh clone, when the README setup runs,
    then the resolved versions match the lockfile. Only `uv sync --locked` (or
    `--frozen`) does that; `uv pip install` never reads uv.lock. The setup is
    every fenced block that installs the service, not only `## Install`: the
    Windows block and the provider-extras blocks (README and docs/config.md)
    would otherwise re-resolve from pyproject's bounds, and because `uv sync`
    is exact, an SDK added with `uv pip install` is removed the next time the
    Install block runs. Running the installs here would need the network, so
    the gate is on the commands themselves."""
    readme = README.read_text(encoding="utf-8")
    install = _section(readme, "Install")
    assert install is not None and any(_installs_from_the_lockfile(c) for c in _fenced_commands(install)), (
        "readme.md's ## Install section must install with `uv sync --locked`"
    )
    unlocked = [
        f"{doc.relative_to(REPO)}: {command}"
        for doc in DOCS
        for command in _fenced_commands(doc.read_text(encoding="utf-8"))
        if re.search(r"\buv (pip install|sync)\b", command) and not _installs_from_the_lockfile(command)
    ]
    assert not unlocked, (
        "every doc block that installs the service must use `uv sync --locked` "
        f"(or --frozen), not re-resolve with `uv pip install`: {unlocked}"
    )


def test_ci_installs_from_the_lockfile_before_pytest():
    """Card #425, Done-when 2: given CI, when it installs, then it installs from
    the lockfile and fails if the lockfile and pyproject disagree. Only
    `uv sync --locked` does both: `--frozen` installs a stale lock without
    complaint, and `uv pip install` re-resolves instead."""
    not_locked = [
        c for c in _ci_pytest_commands() if not _installs_from_the_lockfile(c, flags=("--locked",))
    ]
    assert not not_locked, f"CI's pytest step does not install with uv sync --locked: {not_locked}"


def test_ci_builds_and_smoke_tests_the_executable():
    """Card #399, Done-when 3: given the repo's test command, when it runs,
    then the build and the executable smoke test are part of it. The run that
    gates merges is CI's, so its pytest step must pass --build-binary and
    install the `build` extra PyInstaller comes from; with either missing, the
    smoke tests in test_binary.py skip on every CI run and Done-when 1 and 2
    are never checked where it counts.

    Card #434, Done-when 1 and 3: every job that builds also syncs the `cloud`
    and `openai` extras, on every platform alike. PyInstaller bundles what the
    build venv has, so a job that syncs only dev and build ships an executable
    whose `--backend claude` prints an install hint that means nothing
    inside a binary (the smoke test in test_binary.py proves the SDKs import;
    this gate keeps the extras in the command that builds)."""
    commands = _ci_pytest_commands()
    lacking_extras = _lacking_build_extras(commands)
    not_building = [c for c in commands if "--build-binary" not in c or c in lacking_extras]
    assert not not_building, (
        f"CI's pytest step must install {' '.join(BUILD_EXTRAS)} and run `pytest --build-binary`: "
        f"{not_building}"
    )


def test_brief_explains_ci_as_installing_from_the_lockfile():
    """The brief's paragraph on the two test commands explains what CI does with
    service/pyproject.toml. Since card #425 CI installs service/uv.lock with
    `uv sync --locked` rather than resolving pyproject, so the paragraph must
    name the lockfile and cannot still call it uncommitted."""
    brief = BRIEF.read_text(encoding="utf-8")
    paragraph = re.search(r"^- \*\*There are two test commands.*?(?=\n\n)", brief, re.MULTILINE | re.DOTALL)
    assert paragraph, "docs/brief.md no longer explains the two test commands"
    explanation = paragraph.group(0)
    assert "`service/uv.lock`" in explanation and "not committed" not in explanation, (
        "docs/brief.md's two-test-commands paragraph must say CI installs from "
        f"`service/uv.lock` and must not call the lockfile uncommitted:\n{explanation}"
    )


def test_agents_md_points_at_the_brief_without_restating_it():
    """docs/brief.md is the source of truth for build, test and run. AGENTS.md
    keeps the pointer; a second copy of the contract's values would drift."""
    agents = AGENTS_MD.read_text(encoding="utf-8")
    assert "`docs/brief.md`" in agents, "AGENTS.md must point at docs/brief.md"
    brief = BRIEF.read_text(encoding="utf-8")
    contract_values = [
        re.search(r"^- test: (`[^`]+`)", brief, re.MULTILINE),
        re.search(r"lockfile is (`[^`]+`)", brief),
    ]
    assert all(contract_values), "docs/brief.md no longer states its test command or lockfile"
    restated = [m.group(1) for m in contract_values if m.group(1) in agents]
    assert not restated, f"AGENTS.md restates stack-contract values that live in docs/brief.md: {restated}"


def test_agents_md_points_at_the_standard_and_names_the_tracker():
    """Card #410, Done-when 3: given AGENTS.md, when read, then it points at the
    standard and names this board as the tracker. The standard is its two files
    in the on-purpose checkout; the tracker is a `tracker:` line naming the board."""
    agents = AGENTS_MD.read_text(encoding="utf-8")
    standard = [
        f"`plugins/standard/standards/{name}.md`" for name in ("ticket", "tdd")
    ]
    missing = [s for s in standard if s not in agents]
    assert not missing, f"AGENTS.md does not point at the standard: {missing}"
    tracker = re.findall(r"^tracker: (.+)$", agents, re.MULTILINE)
    assert tracker, "AGENTS.md has no `tracker: ` line"
    assert "board=Melampus" in tracker[0], (
        f"AGENTS.md's tracker line does not name this board: {tracker[0]!r}"
    )


def test_docs_name_the_build_and_its_smoke_test():
    """Card #399: the executable is built by tools/build_binary.py and the test
    command builds and smoke-tests it with --build-binary. The stack contract's
    `build:` line and readme.md must name both, or nobody finds them.

    Round 2, finding 1: the option is what builds, not what runs the smoke
    tests. `built_executable` (conftest.py) builds only when the option is
    given and then skips only when no executable is there, so without the
    option the smoke tests run against an existing build and skip only when
    there is none — which is what conftest.py's own docstring says and what
    readme.md tells the reader. The brief's sentence on them says the same,
    or it states a skip the suite does not have."""
    brief = BRIEF.read_text(encoding="utf-8")
    build = re.search(r"^- build: (`[^`]+`)", brief, re.MULTILINE)
    assert build and build.group(1) == "`.venv/bin/python tools/build_binary.py`", (
        f"docs/brief.md's stack contract does not name the build: {build and build.group(1)!r}"
    )
    readme = (REPO / "readme.md").read_text(encoding="utf-8")
    # readme.md shows commands in fenced blocks, so match the bare text.
    missing = [
        command for command in (build.group(1).strip("`"),
                                ".venv/bin/python -m pytest -q --build-binary")
        if command not in readme
    ]
    assert not missing, f"readme.md does not name: {missing}"
    about_the_smoke_tests = [
        sentence for sentence in _sentences(brief)
        if "--build-binary" in sentence and "smoke test" in sentence
    ]
    assert about_the_smoke_tests, (
        "docs/brief.md does not say what --build-binary does to the smoke tests"
    )
    unconditional = [
        sentence for sentence in about_the_smoke_tests
        if not re.search(r"existing build|dist/melampus|no build|when (?:one|it) exists",
                         sentence)
    ]
    assert not unconditional, (
        "docs/brief.md has the smoke tests skipping on a missing option; they skip on a "
        f"missing build (service/tests/conftest.py's built_executable): {unconditional}"
    )


def test_the_brief_names_skips_as_what_the_two_runs_differ_in():
    """Round 3, finding 1: the correction in the test above moved the
    sentence's axis from skips to the build, and the differences it introduces
    are not the build's: `test_escalation.py`, `test_quality.py`'s corpus
    tests and the installed-checkout test all skip on what the runner has,
    whichever way the option is passed. The axis is skips, or the sentence
    promises a list it does not deliver."""
    brief = BRIEF.read_text(encoding="utf-8")
    about_the_two_runs = [
        sentence for sentence in _sentences(brief) if "differ only in" in sentence
    ]
    assert about_the_two_runs, (
        "docs/brief.md does not say what the two test runs differ only in"
    )
    axes = [
        axis
        for sentence in about_the_two_runs
        for axis in re.findall(r"differ only in ([\w-]+)", sentence)
    ]
    assert "skips" in axes, (
        "docs/brief.md's sentence on the two runs names an axis the list it introduces is "
        "not: those differences are skips the runner's environment causes, not the build's "
        f"doing: {axes or about_the_two_runs}"
    )


def test_readme_build_blocks_sync_the_sdk_extras():
    """Card #434: the executable carries the cloud SDKs, and PyInstaller bundles
    what the build venv has, so every `uv sync` in readme.md's build section
    (the macOS block and the Windows one) names the build, cloud and openai
    extras."""
    readme = README.read_text(encoding="utf-8")
    section = _section(readme, "Building the executable")
    assert section is not None, "readme.md has no ## Building the executable section"
    syncs = [c for c in _fenced_commands(section) if re.search(r"\buv sync\b", c)]
    assert syncs, "readme.md's build section has no uv sync command"
    without = _lacking_build_extras(syncs)
    assert not without, (
        f"readme.md's build section must sync {' '.join(BUILD_EXTRAS)}, or the executable "
        f"it builds lacks the SDKs: {without}"
    )


def test_readme_build_section_says_where_pyinstallers_cache_goes():
    """Card #440: the build keeps PyInstaller's cache inside the checkout, and
    readme.md's build section says so: it names the directory, the switch a
    caller sets to choose another, and that a corrupt cache is deleted."""
    section = _section(README.read_text(encoding="utf-8"), "Building the executable")
    assert section is not None, "readme.md has no ## Building the executable section"
    for phrase in ("PYINSTALLER_CONFIG_DIR", "build/pyinstaller-config", "delete"):
        assert phrase in section, f"readme.md's build section does not mention {phrase!r}"


def _jobs() -> dict[str, str]:
    """ci.yml's jobs, by name, each as its text."""
    text = CI_WORKFLOW.read_text(encoding="utf-8").split("\njobs:\n", 1)[1]
    parts = re.split(r"^  (?=\w[\w-]*:\s*$)", text, flags=re.MULTILINE)
    return {part.split(":", 1)[0]: part for part in parts if part.strip()}


def _windows_job() -> str:
    """The text of ci.yml's job on a Windows runner."""
    windows = [job for job in _jobs().values() if re.search(r"runs-on: windows-", job)]
    assert windows, f"{CI_WORKFLOW.name} has no job on a Windows runner"
    return windows[0]


def _windows_pytest_commands() -> list[str]:
    """The pytest commands of ci.yml's Windows job."""
    job = _windows_job()
    commands = [c for c in _ci_pytest_commands() if c in job]
    assert commands, "the Windows job runs no pytest step"
    return commands


def test_ci_builds_and_smoke_tests_the_windows_executable():
    """Card #400, Done-when 1: given the CI workflow runs on a Windows runner,
    when it finishes, then a melampus.exe exists that starts and analyzes a
    fixture image with the scripted backend. The proof is the run itself; this
    gate keeps the job in the workflow: a job on a Windows runner whose pytest
    step builds with --build-binary, as the test command does, and runs the
    binary smoke tests, and which uploads dist/melampus.exe as an artifact.
    The proof has to be readable where the owner looks, the job log: the step
    names every test it ran and its outcome (-v) and the reason for each skip
    (-rs), so the log says which tests ran against dist/melampus.exe rather
    than a count of dots."""
    job = _windows_job()
    pytest_steps = _windows_pytest_commands()
    assert all("--build-binary" in c and "tests/test_binary.py" in c for c in pytest_steps), (
        f"the Windows job's pytest step must build with --build-binary and run "
        f"the binary smoke tests: {pytest_steps}"
    )
    assert all({"-v", "-rs"} <= set(c.split()) for c in pytest_steps), (
        f"the Windows job's pytest step must name every test it ran (-v) and "
        f"the reason for each skip (-rs), so the log is the proof: {pytest_steps}"
    )
    assert re.search(r"uses: actions/upload-artifact@", job), "the Windows job uploads no artifact"
    assert "dist/melampus.exe" in job, "the Windows job does not upload dist/melampus.exe"


def test_ci_runs_the_plugin_command_through_cmd_exe_on_windows():
    """Card #401, Done-when 3: both invocation paths are covered. The macOS
    job runs the Lua suites and the command the plugin builds through sh
    against dist/melampus; the Windows job must run tests/test_lua_plugin.py
    too, so the command the plugin builds for cmd.exe is run by cmd.exe
    against dist/melampus.exe, on the one runner that has both. That takes a
    Lua interpreter on the runner (the suite self-skips without one),
    installed by a step of the job, and the file on its pytest line. The Lua
    the job installs is Lua 5.1, Lightroom's own, so on that runner the file
    catches dialect errors too; neither its module docstring nor docs/plugin.md's
    § Tests can still say the suites catch logic errors and not dialect ones."""
    job = _windows_job()
    installs_lua = [
        line for line in job.splitlines()
        if not line.strip().startswith("#") and "install" in line and re.search(r"\blua\b", line)
    ]
    assert installs_lua, "the Windows job installs no Lua interpreter, so the plugin tests skip there"
    pytest_steps = _windows_pytest_commands()
    assert all("tests/test_lua_plugin.py" in c for c in pytest_steps), (
        "the Windows job's pytest step must run tests/test_lua_plugin.py, so the "
        f"command the plugin builds for cmd.exe is run by cmd.exe: {pytest_steps}"
    )
    still_says_not_dialect = [
        text.relative_to(REPO).as_posix()
        for text in (LUA_PLUGIN_TESTS, PLUGIN_DOC)
        if "not dialect" in text.read_text(encoding="utf-8")
    ]
    assert not still_says_not_dialect, (
        f"{still_says_not_dialect} still say the Lua suites catch logic errors, not "
        "dialect ones; the Windows job runs them on Lua 5.1"
    )


def test_every_workflow_pins_every_pip_install_to_an_exact_version():
    """Security: a tool CI installs with pip outside the lockfile (uv, on the
    Windows runner) is fetched from PyPI at build time and then produces the
    executable that is uploaded as an artifact, so `pip install <name>` with no
    `==` runs whatever PyPI serves that day. Every pip install in every
    workflow names an exact version: ci.yml's executable is an artifact on a
    pull request and, on a v* tag, what ships (card #402)."""
    pip_installs = [
        (workflow.name, arguments)
        for workflow in _workflows()
        for arguments in re.findall(
            r"^\s*run:.*\bpip install\b(.*?)\s*$", workflow.read_text(encoding="utf-8"), re.MULTILINE
        )
    ]
    assert CI_WORKFLOW.name in {name for name, _ in pip_installs}, (
        f"ci.yml has no pip install step: {pip_installs}"
    )
    unpinned = [
        f"{name}: {requirement}"
        for name, arguments in pip_installs
        for requirement in arguments.split()
        if not requirement.startswith("-") and not re.fullmatch(r"[\w.\-\[\]]+==[\w.]+", requirement)
    ]
    assert not unpinned, f"a workflow installs from PyPI without an exact version: {unpinned}"


def test_the_pip_pinning_gate_reads_yaml_workflows_too(tmp_path, monkeypatch):
    """Round 3, finding 2: GitHub runs `.yaml` workflows as well as `.yml`,
    and the gate above promises every workflow, so an `x.yaml` whose pip
    install names no `==` must fail it rather than slip past a glob that
    spells only `.yml`. The folder is a stand-in read at call time; ci.yml is
    in it, pinned, so the only thing wrong is the .yaml file."""
    (tmp_path / "ci.yml").write_text("        run: pip install uv==0.8.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text("        run: pip install uv\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError, match=r"x\.yaml: uv"):
        test_every_workflow_pins_every_pip_install_to_an_exact_version()


def test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version():
    """Card #439, Done-when 1 and 2: given every `uses:` line in
    .github/workflows, when read, then it names a full commit SHA with the
    version as a trailing comment, and a line that names a moving tag instead
    fails this test. A tag can be moved to different code; a SHA cannot, and
    the comment is what a reader (and a future bump) sees the SHA as."""
    texts = {workflow.name: workflow.read_text(encoding="utf-8") for workflow in _workflows()}
    using = {name for name, text in texts.items() if _uses_lines(text)}
    assert CI_WORKFLOW.name in using, f"a workflow uses no action: {sorted(texts)}"
    unpinned = [f"{name}: {line}" for name, text in texts.items() for line in _unpinned_actions(text)]
    assert not unpinned, f"a workflow names an action by tag, not a commit SHA with its version: {unpinned}"


# The action-pinning gate's stand-in files are whole workflows, the way GitHub
# reads them: each test's lines follow a job's `steps:`, or the job itself
# where a line is the job's own key, six spaces in as in .github/workflows.
WORKFLOW_JOB = "jobs:\n  build:\n"
WORKFLOW_STEPS = WORKFLOW_JOB + "    steps:\n"


def test_the_action_pinning_gate_reads_flow_style_steps_too(tmp_path, monkeypatch):
    """Security review of card #439: a step written as a YAML flow mapping,
    `- {uses: actions/checkout@v4}`, is a `uses:` line naming a tag, and
    Done-when 2 promises the gate fails on it; a detection that only knows
    `- uses:` at the start of a line let it through. The folder is a stand-in
    read at call time; ci.yml in it is pinned, so the only thing wrong is the
    flow-style step in the .yaml file."""
    (tmp_path / "ci.yml").write_text(
        WORKFLOW_STEPS + "      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0\n", encoding="utf-8"
    )
    (tmp_path / "x.yaml").write_text(WORKFLOW_STEPS + "      - {uses: actions/checkout@v4}\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError, match=r"x\.yaml: - \{uses: actions/checkout@v4\}"):
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()


def test_the_action_pinning_gate_counts_a_flow_style_step_as_using_an_action(tmp_path, monkeypatch):
    """Round 1, finding 2: the gate asks "does this workflow use an action"
    before it asks "is every use pinned", and both questions are about the
    same lines, so they must be answered by one definition. A ci.yml whose
    only step is a flow mapping, pinned, with the version as a comment inside
    the mapping's continuation, uses an action and is pinned: the gate passes
    on it rather than reporting that the workflow uses no action."""
    (tmp_path / "ci.yml").write_text(
        WORKFLOW_STEPS + "      - { uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0\n        }\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()


def test_the_action_pinning_gate_reads_a_uses_key_wherever_yaml_puts_one(tmp_path, monkeypatch):
    """Round 4, finding 1: the gate is only as good as its idea of a `uses:`
    key, and two YAML spellings slipped past it, each letting a moving tag
    through. A `#` inside a quoted scalar is part of that scalar, not the
    start of a comment, so cutting the line at it threw the step's real
    `uses:` key away; and a key may be quoted, `"uses":`, which a pattern
    demanding `uses:` right after a space, `{` or `,` never saw. Both files
    name actions/checkout by tag, so both must be reported. The folder is a
    stand-in read at call time; ci.yml in it is pinned."""
    (tmp_path / "ci.yml").write_text(
        WORKFLOW_STEPS + "      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0\n", encoding="utf-8"
    )
    (tmp_path / "quoted-scalar.yaml").write_text(
        WORKFLOW_STEPS + "      - {name: 'Checkout # source', uses: actions/checkout@v4}\n", encoding="utf-8"
    )
    (tmp_path / "quoted-key.yaml").write_text(WORKFLOW_STEPS + '      - "uses": actions/checkout@v4\n', encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    assert "quoted-scalar.yaml: - {name: 'Checkout # source', uses: actions/checkout@v4}" in reported, reported
    assert 'quoted-key.yaml: - "uses": actions/checkout@v4' in reported, reported


def test_the_action_pinning_gate_reads_the_pin_in_the_code_not_in_a_comment(tmp_path, monkeypatch):
    """Round 4, finding 2: the pin check searched the whole line, comment
    and all, for a SHA with a version after it, so a step that still names a
    moving tag passed by mentioning a SHA in its comment. What a workflow
    runs is the reference in the line's code; the version is what the
    comment is for. The folder is a stand-in read at call time; ci.yml in it
    is pinned, so the only thing wrong is the .yaml file's tag."""
    (tmp_path / "ci.yml").write_text(
        WORKFLOW_STEPS + "      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0\n", encoding="utf-8"
    )
    (tmp_path / "x.yaml").write_text(
        WORKFLOW_STEPS + "      - uses: actions/checkout@v4 # formerly uses: "
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError, match=r"x\.yaml: - uses: actions/checkout@v4 # formerly"):
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()


def test_the_action_pinning_gate_reads_the_pin_of_every_uses_on_the_line(tmp_path, monkeypatch):
    """Round 5, finding 1: one line may hold more than one `uses:` key -- a
    whole `steps:` sequence written as a flow list is valid YAML that Actions
    runs -- and the pin was judged once per line, so a SHA anywhere in the
    code, with a `# v4` anywhere in the comment, vouched for every other
    `uses:` on that line unexamined. Every key's own reference must name a
    SHA; and one trailing comment cannot honestly name two actions'
    versions, so a line that uses two actions is reported whether or not
    both are pinned. The folder is a stand-in read at call time; ci.yml in it
    is pinned, so the only things wrong are the .yaml files."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "first-pinned.yaml").write_text(
        WORKFLOW_JOB + "      steps: [{uses: actions/checkout@" + sha + "}, {uses: actions/checkout@v4}] # v4.4.0\n",
        encoding="utf-8",
    )
    (tmp_path / "last-pinned.yaml").write_text(
        WORKFLOW_JOB + "      steps: [{uses: actions/checkout@v4}, {uses: actions/checkout@" + sha + "}] # v4.4.0\n",
        encoding="utf-8",
    )
    (tmp_path / "two-pinned.yaml").write_text(
        WORKFLOW_JOB + "      steps: [{uses: actions/checkout@" + sha + "}, {uses: actions/setup-python@" + sha + "}] # v4.4.0\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for name in ("first-pinned.yaml", "last-pinned.yaml", "two-pinned.yaml"):
        assert name in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_opens_a_quoted_scalar_only_where_one_can_begin(tmp_path, monkeypatch):
    """Round 5, finding 2: a `'` or `"` was taken as opening a quoted scalar
    wherever it appeared, so the apostrophe in the plain scalar `don't`
    inverted the line's quote parity. In YAML a quote is an indicator only
    where a scalar can begin, and a quoted scalar ends at its closing quote,
    `''` inside it being one apostrophe, not that end. Both spellings cost
    the gate a real step: the comment was cut inside a quoted `name:`, so
    the step's `uses:` key was thrown away with it and a moving tag was
    never even looked at. ci.yml here is a pinned step whose name holds an
    apostrophe: reading it as an open quote swallowed its `# v4.4.0` and
    raised a false alarm on a step that is pinned, so ci.yml must not be
    reported while both .yaml files must be."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(
        WORKFLOW_STEPS + "      - {name: Don't touch, uses: actions/checkout@" + sha + "} # v4.4.0\n", encoding="utf-8"
    )
    (tmp_path / "plain-apostrophe.yaml").write_text(
        WORKFLOW_STEPS + "      - {a: don't, name: 'Checkout # source', uses: actions/checkout@v4}\n", encoding="utf-8"
    )
    (tmp_path / "escaped-quote.yaml").write_text(
        WORKFLOW_STEPS + "      - {name: 'Checkout '' # '' source', uses: actions/checkout@v4}\n", encoding="utf-8"
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for name in ("plain-apostrophe.yaml", "escaped-quote.yaml"):
        assert name in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_finds_a_step_whatever_its_scalars_hold(tmp_path, monkeypatch):
    """Open finding 1 at b275279 (Codex round 2, 1; Claude round 4, 1):
    whether a quote opened a scalar was guessed from the one character
    before it, so a plain scalar that merely held a `-` or `:` before an
    apostrophe, `pre-'fix` or `a:'b`, and a quoted scalar after a tag or
    anchor, `!!str "a # b"` or `&n "a # b"`, each put the guess out of step
    with the line: it was cut at a `#` inside a quoted scalar, the step's
    `uses:` key was thrown away with the rest, and the moving tag was never
    looked at. Where a scalar begins and ends is YAML's to say, so the gate
    reads what YAML parses. ci.yml here is the first such step, pinned: it
    uses an action and is not reported, while every .yaml file is."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(
        WORKFLOW_STEPS + "      - {name: pre-'fix, env: {NOTE: 'Checkout # source'}, uses: actions/checkout@" + sha + "} # v4.4.0\n",
        encoding="utf-8",
    )
    steps = {
        "hyphen.yaml": "- {name: pre-'fix, env: {NOTE: 'Checkout # source'}, uses: actions/checkout@v4}",
        "colon.yaml": "- {name: a:'b, env: {NOTE: 'Checkout # source'}, uses: actions/checkout@v4}",
        "tag.yaml": '- {name: !!str "a # b", uses: actions/checkout@v4}',
        "anchor.yaml": '- {name: &n "a # b", uses: actions/checkout@v4}',
    }
    for name, step in steps.items():
        (tmp_path / name).write_text(f"{WORKFLOW_STEPS}      {step}\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for name, step in steps.items():
        assert f"{name}: {step}" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_takes_only_a_whole_sha_as_a_pin(tmp_path, monkeypatch):
    """Open finding 2 at b275279 (Codex round 2, 2, and its security review;
    Claude round 4, 2): the pin was forty hex digits followed by a word
    boundary, and every non-word character is one, so a ref whose name
    merely begins with a SHA -- a branch or tag `<sha>-moving`, `<sha>.1`,
    `<sha>/x`, each of which can be moved to different code -- passed as a
    commit pin. Done-when 1 is a full commit SHA, so the whole reference must
    be the action and the SHA, nothing after it. The folder is a stand-in
    read at call time; ci.yml in it is pinned, so every .yaml file, and
    only those, must be reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    steps = {
        "hyphen.yaml": f"- uses: actions/checkout@{sha}-moving # v4.4.0",
        "dot.yaml": f"- uses: actions/checkout@{sha}.1 # v4.4.0",
        "slash.yaml": f"- uses: actions/checkout@{sha}/x # v4.4.0",
    }
    for name, step in steps.items():
        (tmp_path / name).write_text(f"{WORKFLOW_STEPS}      {step}\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for name, step in steps.items():
        assert f"{name}: {step}" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_finds_no_key_inside_a_quoted_scalar(tmp_path, monkeypatch):
    """Open finding 3 at b275279 (Codex round 2, 3): a `uses:` key was
    found by a pattern over the line's code, so the words `uses:
    actions/checkout@v4` inside a quoted `name:` counted as a second key, and
    a step pinned correctly was reported as holding two action references.
    A key is what YAML parses as one, however it is quoted, and the text of
    a quoted scalar never is. ci.yml here is two such steps, single- and
    double-quoted, each pinned, so the gate passes on it."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(
        WORKFLOW_STEPS + "      - {name: 'Replaces uses: actions/checkout@v4', uses: actions/checkout@" + sha + "} # v4.4.0\n"
        '      - {name: "Replaces uses: actions/checkout@v4", uses: actions/checkout@' + sha + "} # v4.4.0\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()


def test_the_action_pinning_gate_reports_a_workflow_that_does_not_parse(tmp_path, monkeypatch):
    """The gate reads each workflow as YAML parses it, so a workflow that
    does not parse cannot be read for its `uses:` keys. That is a failure of
    the gate, reported by the file's name, not a file skipped: a gate that
    passed it would vouch for references it never read. ci.yml here is a
    flow mapping never closed, which a line scan took for a pinned step, and
    x.yaml has no `uses:` at all, so a line scan never looked at it; the gate
    names each as not parsing, not ci.yml as using no action."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"      - {{uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text("on: [push\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for name in ("ci.yml", "x.yaml"):
        assert f"{name}: does not parse as YAML" in reported, reported


def test_the_action_pinning_gate_reads_the_last_line_without_a_newline(tmp_path, monkeypatch):
    """The version comment is the text after the last node that ends on the
    line, but a block collection does not end where its text does: its end
    mark is where the parser found the next token, past any comment. On a
    file's last line with no newline after it, the step's block mapping and
    sequence end after its `# v4.4.0`, and a gate that counted them found no
    comment and reported a pinned step. Only a scalar or a flow collection
    ends where its text does. ci.yml here is one pinned step with no
    trailing newline, so the gate passes on it."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()


def test_the_action_pinning_gate_reports_a_uses_that_is_not_a_string(tmp_path, monkeypatch):
    """An action reference is a string, and a `uses:` whose value YAML
    parses as a list or a mapping names no action at all, whatever SHA is
    written inside it. It is not pinned, and it is reported by the file's
    name like any other line the gate cannot pass, not a crash out of the
    pin check. The folder is a stand-in read at call time; ci.yml in it is
    pinned, so both .yaml files, and only those, must be reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    steps = {
        "list.yaml": f"- uses: [actions/checkout@{sha}] # v4.4.0",
        "mapping.yaml": f"- uses: {{ref: actions/checkout@{sha}}} # v4.4.0",
    }
    for name, step in steps.items():
        (tmp_path / name).write_text(f"{WORKFLOW_STEPS}      {step}\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for name, step in steps.items():
        assert f"{name}: {step}" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_ends_on_an_alias_to_itself(tmp_path, monkeypatch):
    """YAML lets an anchored collection hold an alias to itself, and PyYAML
    composes that as a node that contains itself, so a walk that went into
    every node it met never ended: the gate hung instead of judging the
    workflow. The walk visits each node once. The folder is a stand-in read
    at call time; ci.yml in it is a pinned step whose `with:` holds such a
    loop, x.yaml a moving tag beside one, so the gate ends and reports x.yaml
    alone."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(
        f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n        with: &loop [*loop]\n", encoding="utf-8"
    )
    (tmp_path / "x.yaml").write_text(
        WORKFLOW_STEPS + "      - {uses: actions/checkout@v4, with: &loop {self: *loop}}\n", encoding="utf-8"
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    assert "x.yaml: - {uses: actions/checkout@v4, with: &loop {self: *loop}}" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_reports_a_value_that_runs_past_its_first_line(tmp_path, monkeypatch):
    """A `uses:` value may start on the line after its key and run on to
    the next, as a plain scalar folded over two lines. Then no scalar or
    flow collection ends on the line where it starts, so there is no text
    after one to read a comment from, and the gate crashed with ValueError
    instead of judging the line: it has no trailing comment, so it is not
    pinned, and it is reported by the file's name. The folder is a stand-in
    read at call time; ci.yml in it is pinned, so x.yaml, and only it, must
    be reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text(
        WORKFLOW_STEPS + "      - uses:\n          actions/checkout@v4\n          continued # v4\n", encoding="utf-8"
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    assert "x.yaml: actions/checkout@v4" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_reports_a_uses_key_with_no_value_at_the_end(tmp_path, monkeypatch):
    """YAML lets a key be written explicitly, `? uses`, with no value after
    it. PyYAML places that absent value where the next token would begin,
    and at the end of the file that is past its last line, so the gate
    looked for a line that is not there and crashed with IndexError instead
    of naming the file. A value placed past the text is reported on its
    key's line: it names no action, so it is not pinned. The folder is a
    stand-in read at call time; ci.yml in it is pinned, so both .yaml files,
    and only those, must be reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "last.yaml").write_text(WORKFLOW_JOB + "      ? uses\n", encoding="utf-8")
    (tmp_path / "then-comment.yaml").write_text(WORKFLOW_JOB + "      ? uses\n      # trailing\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for name in ("last.yaml", "then-comment.yaml"):
        assert f"{name}: ? uses" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_reports_a_workflow_nested_too_deep_to_parse(tmp_path, monkeypatch):
    """PyYAML composes a collection by recursing into it, so a workflow
    nested deeper than Python's recursion limit, a thousand flow sequences
    one inside another, cannot be composed: it raised RecursionError, which
    is not a YAMLError, so it escaped the gate instead of naming the file.
    A workflow the parser cannot read to the end is one that does not parse,
    reported by its name. The folder is a stand-in read at call time;
    ci.yml in it is pinned, so deep.yaml, and only it, must be reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "deep.yaml").write_text("on: " + "[" * 1000 + "]" * 1000 + "\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    assert "deep.yaml: does not parse as YAML" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_reads_no_comment_inside_a_scalar_that_runs_on(tmp_path, monkeypatch):
    """Round 3 (Codex code 1, security 1) and Claude code round 5, 1: the
    comment was the text after the last node that ends on the reference's
    line, but a quoted or plain scalar that starts on that line and runs on
    to the next ends on neither, so its first line's text was read as the
    comment, and a `# v4` inside a `name:` passed a step that has no version
    comment at all. The comment now stops where the first such scalar
    begins. A block scalar's header, `|- # v4.4.0`, is a real comment and
    still counts. Each file is a whole workflow, `jobs.<id>.steps`. ci.yml
    here is pinned in both spellings, so every .yaml file, and only those,
    must be reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(
        f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n"
        f"      - uses: |- # v4.4.0\n          actions/checkout@{sha}\n",
        encoding="utf-8",
    )
    first_lines = {
        "double.yaml": f'- {{uses: actions/checkout@{sha}, name: "pin # v4 kept',
        "single.yaml": f"- {{uses: actions/checkout@{sha}, name: 'pin # v4 kept",
        "plain.yaml": f"- {{uses: actions/checkout@{sha}, name: pin#v4",
    }
    (tmp_path / "double.yaml").write_text(
        f'{WORKFLOW_STEPS}      {first_lines["double.yaml"]}\n          for reference"}}\n', encoding="utf-8"
    )
    (tmp_path / "single.yaml").write_text(
        f"{WORKFLOW_STEPS}      {first_lines['single.yaml']}\n          for reference'}}\n", encoding="utf-8"
    )
    (tmp_path / "plain.yaml").write_text(f"{WORKFLOW_STEPS}      {first_lines['plain.yaml']}\n          kept}}\n", encoding="utf-8")
    first_lines["one-line-steps.yaml"] = f'steps: [{{uses: actions/checkout@{sha}}}, {{run: echo, name: "# v4'
    (tmp_path / "one-line-steps.yaml").write_text(
        f'{WORKFLOW_JOB}    {first_lines["one-line-steps.yaml"]}\n        x"}}]\n', encoding="utf-8"
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for name, line in first_lines.items():
        assert f"{name}: {line}" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_judges_an_alias_where_it_is_written(tmp_path, monkeypatch):
    """Round 3, Codex code 2: GitHub Actions supports YAML anchors and
    aliases, and PyYAML composes an alias as the very node its anchor names,
    marked where the anchor is written. So `uses: *checkout` was judged on
    the anchor's line: a pinned, commented anchor and a commented alias
    were two references on one line and reported, while an alias with no
    comment passed on the anchor's. A reference is judged where it is
    written. Each file is a whole workflow, `jobs.<id>.steps`. ci.yml here
    has both uses commented, so it passes; x.yaml's alias has no comment,
    so that line, and only it, is reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    anchored = f"- uses: &checkout actions/checkout@{sha} # v4.4.0"
    steps = f"{WORKFLOW_STEPS}      {anchored}\n"
    (tmp_path / "ci.yml").write_text(f"{steps}      - uses: *checkout # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text(f"{steps}      - uses: *checkout\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    assert "x.yaml: - uses: *checkout" in reported, reported
    assert f"x.yaml: {anchored}" not in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_takes_no_at_sign_before_the_sha(tmp_path, monkeypatch):
    """Round 3, noted by both reviewers: the pin was `\\S+@<40 hex>`, and
    `\\S+` takes an `@` too, so `actions/checkout@v4@<sha>`, a reference
    whose ref is `v4@<sha>` rather than a SHA, passed. GitHub refuses a
    reference with two `@` when it parses the workflow, but the gate should
    say so itself: the action is everything before the one `@`. The file is
    a whole workflow, `jobs.<id>.steps`; ci.yml here is pinned, so x.yaml,
    and only it, must be reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@v4@{sha} # v4.4.0\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    assert f"x.yaml: - uses: actions/checkout@v4@{sha} # v4.4.0" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_reads_uses_only_where_github_does(tmp_path, monkeypatch):
    """Round 3, Codex code 3 and security 2: every mapping key named `uses`
    was taken for an action reference, so ordinary data that happens to be
    named `uses` -- a `workflow_call` input, an `env` variable, matrix values
    and a matrix `include`, a step's `with:` input -- failed a workflow
    whose every action is pinned. GitHub reads a `uses:` in two places only:
    a job's steps, `jobs.<id>.steps[*].uses`, and a job itself, calling a
    reusable workflow, `jobs.<id>.uses`. Both files here hold all that data
    beside a real step and a real reusable-workflow job, and reuse the whole
    first job through an alias, `again: *build`, as GitHub's docs show; that
    job's step is still one reference, read once. In ci.yml both references
    are pinned, so it passes; in x.yaml both name a tag, so both are found
    and reported, and none of the data is."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    data = [
        "uses: {type: string}",
        "uses: ordinary-data",
        "env: {uses: ordinary-data}",
        "uses: [a, b]",
        "include: [{uses: c}]",
        "uses: an-input",
    ]
    def workflow(step: str, call: str) -> str:
        return (
            f"on:\n  workflow_call:\n    inputs:\n      {data[0]}\n"
            f"env:\n  {data[1]}\n"
            f"jobs:\n  build: &build\n    {data[2]}\n"
            f"    strategy:\n      matrix:\n        {data[3]}\n        {data[4]}\n"
            f"    steps:\n      - uses: {step}\n"
            f"        with:\n          {data[5]}\n"
            f"  call:\n    uses: {call}\n"
            "  again: *build\n"
        )

    pinned = {"step": f"actions/checkout@{sha} # v4.4.0", "call": f"org/repo/.github/workflows/reusable.yml@{sha} # v1.0.0"}
    (tmp_path / "ci.yml").write_text(workflow(**pinned), encoding="utf-8")
    tagged = {"step": "actions/checkout@v4", "call": "org/repo/.github/workflows/reusable.yml@v1"}
    (tmp_path / "x.yaml").write_text(workflow(**tagged), encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    assert f"x.yaml: - uses: {tagged['step']}" in reported, reported
    assert f"x.yaml: uses: {tagged['call']}" in reported, reported
    for line in data:
        assert f"x.yaml: {line}" not in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_takes_only_a_version_as_the_comment(tmp_path, monkeypatch):
    """Claude code round 6, 1: Done-when 1 is a SHA with the version as a
    trailing comment, and the gate asks the comment for `# v<digit>`, but no
    test held a SHA-pinned line whose comment is something else, so
    loosening that to any `#` left every test green. A comment that names
    no version, `# pinned`, tells a reader and a future bump nothing about
    which release the SHA is. The file is a whole workflow; ci.yml here is
    pinned with its version, so x.yaml, and only it, must be reported."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # pinned\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    assert f"x.yaml: - uses: actions/checkout@{sha} # pinned" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_reads_no_comment_inside_a_scalar_that_ends_on_the_line(tmp_path, monkeypatch):
    """Claude code round 7, 1: the comment is the text after the last node
    that ends on the line, and no test held where that starts. Reading from
    the line's start, or from the first node to end there, left every test
    green, because each stand-in with a `#` inside a one-line scalar also
    named a tag or had a real comment besides. Then a SHA-pinned step with no
    comment passed on a `# v...` inside its `name:`. Each step here is pinned
    with no comment, its only `# v...` inside a scalar that ends on its line:
    double-quoted after `uses`, single-quoted before it, and plain. Each is
    reported; ci.yml is pinned with its version and is not."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    steps = [
        f'- {{uses: actions/checkout@{sha}, name: "Checkout # v4.4.0"}}',
        f"- {{name: 'Checkout # v4.4.0', uses: actions/checkout@{sha}}}",
        f"- {{uses: actions/checkout@{sha}, name: pin#v4}}",
    ]
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text(WORKFLOW_STEPS + "".join(f"      {step}\n" for step in steps), encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for step in steps:
        assert f"x.yaml: {step}" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_takes_only_forty_hex_digits_as_the_sha(tmp_path, monkeypatch):
    """Claude code round 7, 2: Done-when 1 is a full commit SHA, and the
    pin asks for forty hex digits, but every stand-in's ref was either not
    hex (`v4`, `v1`) or a full SHA with something after it, so loosening the
    forty to any count, or the hex to any word character, left every test
    green. A short SHA, a tag whose name happens to be hex (`1`, `cafe`) and
    a branch named with forty word characters can each be moved or be
    ambiguous; none is a full commit SHA. Each carries a version comment, so
    only the ref is wrong, and each is reported; ci.yml is pinned and is not."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    steps = [
        "- uses: actions/checkout@11d5960 # v4.4.0",
        "- uses: someorg/someaction@1 # v1",
        "- uses: someorg/someaction@cafe # v1",
        f"- uses: someorg/someaction@{'main' * 10} # v1",
    ]
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text(WORKFLOW_STEPS + "".join(f"      {step}\n" for step in steps), encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for step in steps:
        assert f"x.yaml: {step}" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_takes_neither_a_word_nor_an_anchor_as_the_version(tmp_path, monkeypatch):
    """Claude code round 7, 3: the version comment is `#`, then `v` and a
    digit, and the test beside this one holds only the loosening to any
    `#`. Dropping the digit passes a comment that is a word, `# vendored`,
    or a bare `# v`; dropping the `#` passes a step with no comment at all
    when an anchor on its line supplies `v` and a digit, `&v1`. Each step
    here is pinned and none has a version comment, so each is reported;
    ci.yml is pinned with its version and is not."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    reported_lines = [
        f"- uses: actions/checkout@{sha} # vendored",
        f"- uses: actions/checkout@{sha} # v",
        "- uses: &v1 |-",
    ]
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text(
        f"{WORKFLOW_STEPS}      {reported_lines[0]}\n      {reported_lines[1]}\n"
        f"      {reported_lines[2]}\n          actions/checkout@{sha}\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for line in reported_lines:
        assert f"x.yaml: {line}" in reported, reported
    assert "ci.yml" not in reported, reported


def test_the_action_pinning_gate_takes_no_other_length_or_case_of_sha(tmp_path, monkeypatch):
    """Beside round 7, 2: a commit SHA is exactly forty lowercase hex
    digits, and the test above holds only the loosenings to any count from
    one, to seven through forty, and to any word character. Allowing
    thirty-nine or forty-one, any number from forty up, upper case, or a
    letter past `f` still left every test green. None of these refs is a
    commit SHA GitHub reads as one, so each can only be a branch or tag
    name, which can be moved: under any of those loosenings a movable ref
    would pass as a pin. Each has a version comment, so only the ref is
    wrong, and each is reported; ci.yml is pinned and is not."""
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    steps = [
        f"- uses: actions/checkout@{sha[:-1]} # v4.4.0",
        f"- uses: actions/checkout@{sha}0 # v4.4.0",
        f"- uses: actions/checkout@{sha}{sha[:24]} # v4.4.0",
        f"- uses: actions/checkout@{sha.upper()} # v4.4.0",
        f"- uses: actions/checkout@{sha[:-1]}g # v4.4.0",
    ]
    (tmp_path / "ci.yml").write_text(f"{WORKFLOW_STEPS}      - uses: actions/checkout@{sha} # v4.4.0\n", encoding="utf-8")
    (tmp_path / "x.yaml").write_text(WORKFLOW_STEPS + "".join(f"      {step}\n" for step in steps), encoding="utf-8")
    monkeypatch.setitem(globals(), "WORKFLOWS", tmp_path)
    with pytest.raises(AssertionError) as unpinned:
        test_every_workflow_pins_every_action_to_a_commit_sha_with_its_version()
    reported = str(unpinned.value)
    for step in steps:
        assert f"x.yaml: {step}" in reported, reported
    assert "ci.yml" not in reported, reported


RELEASE_ZIPS = ("Melampus-macOS.zip", "Melampus-Windows.zip")


class _AliasWhereWritten(yaml.SafeLoader):
    """The safe loader, except that an alias to a scalar composes to a copy
    of that scalar carrying the alias's own marks. PyYAML otherwise hands
    back the anchored node itself, marked where the anchor is written, so a
    reference written `*checkout` would be judged on another line."""

    def compose_node(self, parent, index):
        alias = self.peek_event() if self.check_event(yaml.AliasEvent) else None
        node = super().compose_node(parent, index)
        if alias is None or not isinstance(node, yaml.ScalarNode):
            return node
        return yaml.ScalarNode(node.tag, node.value, alias.start_mark, alias.end_mark, style=node.style)


def _nodes(documents: list[yaml.Node]) -> list[yaml.Node]:
    """Every node of those composed documents, once each, carrying the marks
    of where it is written. What is a key, a value or a quoted scalar is the
    parser's to say, not a guess from the characters around it. An alias to
    a collection is the very node it names, so a collection holding an alias
    to itself contains itself, and the walk visits a node it has met once."""
    nodes, pending = {}, list(documents)
    while pending:
        node = pending.pop()
        if id(node) in nodes:
            continue
        nodes[id(node)] = node
        if isinstance(node, yaml.MappingNode):
            pending.extend(part for pair in node.value for part in pair)
        elif isinstance(node, yaml.SequenceNode):
            pending.extend(node.value)
    return list(nodes.values())


def _keyed(node: yaml.Node, key: str) -> list[yaml.Node]:
    """The values under that key, if the node is a mapping; none if not."""
    if not isinstance(node, yaml.MappingNode):
        return []
    return [value for name, value in node.value if name.value == key]


def _action_references(text: str) -> list[tuple[str, bool]]:
    """Each line of that workflow text on which an action is referenced,
    stripped, and whether it is pinned. A reference is the value of a `uses`
    key where GitHub reads one: on a job under `jobs`, calling a reusable
    workflow, and on each step of a job's `steps`. A key named `uses`
    anywhere else, an `env` variable, a `with:` input, matrix data, a
    `workflow_call` input, is data. A job reused through an alias is the
    same node, read once. A key counts by its value as YAML parses it,
    however it is quoted, and the text of a quoted scalar is never a key.
    GitHub does not support the merge key `<<`, so a step built from one is
    not a step it runs. What the workflow runs is
    that value, so a SHA quoted in a comment pins nothing, and all of it must
    be one string, the action and a commit SHA: a ref that merely begins
    with a SHA, `<sha>-moving`, can be moved, and a list or mapping names no
    action at all. The version is what the line's trailing comment says.
    PyYAML drops comments, so the comment is the text after the last scalar
    or flow collection that ends on the line, which leaves a `#` inside a
    quoted scalar, or a flow mapping's closing brace, where it belongs; a
    block collection ends where the next token begins, past any comment, so
    it does not say where the line's text ends. The comment stops where a
    quoted or plain scalar that starts on the line and runs on past it
    begins, since everything after that is the scalar's text; a block
    scalar's header line, `|- # v4.4.0`, holds a real comment. A value that
    starts on the line and runs on past it leaves nothing ending there, so
    that line has no comment; an absent value, `? uses` at the end of the
    file, is placed past the last line, so it is read on its key's line and
    names no action. One trailing comment cannot name two actions'
    versions, so a line holding two references is not pinned whatever each
    names. A workflow that does not parse, or nests deeper than the parser
    can recurse, cannot be read for its references, so it is one line
    nothing pins, the parser's error, reported by its name."""
    try:
        documents = list(yaml.compose_all(text, Loader=_AliasWhereWritten))
    except (yaml.YAMLError, RecursionError) as error:
        return [(f"does not parse as YAML: {' '.join(str(error).split())}", False)]
    nodes = _nodes(documents)
    jobs = [
        job
        for document in documents
        for table in _keyed(document, "jobs")
        if isinstance(table, yaml.MappingNode)
        for _, job in table.value
    ]
    steps = [
        step
        for job in jobs
        for sequence in _keyed(job, "steps")
        if isinstance(sequence, yaml.SequenceNode)
        for step in sequence.value
    ]
    holders = {id(holder): holder for holder in jobs + steps}
    lines = text.splitlines()
    references = {}
    for holder in holders.values():
        if isinstance(holder, yaml.MappingNode):
            for key, value in holder.value:
                if key.value == "uses":
                    line = value.start_mark.line if value.start_mark.line < len(lines) else key.start_mark.line
                    references.setdefault(line, []).append(value)
    judged = []
    for line, values in sorted(references.items()):
        written = max(
            (
                node.end_mark.column
                for node in nodes
                if node.end_mark.line == line and (isinstance(node, yaml.ScalarNode) or node.flow_style)
            ),
            default=len(lines[line]),
        )
        runs_on = min(
            (
                node.start_mark.column
                for node in nodes
                if isinstance(node, yaml.ScalarNode)
                and node.style in (None, "'", '"')
                and node.start_mark.line == line < node.end_mark.line
            ),
            default=len(lines[line]),
        )
        pinned = (
            len(values) == 1
            and isinstance(values[0], yaml.ScalarNode)
            and re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", values[0].value)
            and re.search(r"#\s*v\d", lines[line][written:runs_on])
        )
        judged.append((lines[line].strip(), bool(pinned)))
    return judged


def _uses_lines(text: str) -> list[str]:
    """The lines of that workflow text on which an action is referenced,
    stripped: the same definition the pin check reads."""
    return [line for line, _ in _action_references(text)]


def _unpinned_actions(text: str) -> list[str]:
    """The lines of that workflow text on which an action is referenced but
    not pinned to a commit SHA with the version in a trailing comment."""
    return [line for line, pinned in _action_references(text) if not pinned]


def test_ci_packages_a_zip_per_platform_and_a_tag_releases_both():
    """Card #402, Done-when 1: given a tag is pushed, when the workflow runs,
    then a GitHub release exists with Melampus-macOS.zip and
    Melampus-Windows.zip attached. The proof is the first tagged run; this
    gate keeps the workflow honest before it, and there is one workflow:
    ci.yml, extended (rule 2), not copied, so what ships is what was tested
    by construction and no second file's brew, choco or pytest lines drift.
    It triggers on v* tags and still on pull requests; every job that builds
    (--build-binary) also packages through tools/package_plugin.py, the one
    place that knows the layout (no second copy in YAML), so the command that
    ships the zips runs on every pull request, not first on the tag; the
    Windows job's pytest line names tests/test_package_plugin.py, so the
    Windows zip ships from a script tested on Windows; both zip names are in
    it; a `release` job needs every packaging job, runs only on a tag, and
    alone holds `contents: write`, with no scope beyond contents anywhere in
    the file. That every `uses:` line is pinned is card #439's gate, over
    every workflow, so it is not asserted again here."""
    ci = CI_WORKFLOW.read_text(encoding="utf-8")
    copies = [w.name for w in _workflows() if w != CI_WORKFLOW and "pytest" in w.read_text(encoding="utf-8")]
    assert not copies, f"a second workflow copies ci.yml's build steps; extend ci.yml instead: {copies}"

    on = re.search(r"^on:\n((?:  .*\n)+)", ci, re.MULTILINE)
    assert on, "ci.yml has no on: block"
    assert re.search(r"^  push:\n(?:    .*\n)*?    tags:\s*\[\s*['\"]?v\*", on.group(1), re.MULTILINE), (
        "ci.yml does not trigger on pushed v* tags")
    assert re.search(r"^  pull_request:", on.group(1), re.MULTILINE), "ci.yml no longer runs on pull requests"

    jobs = _jobs()
    building = {name for name, job in jobs.items() if "--build-binary" in job}
    packaging = {name for name, job in jobs.items() if "tools/package_plugin.py" in job}
    assert building and packaging == building, (
        f"every job that builds must package through tools/package_plugin.py: builds {sorted(building)}, "
        f"packages {sorted(packaging)}")
    windows_steps = _windows_pytest_commands()
    assert all("tests/test_package_plugin.py" in c for c in windows_steps), (
        "the Windows job must run tests/test_package_plugin.py, so the Windows zip "
        f"ships from a script tested on Windows: {windows_steps}")
    missing = [z for z in RELEASE_ZIPS if z not in ci]
    assert not missing, f"ci.yml does not name {missing}"

    release = jobs.get("release")
    assert release, "ci.yml has no release job"
    assert re.search(r"^\s+if:\s*github\.ref_type == 'tag'\s*$", release, re.MULTILINE), (
        "the release job must run only on a tag: if: github.ref_type == 'tag'")
    needs = re.search(r"^\s+needs:\s*\[(.*?)\]", release, re.MULTILINE)
    needed = {n.strip() for n in needs.group(1).split(",")} if needs else set()
    assert packaging <= needed, f"the release job must need every packaging job: needs {sorted(needed)}"
    assert "gh release create" in release, "the release job does not create the release"
    scopes = re.findall(r"^\s+([\w-]+): (read|write|none)$", ci, re.MULTILINE)
    other = [f"{scope}: {level}" for scope, level in scopes if scope != "contents"]
    assert not other, f"ci.yml grants more than contents: {other}"
    writes = [line for line in ci.splitlines() if re.search(r"^\s+contents: write$", line)]
    assert len(writes) == 1 and "contents: write" in release, "contents: write must be granted once, on the release job"


def test_install_docs_name_the_release_zips_and_keep_the_from_source_path():
    """Card #402: a user installs from a release download, one zip per
    platform, through Plug-in Manager; readme.md's Lightroom section must name
    both zips, and keep the copy-from-dist step for a build from source. The
    section sends the user to docs/plugin.md for the install steps, so that
    page's ## Install must tell the same story: both zips, and the copy for a
    build from source, not a bare executable downloaded from a release."""
    sections = {
        README: ("Reviewing in Lightroom", README.read_text(encoding="utf-8")),
        PLUGIN_DOC: ("Install", PLUGIN_DOC.read_text(encoding="utf-8")),
    }
    for doc, (heading, text) in sections.items():
        name = doc.relative_to(REPO).as_posix()
        section = _section(text, heading)
        assert section is not None, f"{name} has no ## {heading} section"
        missing = [z for z in RELEASE_ZIPS if z not in section]
        assert not missing, f"{name}'s {heading} section does not name {missing}"
        assert "cp dist/melampus plugin/Melampus.lrplugin/" in section, (
            f"{name}'s {heading} section lost the from-source install")


def _picker() -> list[tuple[str, str]]:
    """The engines the plugin's picker offers, in its order, as (name, title)
    pairs, read from providers.detect_engines, the list the picker is built
    from (card #423): the one copy every engine gate below reads, so an
    engine added there reaches them all. The title is the verdict's up to
    its " — ", what the picker calls the engine. Detection stays on this
    machine and runs nothing: Ollama is not asked (stubbed here, for the
    call), and the subscription CLIs are not run (conftest's
    no_ambient_subscription_cli stubs their verdicts in every test)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(providers, "ollama_answers", lambda url=None: False)
        verdicts = providers.detect_engines()
    return [(verdict.engine, verdict.title.split(" — ")[0]) for verdict in verdicts]


def test_readme_opening_lists_the_engines_providers_offers_and_names_the_spec():
    """Card #491, Done-when 1 and 3: given readme.md's first screen (everything
    before its first `## ` heading), when read, then its engine table names
    exactly the engines a user can pick, in the picker's order: providers'
    detect_engines, read through `_picker`; and each row says what the
    engine bills, from the same module: nothing for a local engine, an API
    key for one in KEY_VARIABLES, and for one in CLI_ENGINES the
    subscription its CliEngine names (review round 4, finding 1: the word
    "subscription" alone let the two CLIs' cells swap and stay green).
    An engine that bills runs its model off this machine, so its Where it
    runs cell names whose API or servers every frame goes to (security
    review, round 4): the subscription CLIs' cells named only the program
    installed here, the way the `ollama` row names the server on this
    machine, while Claude Code reads the staged frame into its conversation
    with Anthropic and Codex attaches it to its first message to OpenAI.
    The `ollama` row's nothing is qualified with `ollama_model` (review
    round 4, finding 2): the same opening says a cloud model there runs
    under the Ollama account the server is signed in to, so a bare
    "nothing" is false for that setting.
    `scripted` (the fake) and `command` (the seam, not in the picker) must
    not appear. The opening also names `AGENTS.md` and `docs/brief.md`, not
    CLAUDE.md, as the build specification: CLAUDE.md is two includes now."""
    # The raw opening, not `_opening`: the rows below are matched line by line.
    opening = _raw_opening(README.read_text(encoding="utf-8"))
    rows = re.findall(r"^\| `([\w-]+)` \|(.*)$", opening, re.MULTILINE)
    listed = [name for name, _ in rows]
    picker = [name for name, _ in _picker()]
    assert listed == picker, (
        f"readme.md's opening must list the engines the picker offers, in its order: {picker}, not {listed}"
    )
    subscriptions = {cli.engine: cli.subscription for cli in providers.CLI_ENGINES}
    for name, row in rows:
        if name in providers.KEY_VARIABLES:
            expected = "API key"
        elif name in subscriptions:
            expected = subscriptions[name]
        else:
            expected = "nothing"
        # The What it bills cell alone. Read against the whole row, the word
        # is satisfied by the Where it runs cell that already carries it, and
        # a row claiming a subscription CLI costs nothing stays green.
        cells = [cell.strip() for cell in row.split("|")]
        assert len(cells) == 3 and not cells[-1], (
            f"readme.md's row for `{name}` is not an Engine / Where it runs / What it bills "
            f"row: {row.strip()}")
        bills = cells[1]
        assert expected in bills, (
            f"readme.md's What it bills cell for `{name}` does not say it bills "
            f"{expected!r}: {bills!r}")
        assert name != providers.OLLAMA or "`ollama_model`" in bills, (
            f"readme.md's What it bills cell for `{name}` says {bills!r} without naming "
            "`ollama_model`, which can name one of Ollama's cloud models")
        where = cells[0]
        assert expected == "nothing" or re.search(r"\b\w+'s (?:API|servers)\b", where), (
            f"readme.md's Where it runs cell for `{name}` does not say whose API or servers "
            f"every frame goes to, though it bills {expected!r}: {where!r}")
    for spec in ("`AGENTS.md`", "`docs/brief.md`"):
        assert spec in opening, f"readme.md's opening does not name {spec} as the build specification"
    assert "CLAUDE.md" not in opening, "readme.md's opening still calls CLAUDE.md the build specification"


def test_the_openings_privacy_claim_names_the_settings_that_can_send_the_image_elsewhere():
    """Security review, rounds 1 and 3: given an opening that promises no
    image leaves the machine on a local engine, when read, then it names
    both settings that break the promise for `ollama`, in the same breath.
    `mlx` runs in-process, but the `ollama` backend posts every staged frame
    to whatever `[model] ollama_url` names, and docs/config.md documents
    setting it "for a server on another port or host", https included
    (round 1); and it asks for whatever `[model] ollama_model` names, "a tag
    from ollama.com/library" by docs/config.md, where a tag may be one of
    Ollama's cloud models, which the server on this machine runs on Ollama's
    own under the account it is signed in to (round 3). Naming one and not
    the other reads as the whole list, so the first screen still promises a
    confidentiality the configuration does not enforce."""
    settings = {
        "ollama_url": "the setting that can point the ollama engine at another host",
        "ollama_model": "the setting that can name one of Ollama's cloud models",
    }
    for doc in (README, BRIEF):
        opening = _opening(doc.read_text(encoding="utf-8"))
        if "leaves the machine" not in opening:
            continue
        for setting, why in settings.items():
            assert setting in opening, (
                f"{doc.name}'s opening promises no image leaves the machine without naming "
                f"`{setting}`, {why}")


def test_only_the_readme_opening_lists_the_engines_the_other_openings_point_at_it():
    """Review round 1, finding 3: readme.md's opening carries the engine
    table, and the gate above holds it to providers.py. The brief's and
    architecture's openings may name the shape — local first, or a cloud API,
    or a subscription CLI — and cite that table; they may not restate the
    names, by key or by the title providers.py gives them, because a copy no
    gate reads is exactly what drifted before this card. A sentence that names
    one engine to qualify a claim about it (`[model] ollama_url`, the privacy
    caveat above) is not a list and does not trip this: engine keys are read
    as backticked tokens, titles as whole words, both from `_picker`."""
    picker = _picker()
    names = [name for name, _ in picker]
    words = [title for _, title in picker]
    for doc in (BRIEF, REPO / "docs" / "architecture.md"):
        opening = _opening(doc.read_text(encoding="utf-8"))
        assert "readme.md" in opening, (
            f"{doc.name}'s opening does not cite readme.md, which carries the engine list")
        restated = [n for n in names if n in re.findall(r"`([\w-]+)`", opening)]
        restated += [w for w in words if re.search(rf"\b{re.escape(w)}\b", opening)]
        assert not restated, (
            f"{doc.name}'s opening restates readme.md's engine list ({restated}); only the "
            "README's copy is held to providers.py, so name the shape and cite the table")


def test_the_engine_gates_read_the_picker_detect_engines_builds(monkeypatch, tmp_path):
    """Review round 3, finding 3: the two gates above built the picker's
    engine list by hand, one from BACKEND_CHOICES, the other from
    ENGINE_TITLES, both adding the two CLIs themselves; the picker is built
    from providers.detect_engines (card #423), and a seventh verdict there
    left both green with readme.md listing six. Given a seventh verdict,
    readme.md's opening, which lists six, fails the first gate, and an
    opening that names the seventh, by key or by title, fails the second.
    And the list is read without asking Ollama: the probe here records, and
    must not be reached."""
    detect = providers.detect_engines
    seventh = providers.EngineVerdict("gemini", "Gemini — cloud, needs an API key", True, "API key required")
    monkeypatch.setattr(providers, "detect_engines", lambda ollama_at=None: [*detect(ollama_at), seventh])
    asked = []
    monkeypatch.setattr(providers, "ollama_answers", lambda url=None: asked.append(url) or False)
    with pytest.raises(AssertionError, match="gemini"):
        test_readme_opening_lists_the_engines_providers_offers_and_names_the_spec()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "architecture.md").write_text("# Architecture\n\nThe engines are readme.md's table.\n",
                                          encoding="utf-8")
    monkeypatch.setitem(globals(), "REPO", tmp_path)
    monkeypatch.setitem(globals(), "BRIEF", docs / "brief.md")
    for restated, written in (("gemini", "`gemini`"), ("Gemini", "Gemini")):
        (docs / "brief.md").write_text(
            f"# brief\n\nThe engines are readme.md's table, {written} among them.\n", encoding="utf-8")
        with pytest.raises(AssertionError, match=re.escape(f"['{restated}']")):
            test_only_the_readme_opening_lists_the_engines_the_other_openings_point_at_it()
    assert not asked, f"reading the picker asked Ollama at {asked}"


def test_the_readme_billing_check_reads_the_subscription_clis_from_cli_engines(monkeypatch, tmp_path):
    """After review round 3: the README gate's billing check named the two
    subscription CLIs by hand, so a third, in providers.CLI_ENGINES and the
    picker alike, fell through to "nothing" and a row saying it costs
    nothing passed. Given a third CLI engine in both, a readme.md whose row
    for it bills nothing fails the gate: its What it bills cell must name
    that CLI's subscription."""
    import dataclasses

    third = dataclasses.replace(providers.CODEX_CLI, engine="gemini-cli", title="Gemini CLI",
                                subscription="a Gemini subscription")
    monkeypatch.setattr(providers, "CLI_ENGINES", (*providers.CLI_ENGINES, third))
    detect = providers.detect_engines
    verdict = providers.EngineVerdict(third.engine, third.title, False, "Gemini CLI is not installed")
    monkeypatch.setattr(providers, "detect_engines", lambda *args, **kwargs: [*detect(*args, **kwargs), verdict])
    text = README.read_text(encoding="utf-8")
    codex_row = re.search(r"^\| `codex` \|.*\n", text, re.MULTILINE)
    row = "| `gemini-cli` | Gemini CLI, installed and signed in | nothing |\n"
    readme = tmp_path / "readme.md"
    readme.write_text(text[:codex_row.end()] + row + text[codex_row.end():], encoding="utf-8")
    monkeypatch.setitem(globals(), "README", readme)
    with pytest.raises(AssertionError, match=re.escape(f"`gemini-cli` does not say it bills {third.subscription!r}")):
        test_readme_opening_lists_the_engines_providers_offers_and_names_the_spec()


def test_the_readme_billing_check_holds_each_cli_to_its_own_subscription(monkeypatch, tmp_path):
    """Review round 4, finding 1: the billing check took only the engine
    names from CLI_ENGINES, so any CLI row saying "subscription" passed, and
    a readme.md whose claude-code and codex rows swap their What it bills
    cells, Claude Code billing the ChatGPT plan, stayed green. Each CLI's
    cell must name the subscription providers.py says it runs on
    (`CliEngine.subscription`), so the swap fails at the first CLI's row."""
    text = README.read_text(encoding="utf-8")
    first, second = providers.CLI_ENGINES[:2]
    rows = [re.search(rf"^\| `{re.escape(cli.engine)}` \|.*$", text, re.MULTILINE).group(0)
            for cli in (first, second)]
    cells = [row.split("|") for row in rows]
    cells[0][3], cells[1][3] = cells[1][3], cells[0][3]
    for row, swapped in zip(rows, cells):
        text = text.replace(row, "|".join(swapped))
    readme = tmp_path / "readme.md"
    readme.write_text(text, encoding="utf-8")
    monkeypatch.setitem(globals(), "README", readme)
    with pytest.raises(AssertionError, match=re.escape(f"`{first.engine}` does not say it bills {first.subscription!r}")):
        test_readme_opening_lists_the_engines_providers_offers_and_names_the_spec()


def test_a_doc_whose_opening_says_macos_and_windows_does_not_still_offer_linux():
    """Review round 1, finding 2: card #491 made the openings say the
    platforms this ships on — "macOS and Windows", the owner's About — and
    dropped Linux from readme.md's Requirements. A page whose own opening
    says that may not, further down, still tell the reader the local engine
    is the backend "on Windows and Linux": both sentences are in the same
    file and only one of them can be true of what a user can install. Where
    Ollama itself runs is a different claim, made by docs/config.md and the
    modules, and is not this gate's business."""
    for doc in (README, BRIEF, REPO / "docs" / "architecture.md"):
        text = doc.read_text(encoding="utf-8")
        if "macOS and Windows" not in _opening(text):
            continue
        offers = [line.strip() for line in text.splitlines() if "Linux" in line]
        assert not offers, (
            f"{doc.name}'s opening says macOS and Windows, but it still offers Linux: {offers}")


def test_readme_requirements_say_what_each_engine_needs():
    """Review round 4, finding 3 (card #491, Done-when 2): readme.md's
    Requirements named only `mlx` and `ollama` and then said "Nothing else,
    for a user.", which is false on Windows: `mlx` cannot run there, and
    every other engine needs something the user brings, an Ollama server,
    an API key, a CLI installed and signed in. A Windows user with only what
    the section listed had no engine that runs. So Requirements names every
    engine the picker offers (`_picker`), with what it needs, and a user on
    either platform can see what makes at least one of them run."""
    section = _section(README.read_text(encoding="utf-8"), "Requirements")
    assert section is not None, "readme.md has no ## Requirements section"
    missing = [name for name, _ in _picker() if f"`{name}`" not in section]
    assert not missing, f"readme.md's Requirements do not say what {missing} need to run"


# A count, in digits or in the words a doc spells one out with.
NUMBER = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
# "Six engines", "(six engines)", "7 engines": a number right before the noun;
# and "one of two things", the count of ways to run one that readme.md's
# Windows section put before the three it lists; and "The other four bring
# their own", the engines left once two are named, with no noun after the
# number. "the two local engines" is not one: a subset qualified in place is
# named member by member in the same sentence, so it does not move with the
# total. Nor is "the other two routes": a plural noun after the number says
# what is counted, and when that noun is engines the first form has it.
ENGINE_COUNT = re.compile(
    rf"\b{NUMBER}\s+engines\b|\bone of {NUMBER}\b|\bthe other {NUMBER}\b(?!\s+[a-z]+s\b)",
    re.IGNORECASE,
)


def test_the_docs_card_491_rewrote_state_no_engine_count():
    """Review round 3, finding 1: readme.md, the brief and architecture said
    "six engines" in five places, and no gate read the number: changed to
    seven, four or three, every gate stayed green. How many engines there are
    is readme.md's table, which the gate above holds to providers.py; a count
    written anywhere else is wrong the day an engine is added, the reason card
    #437 took the test counts out. So these docs name the engines' shape and
    cite the table, and state no count of them, in prose, in the Status table
    or in the diagram.

    Review round 3, finding 2: readme.md's Windows section said the primary
    backend there "is one of two things" and then listed three — Ollama, a
    subscription CLI, the cloud. The same count, in the same place a new
    engine goes, so the same gate reads it.

    After round 3: readme.md's opening, having named the two local engines,
    said "The other four bring their own", a count of the rest that is
    wrong the day an engine is added, the same as the total."""
    for counted in ("Six engines, picked in the plugin's Settings dialog.",
                    "• VLM inference (six engines)", "7 engines behind one seam",
                    "the primary backend is one of two things.",
                    "The other four bring their own."):
        assert ENGINE_COUNT.search(counted), f"the gate misses a stated count: {counted!r}"
    for uncounted in ("On the two local engines no image leaves the machine: `mlx` and `ollama`.",
                      "`test_prompt_rejects_unapproved_context` cover the other two routes in."):
        assert not ENGINE_COUNT.search(uncounted), f"the gate calls this an engine count: {uncounted!r}"
    stated = [
        f"{doc.name}: {sentence}"
        for doc in (README, BRIEF, REPO / "docs" / "architecture.md")
        for sentence in _sentences(doc.read_text(encoding="utf-8"))
        if ENGINE_COUNT.search(sentence)
    ]
    assert not stated, f"docs state an engine count no gate checks; cite readme.md's table: {stated}"


def test_docs_name_engine_detection_where_the_default_and_the_refusal_are_described():
    """Card #404: the backend's default is now the first engine that can run
    here, and `--detect-engines` is how a user (and card #405's dialog) sees
    the verdicts. docs/config.md's `backend` row and readme.md's Windows
    section describe the default and the refusal, so both must name the flag,
    and neither may still promise that detection is yet to come."""
    config_doc = CONFIG_DOC.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    backend_row = _row(config_doc, "backend")
    assert "`--detect-engines`" in backend_row, "docs/config.md's backend row does not name --detect-engines"
    assert "turns that into" not in backend_row, "docs/config.md still says detection is yet to come"
    assert "`--detect-engines`" in readme, "readme.md does not name --detect-engines"


def test_brief_names_ollama_as_the_windows_executables_local_option():
    """Card #406: Ollama is the local engine on Windows, as readme.md § Windows,
    docs/architecture.md and docs/config.md say. The stack contract's build
    line describes the same executable, so it must say the same and may no
    longer call `dist/melampus.exe` cloud-only."""
    brief = BRIEF.read_text(encoding="utf-8")
    build = re.search(r"^- build: (.*?)(?=^- )", brief, re.MULTILINE | re.DOTALL)
    assert build, "docs/brief.md's stack contract has no build line"
    line = " ".join(build.group(1).split())
    assert "`dist/melampus.exe`" in line, "the build line does not name the Windows executable"
    assert "cloud engines only" not in line, "docs/brief.md still calls the Windows executable cloud-only"
    assert "Ollama" in line, "docs/brief.md's build line does not name Ollama as the Windows local option"


def test_docs_describe_the_engine_picker_and_where_the_key_lives():
    """Card #405: the engine has a control in Settings now. docs/plugin.md's
    engine section must describe the picker (what greys an engine, where the
    key goes: LrPasswords, never a file) instead of promising the dialog is
    yet to come, and readme.md's Lightroom section must show the dialog: the
    screenshot's reference, docs/settings-dialog.png, which the owner takes."""
    engine = _section(PLUGIN_DOC.read_text(encoding="utf-8"), "The engine")
    assert engine is not None, "docs/plugin.md has no ## The engine section"
    prose = " ".join(engine.split())
    assert "until then the preference is unset" not in prose, (
        "docs/plugin.md still says the engine has no control in Settings")
    for named in ("`--detect-engines`", "LrPasswords", "settings-dialog.png"):
        assert named in prose, f"docs/plugin.md's engine section does not name {named}"
    section = _section(README.read_text(encoding="utf-8"), "Reviewing in Lightroom")
    assert section is not None, "readme.md has no ## Reviewing in Lightroom section"
    assert "docs/settings-dialog.png" in section, (
        "readme.md's Lightroom section does not show the settings dialog")
    # The section is read on Windows too, which has no keychain: where the
    # key is kept is said in platform-neutral words, as the dialog says it.
    assert "keychain" not in section.lower(), (
        "readme.md's Lightroom section says keychain, which Windows has not")


def test_docs_describe_the_cli_engines_in_the_picker():
    """Card #423: the two subscription CLIs are in the picker. docs/plugin.md's
    engine section names them, says they take no key and bill to the
    subscription, and where their titles come from (the verdict); readme.md's
    Lightroom section lists them among what the picker offers; docs/config.md's
    backend row, readme.md and the engine section no longer defer the picker
    to a card yet to come. Both places that send a reader to
    docs/settings-dialog.png, which the owner takes and this card's picker has
    outgrown, carry the note the Download row already has, in the same words:
    docs/plugin.md's engine section, which links it (review round 9, finding
    3), and readme.md's Lightroom section, which embeds it (review round 10,
    finding 1)."""
    plugin_doc = PLUGIN_DOC.read_text(encoding="utf-8")
    engine = _section(plugin_doc, "The engine")
    assert engine is not None, "docs/plugin.md has no ## The engine section"
    prose = " ".join(engine.split())
    for named in ("`claude-code`", "`codex`", "subscription", "no API key", "`title`"):
        assert named in prose, f"docs/plugin.md's engine section does not say {named}"
    # The picture the section sends a reader to is card #405's four-engine
    # dialog. Its Download row already carries the note for exactly this
    # (### The Download row); the picker's own prose had none, so a reader
    # went from updated words to a picture of the old dialog.
    picker = " ".join(engine.split("### The Download row")[0].split())
    assert "`docs/settings-dialog.png` predates" in picker, (
        "docs/plugin.md's engine section sends a reader to a picture of the old dialog "
        "without saying the screenshot predates the CLI engines")
    readme = README.read_text(encoding="utf-8")
    section = _section(readme, "Reviewing in Lightroom")
    assert section is not None, "readme.md has no ## Reviewing in Lightroom section"
    for named in ("claude-code", "codex"):
        assert named in section, f"readme.md's Lightroom section does not offer {named}"
    # readme.md embeds that same picture, directly under the paragraph this
    # card edited, so a reader meets the old four-engine dialog there too.
    assert "`docs/settings-dialog.png` predates" in section, (
        "readme.md's Lightroom section embeds a picture of the old dialog without "
        "saying the screenshot predates the CLI engines")
    config_doc = CONFIG_DOC.read_text(encoding="utf-8")
    backend_row = _row(config_doc, "backend")
    for text in (backend_row, readme, prose):
        assert "learns" not in text or "#423" not in text, "still defers the picker to card #423"


def test_docs_name_the_download_command_where_the_model_and_the_protocol_are_described():
    """Card #407: the model arrives by `melampus-id --download-model`, not by
    a manual `hf download` the user must read the readme for. readme.md's
    Install section names the flag and no longer the manual line (the
    environment note moves to docs/troubleshooting.md, which keeps `hf
    download` as the diagnosis it is); docs/config.md documents the flag,
    its exit codes and the progress protocol the plugin parses, in the words
    the code prints, and architecture.md's module table has the module."""
    from melampus.download import CANCELLED, DONE, EXIT_CANCELLED, PROGRESS

    readme = README.read_text(encoding="utf-8")
    install = re.search(r"^## Install\n(.*?)^## ", readme, re.MULTILINE | re.DOTALL).group(1)
    assert "`--download-model`" in install or "--download-model" in "\n".join(_fenced_commands(install)), (
        "readme.md § Install does not name --download-model")
    assert "hf download" not in install, "readme.md § Install still tells the user to run hf download by hand"

    config_doc = CONFIG_DOC.read_text(encoding="utf-8")
    section = re.search(r"^## Downloading the model\n(.*?)(?:^## |\Z)", config_doc, re.MULTILINE | re.DOTALL)
    assert section, "docs/config.md has no `## Downloading the model` section"
    for promise in ("`--download-model`", f"`{PROGRESS} <bytes_done> <bytes_total>`", f"`{DONE} <path>`",
                    f"`{CANCELLED}`", f"exit {EXIT_CANCELLED}", "exit 3", "exit 0", "stderr", "resume", "checksum"):
        assert promise in section.group(1), f"docs/config.md § Downloading the model does not say {promise}"

    architecture = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "| `download.py` |" in architecture, "docs/architecture.md's module table lacks download.py"


def test_docs_say_the_same_button_and_flags_pull_ollamas_model():
    """Card #409: docs/config.md § Downloading the model says the three flags
    act for the picked engine and, for ollama, through which of Ollama's
    endpoints (the ones the code calls); the `ollama_model` row no longer
    tells the user to pull by hand; docs/plugin.md's Download row section
    covers ollama; readme.md § Windows says the model can be pulled from
    Settings."""
    from melampus.download import OLLAMA_DELETE, OLLAMA_PULL, OLLAMA_TAGS

    config_doc = CONFIG_DOC.read_text(encoding="utf-8")
    section = re.search(r"^## Downloading the model\n(.*?)(?:^## |\Z)", config_doc, re.MULTILINE | re.DOTALL).group(1)
    for promise in ("`--backend ollama`", "`[model] backend`", f"`{OLLAMA_PULL}`", f"`{OLLAMA_TAGS}`",
                    f"`{OLLAMA_DELETE}`", "`done <model>`", "`ollama_model`", "`ollama_url`", "size unknown"):
        assert promise in section, f"docs/config.md § Downloading the model does not say {promise}"
    row = re.search(r"^\| `ollama_model` \|.*$", config_doc, re.MULTILINE).group(0)
    assert "yours to do" not in row and "until card #409" not in row, "docs/config.md still says to pull by hand"
    assert "Download" in row, "docs/config.md's ollama_model row does not point at the Download button"

    plugin_doc = (REPO / "docs" / "plugin.md").read_text(encoding="utf-8")
    prose = re.search(r"^### The Download row\n(.*?)(?:^## |\Z)", plugin_doc, re.MULTILINE | re.DOTALL)
    assert prose, "docs/plugin.md has no `### The Download row` section"
    for named in ("ollama", "`--backend`", "pull", "size unknown"):
        assert named in prose.group(1), f"docs/plugin.md's Download row section does not say {named}"

    readme = README.read_text(encoding="utf-8")
    windows = re.search(r"^## Windows.*?\n(.*?)^## ", readme, re.MULTILINE | re.DOTALL).group(1)
    assert "Settings" in windows and "pull" in windows, "readme.md § Windows does not say the model can be pulled from Settings"


def test_config_doc_names_the_command_output_ceiling():
    """Card #420: the `command` backend reads stdout and stderr with a
    ceiling (`CommandBackend.MAX_OUTPUT_BYTES`), past which the program is
    stopped and the frame recorded as an error naming the number of bytes.
    docs/config.md's `command` row names the other per-frame outcomes, so it
    must name this one with the number the error message carries, and must
    change when the number does."""
    from melampus.backend import CommandBackend
    config_doc = CONFIG_DOC.read_text(encoding="utf-8")
    command_row = _row(config_doc, "command")
    assert str(CommandBackend.MAX_OUTPUT_BYTES) in command_row, (
        "docs/config.md's command row does not name the output ceiling in bytes")
    assert "4 MiB" in command_row, "docs/config.md's command row does not name the output ceiling"


def test_config_doc_says_the_commands_exit_ends_its_answer_and_stops_what_it_started():
    """Card #420: the command's exit ends its answer, and everything it
    started is stopped the moment it exits, so a helper it leaves holding
    stdout or stderr is stopped rather than waited on; past the timeout the
    program and everything it started are stopped too. An operator whose
    CLI starts a helper meant to outlive the call (a server it keeps warm)
    finds it stopped after every completion, so docs/config.md's `command`
    row must say so, and its `timeout_seconds` row must say the stop
    reaches everything the program started, not the program alone."""
    model = _section(CONFIG_DOC.read_text(encoding="utf-8"), "`[model]`")  # [escalation] has a timeout_seconds row of its own
    command_row, timeout_row = _row(model, "command"), _row(model, "timeout_seconds")
    assert "exit ends its answer" in command_row, (
        "docs/config.md's command row does not say the command's exit ends its answer")
    assert "everything it started is stopped" in command_row, (
        "docs/config.md's command row does not say everything the command started is stopped at its exit")
    assert "not waited" in command_row, (
        "docs/config.md's command row does not say a helper left holding a stream is stopped, not waited on")
    assert "and everything it started" in timeout_row, (
        "docs/config.md's timeout_seconds row does not say the stop reaches everything the program started")
    assert "counted from before the program is started" in timeout_row, (
        "docs/config.md's timeout_seconds row does not say starting the program counts against the ceiling")


def test_docs_name_the_sigchld_refusal_beside_the_commands_other_refusals():
    """Card #420: docs/config.md's `command` row, docs/architecture.md,
    readme.md and `CommandBackend`'s docstring each enumerate what the
    factory refuses up front (a program not on PATH, a `.cmd`/`.bat`
    shim), so each must also name the launcher that ignores SIGCHLD,
    refused the same way because the kernel would reap the program at its
    exit and the pid its tree is stopped by could be someone else's by
    then; the row must say the fix (a shell, or the default), and the
    docstring must say that refusal is what `_stop_tree` rests on."""
    from melampus.backend import CommandBackend
    command_row = _row(CONFIG_DOC.read_text(encoding="utf-8"), "command")
    for doc, prose in (
        ("docs/architecture.md", (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")),
        ("docs/config.md", command_row),
        ("readme.md", README.read_text(encoding="utf-8")),
        ("CommandBackend's docstring", CommandBackend.__doc__),
    ):
        prose = " ".join(prose.split())  # the prose wraps; the phrase must not hide across a line break
        assert "ignores SIGCHLD" in prose, f"{doc} does not name the SIGCHLD refusal beside the command's other refusals"
    assert "shell" in command_row and "default" in command_row, (
        "docs/config.md's command row does not say how to fix a launcher that ignores SIGCHLD")
    assert "_stop_tree" in CommandBackend.__doc__, (
        "CommandBackend's docstring does not say the SIGCHLD refusal is what _stop_tree rests on")


def test_docs_say_the_command_runs_once_per_completion():
    """Card #420: the `command` backend runs its program once per
    completion, not once per frame: a frame is at least two completions
    (the routing prompt, then the group's identification prompt), and a
    corrective retry or a step down the fallback ladder is another. Each
    of docs/architecture.md, docs/config.md's `backend` row and readme.md
    must say so where it describes the engine, and name the two stages."""
    for doc, prose in (
        ("docs/architecture.md", (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")),
        ("docs/config.md", _row(CONFIG_DOC.read_text(encoding="utf-8"), "backend")),
        ("readme.md", README.read_text(encoding="utf-8")),
    ):
        prose = " ".join(prose.split())  # the prose wraps; the phrase must not hide across a line break
        assert "once per frame" not in prose, f"{doc} still says the command runs once per frame"
        assert "once per completion" in prose, f"{doc} does not say the command runs once per completion"
        assert "routing" in prose and "identification" in prose, (
            f"{doc} does not name the two completions a frame is made of")


@pytest.mark.parametrize(
    ("engine", "must_say"),
    [
        ("claude-code", ("subscription",)),
        ("codex", ("usage limit", "bills per call", "Tricolored Heron")),
    ],
    ids=["claude-code", "codex"],
)
def test_config_doc_quotes_the_cli_template_from_its_one_source(engine, must_say):
    """Cards #421 and #422: one place holds each CLI's template, the
    CliEngine's `command`. docs/config.md quotes it as the `[model] command`
    a user would set to override it, in a TOML block that parses to exactly
    that list, so the doc cannot rot into a second copy; and it says what
    to install, how to sign in, and what is that CLI's own (`must_say`):
    that Claude Code's runs bill to the subscription; that Codex's stop at
    the plan's usage limit, that an API-key sign-in bills per call and is
    refused, and what the real success run answered on the committed
    fixture (a measurement, not a status that goes stale). The readme and
    the architecture doc name the engine and the template's one source."""
    import tomllib

    (cli,) = [c for c in providers.CLI_ENGINES if c.engine == engine]
    text = CONFIG_DOC.read_text(encoding="utf-8")
    blocks = [
        block for block in re.findall(r"```toml\n(.*?)```", text, re.DOTALL)
        if f'backend = "{cli.engine}"' in block
    ]
    assert blocks, f"docs/config.md has no ```toml block with backend = \"{cli.engine}\""
    (block,) = blocks
    assert tomllib.loads(block)["model"]["command"] == cli.command
    for said in (cli.install, f"`{cli.sign_in}`", *must_say):
        assert said in text, f"docs/config.md does not say {said!r}"
    readme = (REPO / "readme.md").read_text(encoding="utf-8")
    assert f"`{cli.engine}`" in readme and f"--backend {cli.engine}" in readme
    architecture = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    source = f"{cli.engine.upper().replace('-', '_')}_COMMAND"
    assert f"`{cli.engine}`" in architecture and source in architecture


@pytest.mark.parametrize("cli", providers.CLI_ENGINES, ids=lambda cli: cli.engine)
def test_config_doc_says_what_environment_the_cli_is_launched_with(cli):
    """Codex review round 3, S1: a CLI engine's status check and runs are
    launched with the CLI's own environment (`providers.CLI_ENVIRONMENT`
    and its settings variable), never melampus's, so a photograph's text
    cannot have an agent that runs commands read the shell's exports into
    its cloud conversation. Each CLI's section of docs/config.md says so,
    names the settings variable that does reach it, and names the one
    source of the list. One case per CliEngine, from `CLI_ENGINES` itself
    (review round 8, C2), and the section's heading is the CliEngine's
    `title`, the word the refusals print (review round 7, C4), so a third
    CLI needs no entry here; the `command` row says the user's own program
    still gets melampus's environment as it is, since that seam is the
    user's; the architecture doc names the mechanism."""
    text = CONFIG_DOC.read_text(encoding="utf-8")
    section = re.search(rf"^### {re.escape(cli.title)}\n(.*?)(?=^### |^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert section, f"docs/config.md has no ### {cli.title} section"
    for said in ("environment", f"`{cli.settings_variable}`", "`providers.CLI_ENVIRONMENT`"):
        assert said in section.group(1), f"docs/config.md § {cli.title} does not say {said!r}"
    command_row = _row(text, "command")
    assert "environment" in command_row, "docs/config.md's command row does not say what environment the program gets"
    architecture = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "CLI_ENVIRONMENT" in architecture, "docs/architecture.md does not name the CLI environment"


def test_config_doc_says_the_codex_profile_leaves_the_shared_temp_directories_writable():
    """Security review round 9, S1: the permission profile `CODEX_COMMAND`
    carries denies writes everywhere it governs except the shared temp
    directories `":minimal" = "read"` grants. Measured on codex-cli 0.155.1
    with `codex sandbox -P` under that exact profile, no model call: a
    command wrote to `/tmp`, `/private/tmp`, `/var/tmp` and
    `/private/var/tmp`, made one of them executable and ran it, and the
    files were still on disk outside the sandbox afterwards, while a write
    in the staged folder and in the home folder was "Operation not
    permitted". So a photograph's text has a channel that outlives the
    frame `--ephemeral` ends. The suite never runs the real Codex, so the
    prose is what can be pinned: docs/config.md must name those paths as
    writable and must not say the profile writes nowhere.

    Review round 10, C1: the sentence that leaves the decision to the owner
    must also say where the decision is recorded, so a reader can follow it.
    It pointed at "its own card" without a number; the card did not exist
    yet and this branch did not invent a number, so what it named was the
    record that did exist, security review round 9 on PR #17. The negative
    keeps the prose from going back to an unnamed card.

    The owner has since filed that card, #505, so the sentence names it as
    well: security review round 9 on PR #17 is where the measurement is,
    card #505 is where the decision is recorded, and the prose no longer
    says the card is his to file."""
    prose = " ".join(CONFIG_DOC.read_text(encoding="utf-8").split())
    for path in ("`/tmp`", "`/private/tmp`", "`/var/tmp`", "`/private/var/tmp`"):
        assert path in prose, f"docs/config.md does not name {path} under the profile"
    for said in ("writable", "execute what it wrote", "from one frame to the next"):
        assert said in prose, f"docs/config.md does not say {said!r} of the shared temp directories"
    assert "no write anywhere" not in prose, (
        "docs/config.md still says the profile writes nowhere; writes land in the shared temp directories")
    deferral = re.search(r"Whether that is acceptable[^.]*\.", prose)
    assert deferral, "docs/config.md no longer says whose call the shared temp writes are"
    for said in ("the owner's call", "security review round 9", "PR #17", "card #505"):
        assert said in deferral.group(0), (
            f"docs/config.md does not say {said!r} where it leaves the shared temp writes to the owner")
    assert "own card" not in prose, (
        "docs/config.md defers the decision to a card it does not name; name where the decision is recorded")


def test_config_doc_says_the_staged_folder_sits_outside_the_shared_temp_directories():
    """Security review round 12, S1: the profile's read boundary is the
    staged folder, and it holds only because of where that folder is made.

    `":minimal" = "read"` grants /tmp (with /private/tmp, /var/tmp and
    /private/var/tmp) whole and writable, and `tempfile` falls back to /tmp
    whenever $TMPDIR is unset — ordinary on Linux, in a container and under
    a cleared environment. A staged folder placed by $TMPDIR alone would sit
    inside the grant on exactly those machines, and the profile's central
    property, that an injection in a photograph cannot read past the one
    staged file, would be void there: measured on codex-cli 0.155.1 with
    `codex sandbox -P` under this profile, workspace root in /tmp, a file in
    another /tmp folder was read, `ls /tmp` listed the directory and the
    staged image itself was overwritten and read back.

    `images.staged_pixels` stages under melampus's own directory instead
    (`images.STAGING_ROOT`, the one `config.cache_file` names), which the
    same measurement refuses in a checkout and in the executable's layout
    alike, and test_pipeline.py pins the code. The suite never runs the real
    Codex, so what docs/config.md can be held to is the dependence it must
    not leave unstated: the doc has to say where the staged folder is made
    and why it is not $TMPDIR."""
    prose = " ".join(CONFIG_DOC.read_text(encoding="utf-8").split())
    # The sentence ends at a full stop followed by a space; the dots inside
    # `images.staged_pixels` and `config.cache_file` are not sentence ends.
    staging = re.search(r"That read boundary.*?\.(?=\s|$)", prose)
    assert staging, "docs/config.md does not say where the staged folder is made"
    for said in ("`images.staged_pixels`", "`config.cache_file`", "not under `$TMPDIR`"):
        assert said in staging.group(0), (
            f"docs/config.md does not say {said!r} where it says where the staged folder is made")
    for said in ("`$TMPDIR` is unset", "overwritten and read back"):
        assert said in prose, (
            f"docs/config.md does not say {said!r} of a staged folder left to $TMPDIR")
    # Security review round 9 (a later round, same line): melampus's own
    # directory is not by itself outside the grant. It follows the checkout
    # root in a checkout and $XDG_DATA_HOME/Melampus inside the executable,
    # and a checkout under /tmp or a frozen run with that variable pointed
    # there puts the staged folder back inside the grant, measured the same
    # way. `images.staging_root` resolves that root and refuses it, so the
    # doc must not leave the dependence on where the root lands unstated.
    for said in ("`$XDG_DATA_HOME`", "`images.staging_root`", "refuses to stage"):
        assert said in prose, (
            f"docs/config.md does not say {said!r} of a staging root that lands in the grant")
    # Codex review round 10, S1: resolving the root is not the whole of the
    # check either. A Mac's boot volume is case-insensitive, so /private/TMP
    # and /private/tmp are one directory (measured: os.path.samefile says so)
    # while `Path.resolve` keeps the case it was handed, and a root spelled
    # that way passed a path comparison. The check compares filesystem
    # identity, so the doc must not describe it as a comparison of paths.
    for said in ("case-insensitive", "filesystem identity"):
        assert said in prose, (
            f"docs/config.md does not say {said!r} of how a staging root is judged")


def test_the_docs_say_a_claude_code_key_comes_from_the_settings_not_the_environment():
    """Review round 8, C1 (review round 7, C2's defect in one more place):
    melampus's own environment never reaches Claude Code
    (`CliEngine.environment`), so an API key its status check reports cannot
    have come from the shell melampus was started in (only from the `env`
    block of a settings file Claude Code loads), and the refusal names what
    to remove and from where (`CLAUDE_CODE_CREDENTIAL_FIX["api_key"]`, "remove
    ANTHROPIC_API_KEY from the `env` block of the settings", pinned in
    test_providers.py). Both prose docs that describe that refusal must say
    the same: neither may put the key in the environment or tell the user to
    unset it."""
    for doc, prose in (
        ("readme.md", README.read_text(encoding="utf-8")),
        ("docs/config.md", CONFIG_DOC.read_text(encoding="utf-8")),
    ):
        prose = " ".join(prose.split())  # the prose wraps; the phrase must not hide across a line break
        assert "`env` block of a settings file" in prose, (
            f"{doc} does not say such a key can only come from the `env` block of a settings file")
        assert "what to remove" in prose, f"{doc} does not say the refusal names what to remove"
        assert "key in the environment" not in prose, f"{doc} still puts the key in melampus's own environment"
        assert "what to unset" not in prose, f"{doc} still says the refusal names what to unset"


def test_config_doc_command_row_says_where_the_program_runs():
    """Security round 2 (S2) changed the command seam's contract for every
    program, not only Claude Code: the program runs with the staged image's
    temporary folder as its working directory, and one named by a relative
    path is made absolute against the folder melampus was launched from
    first. A user's own program that reads a file beside itself through `.`,
    or writes a log there, now does so in a folder that is deleted after the
    frame, so the `command` row itself must say so (the Claude Code section
    is not where a `command` user looks): where the program runs, and how to
    name it."""
    config_doc = CONFIG_DOC.read_text(encoding="utf-8")
    command_row = next(line for line in config_doc.splitlines() if line.startswith("| `command` |"))
    for said in ("working directory", "PATH", "absolute path", "launched from"):
        assert said in command_row, f"docs/config.md's command row does not say {said!r}"


# "182 tests", "46 rules tests", "11 corpus-backed tests", "410 passed",
# "26 skipped", "skips 2": a number and a test noun, with at most one word between.
TEST_COUNT = re.compile(
    r"\b\d+\s+(?:[\w-]+\s+)?(?:tests?|passed|skipped|skips?)\b|\b(?:skips?|skipped)\s+\d+\b",
    re.IGNORECASE,
)
# "126 Python, 56 Lua": a language count, stale only in a sentence about tests.
LANGUAGE_COUNT = re.compile(r"\b\d+\s+(?:Python|Lua)\b")


def _sentences(text: str) -> list[str]:
    """Each sentence of the doc's prose, with hard-wrapped lines joined so a
    count and its noun are seen together whichever line each falls on."""
    return [
        sentence
        for paragraph in re.split(r"\n\s*\n", text)
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(paragraph.split()))
        if sentence
    ]


def _states_a_test_count(sentence: str) -> bool:
    """Whether `sentence` states how many tests there are: a number and a test
    noun, or a count of a language in a sentence that is about tests."""
    return bool(TEST_COUNT.search(sentence) or (
        "test" in sentence.lower() and LANGUAGE_COUNT.search(sentence)
    ))


def test_docs_state_no_test_count():
    """Card #437, Done-when 1: the suite grows with every card, so a count
    written into a doc is wrong the day after, and CI checks no such number.
    The docs say what the tests need and how to run them, never how many
    there are, in any of the forms a count has taken: "182 tests", "46 rules
    tests", "410 passed", "26 skipped", "skips 2", and "126 Python, 56 Lua"
    in a sentence about tests."""
    stated = []
    for doc in DOCS:
        for sentence in _sentences(doc.read_text(encoding="utf-8")):
            if _states_a_test_count(sentence):
                stated.append(f"{doc.relative_to(REPO)}: {sentence}")
    assert not stated, f"docs state a test count that CI does not check: {stated}"


def test_the_test_count_gate_reads_every_form_of_a_count():
    """Round 1, finding 1: the gate above is green today only because no doc
    states a count, so on its own it would stay green if a narrowing edit to
    either pattern, or a `_sentences` that stopped joining hard-wrapped lines,
    took the promise away. Each form the gate names must be caught in the
    hard-wrapped prose the docs are written in, and a count-free sentence
    must not be."""
    wrapped = "The suite is 182\ntests today.\n\nRun them with pytest.\n"
    assert _sentences(wrapped) == ["The suite is 182 tests today.", "Run them with pytest."]
    for counted in ("The suite is 182 tests today.", "The 47 rules tests cover every rule.",
                    "CI reports 410 passed.", "CI reports 26 skipped.", "The run skips 2 on Linux.",
                    "The tests are 126 Python, 56 Lua.", "The tests are 126 Python.",
                    "The tests are 56 Lua."):
        assert _states_a_test_count(counted), f"the gate misses a stated count: {counted!r}"
    for uncounted in ("Run the tests with pytest before you push.", "The plugin is 126 Python."):
        assert not _states_a_test_count(uncounted), f"the gate calls this a test count: {uncounted!r}"


def test_the_test_count_gate_reads_every_doc(tmp_path, monkeypatch):
    """Round 1, finding 2: `DOCS` (:54) is the list every doc-wide gate in
    this file iterates, and a doc written later must be inside this gate by
    default rather than outside it, so the gate reads `DOCS` itself rather
    than a re-listed subset of it. The folder is a stand-in read at call
    time.

    Round 2, finding 2: it holds two docs, not one, because one doc cannot
    tell the two subsets apart. A gate re-listing paths of its own reads the
    real docs, which are clean, and raises nothing; a gate reading part of
    `DOCS` — the shape round 1 had — reads only part of the stand-in folder.
    With one doc, any gate that read it at all raised, and this stayed green
    while the rest of `DOCS` sat outside the gate.

    Round 3, finding 2: both docs state a count and the failure must name
    both. With the count in the later doc alone, a gate reading `DOCS[1:]` —
    readme.md outside it — still raised on `later.md` and this still passed,
    so the test proved only that the last doc is read. Naming both means a
    gate reading any proper subset of `DOCS` misses one of the two and fails
    here."""
    (tmp_path / "earlier.md").write_text("The suite is 182 tests today.\n", encoding="utf-8")
    (tmp_path / "later.md").write_text("CI reports 410 passed.\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "REPO", tmp_path)
    monkeypatch.setitem(globals(), "DOCS", [tmp_path / "earlier.md", tmp_path / "later.md"])
    with pytest.raises(AssertionError) as raised:
        test_docs_state_no_test_count()
    for stated in ("earlier.md: The suite is 182 tests today.",
                   "later.md: CI reports 410 passed."):
        assert stated in str(raised.value), (
            f"the gate read a subset of DOCS: its failure does not name {stated!r}: "
            f"{raised.value}"
        )
