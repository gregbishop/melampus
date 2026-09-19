"""Docs drift gates.

Each test here makes one promise a doc carries mechanical, so the doc cannot
silently fall behind the code or the repo again (as happened when [occurrence] and
[quality] shipped undocumented). The promises: docs/config.md names every
implemented setting; AGENTS.md, and not .gitignore, names the install command for
the recorded plugins; no doc names a file by an uppercase name it does not have;
docs/brief.md names the pytest command CI actually runs and explains it as
installing from the lockfile; every doc block that installs the service, and CI, install
from the lockfile (card #425, Done-when 3 and 2); AGENTS.md points at
docs/brief.md without restating its values; AGENTS.md points at the standard
and names the tracker (card #410, Done-when 3); and the Windows job runs the
plugin tests, so the command built for cmd.exe is run by cmd.exe (card #401,
Done-when 3).

The checks are deliberately dumb — substring presence of the backticked name — so
they never argue with prose style, only with absence.
"""

import json
import re
from pathlib import Path

from melampus.config import MelampusConfig

REPO = Path(__file__).resolve().parents[2]
CONFIG_DOC = REPO / "docs" / "config.md"
AGENTS_MD = REPO / "AGENTS.md"
PLUGIN_CHOICE = REPO / ".agents" / "on-purpose.json"
BRIEF = REPO / "docs" / "brief.md"
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
GITIGNORE = REPO / ".gitignore"
README = REPO / "readme.md"
DOCS = [README, AGENTS_MD, *sorted((REPO / "docs").glob("*.md"))]


LOCKFILE_FLAGS = ("--locked", "--frozen")


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


def test_agents_md_names_the_install_command_and_gitignore_does_not_restate_it():
    """The install outputs (.claude/settings.json, .agents/skills, .codex/agents)
    are machine-local and untracked; a fresh clone must be told how to regenerate
    them, with the same plugins .agents/on-purpose.json records. AGENTS.md is the
    one place that says so: .gitignore, which lists those outputs, points there
    rather than restating the command, so a plugin added later moves one file."""
    plugins = json.loads(PLUGIN_CHOICE.read_text(encoding="utf-8"))["plugins"]
    command = f"node ~/on-purpose/bin/install.mjs {' '.join(plugins)}"
    assert f"`{command}`" in AGENTS_MD.read_text(encoding="utf-8"), (
        f"AGENTS.md must tell a fresh clone to run `{command}` "
        "(the plugins recorded in .agents/on-purpose.json)"
    )
    gitignore = GITIGNORE.read_text(encoding="utf-8")
    assert "install.mjs" not in gitignore, (
        ".gitignore restates the install command that AGENTS.md is gated for; "
        "say the installer regenerates the ignored outputs and point at AGENTS.md"
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


def _ci_pytest_commands() -> list[str]:
    """The `run:` line of every ci.yml step that invokes pytest."""
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    commands = [
        command
        for command in re.findall(r"^\s*run:\s*(.+?)\s*$", workflow, re.MULTILINE)
        if "pytest" in command
    ]
    assert commands, "ci.yml runs no pytest step"
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
    install = re.search(r"^## Install\n(.*?)^## ", readme, re.MULTILINE | re.DOTALL)
    assert install and any(_installs_from_the_lockfile(c) for c in _fenced_commands(install.group(1))), (
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
    are never checked where it counts."""
    not_building = [
        c for c in _ci_pytest_commands() if "--build-binary" not in c or "--extra build" not in c
    ]
    assert not not_building, (
        "CI's pytest step must install `--extra build` and run `pytest --build-binary`: "
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
    `build:` line and readme.md must name both, or nobody finds them."""
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


def _windows_job() -> str:
    """The text of ci.yml's job on a Windows runner."""
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    jobs = re.split(r"^  (?=\w[\w-]*:\s*$)", workflow.split("\njobs:\n", 1)[1], flags=re.MULTILINE)
    windows = [job for job in jobs if re.search(r"runs-on: windows-", job)]
    assert windows, "ci.yml has no job on a Windows runner"
    return windows[0]


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
    pytest_steps = [c for c in _ci_pytest_commands() if c in job]
    assert pytest_steps, "the Windows job runs no pytest step"
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
    installed by a step of the job, and the file on its pytest line."""
    job = _windows_job()
    installs_lua = [
        line for line in job.splitlines()
        if not line.strip().startswith("#") and "install" in line and re.search(r"\blua\b", line)
    ]
    assert installs_lua, "the Windows job installs no Lua interpreter, so the plugin tests skip there"
    pytest_steps = [c for c in _ci_pytest_commands() if c in job]
    assert pytest_steps and all("tests/test_lua_plugin.py" in c for c in pytest_steps), (
        "the Windows job's pytest step must run tests/test_lua_plugin.py, so the "
        f"command the plugin builds for cmd.exe is run by cmd.exe: {pytest_steps}"
    )


def test_ci_pins_every_pip_install_to_an_exact_version():
    """Security: a tool CI installs with pip outside the lockfile (uv, on the
    Windows runner) is fetched from PyPI at build time and then produces the
    executable that is uploaded as an artifact, so `pip install <name>` with no
    `==` runs whatever PyPI serves that day. Every pip install in ci.yml names
    an exact version."""
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    pip_installs = re.findall(r"^\s*run:.*\bpip install\b(.*?)\s*$", workflow, re.MULTILINE)
    assert pip_installs, "ci.yml has no pip install step"
    unpinned = [
        requirement
        for arguments in pip_installs
        for requirement in arguments.split()
        if not requirement.startswith("-") and not re.fullmatch(r"[\w.\-\[\]]+==[\w.]+", requirement)
    ]
    assert not unpinned, f"CI installs from PyPI without an exact version: {unpinned}"
