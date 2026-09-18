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
#434, Done-when 1 and 3); and the release workflow builds with the commands CI
uses and attaches one zip per platform, which readme.md's Lightroom section
names (card #402).

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

from melampus.config import MelampusConfig

REPO = Path(__file__).resolve().parents[2]
CONFIG_DOC = REPO / "docs" / "config.md"
AGENTS_MD = REPO / "AGENTS.md"
PLUGIN_CHOICE = REPO / ".agents" / "on-purpose.json"
INSTALLED_SKILLS = REPO / ".agents" / "skills"
BRIEF = REPO / "docs" / "brief.md"
WORKFLOWS = REPO / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS / "ci.yml"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"
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


def _pytest_commands(workflow: Path) -> list[str]:
    """The `run:` line of every step in that workflow that invokes pytest."""
    text = workflow.read_text(encoding="utf-8")
    commands = [
        command
        for command in re.findall(r"^\s*run:\s*(.+?)\s*$", text, re.MULTILINE)
        if "pytest" in command
    ]
    assert commands, f"{workflow.name} runs no pytest step"
    return commands


def _ci_pytest_commands() -> list[str]:
    return _pytest_commands(CI_WORKFLOW)


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


def _readme_section(heading: str) -> str:
    """The text of readme.md's `## {heading}` section, up to the next `## `."""
    readme = README.read_text(encoding="utf-8")
    section = re.search(rf"^## {re.escape(heading)}\n(.*?)^## ", readme, re.MULTILINE | re.DOTALL)
    assert section, f"readme.md has no ## {heading} section"
    return section.group(1)


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
    install = _readme_section("Install")
    assert any(_installs_from_the_lockfile(c) for c in _fenced_commands(install)), (
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
    whose `--backend anthropic` prints an install hint that means nothing
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


def test_readme_build_blocks_sync_the_sdk_extras():
    """Card #434: the executable carries the cloud SDKs, and PyInstaller bundles
    what the build venv has, so every `uv sync` in readme.md's build section
    (the macOS block and the Windows one) names the build, cloud and openai
    extras."""
    section = _readme_section("Building the executable")
    syncs = [c for c in _fenced_commands(section) if re.search(r"\buv sync\b", c)]
    assert syncs, "readme.md's build section has no uv sync command"
    without = _lacking_build_extras(syncs)
    assert not without, (
        f"readme.md's build section must sync {' '.join(BUILD_EXTRAS)}, or the executable "
        f"it builds lacks the SDKs: {without}"
    )


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
    pytest_steps = [c for c in _ci_pytest_commands() if c in job]
    assert pytest_steps and all("tests/test_lua_plugin.py" in c for c in pytest_steps), (
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


def test_ci_pins_every_pip_install_to_an_exact_version():
    """Security: a tool CI installs with pip outside the lockfile (uv, on the
    Windows runner) is fetched from PyPI at build time and then produces the
    executable that is uploaded as an artifact, so `pip install <name>` with no
    `==` runs whatever PyPI serves that day. Every pip install in every
    workflow names an exact version: ci.yml's executable is an artifact,
    release.yml's is what ships (card #402)."""
    pip_installs = [
        (workflow.name, arguments)
        for workflow in sorted(WORKFLOWS.glob("*.yml"))
        for arguments in re.findall(
            r"^\s*run:.*\bpip install\b(.*?)\s*$", workflow.read_text(encoding="utf-8"), re.MULTILINE
        )
    ]
    assert {name for name, _ in pip_installs} >= {CI_WORKFLOW.name, RELEASE_WORKFLOW.name}, (
        f"a workflow has no pip install step: {pip_installs}"
    )
    unpinned = [
        f"{name}: {requirement}"
        for name, arguments in pip_installs
        for requirement in arguments.split()
        if not requirement.startswith("-") and not re.fullmatch(r"[\w.\-\[\]]+==[\w.]+", requirement)
    ]
    assert not unpinned, f"a workflow installs from PyPI without an exact version: {unpinned}"


RELEASE_ZIPS = ("Melampus-macOS.zip", "Melampus-Windows.zip")


def test_release_workflow_builds_as_ci_does_and_attaches_a_zip_per_platform():
    """Card #402, Done-when 1: given a tag is pushed, when the release workflow
    runs, then a GitHub release exists with Melampus-macOS.zip and
    Melampus-Windows.zip attached. The proof is the first tagged run; this
    gate keeps the workflow honest before it: it triggers on v* tags, its
    pytest steps are exactly CI's (the same sync and --build-binary on each
    runner, so what ships is what was tested, and the docs gates above cover
    both), the zips come from tools/package_plugin.py (the one place that
    knows the layout; no second copy in YAML), both zip names are in it, the
    token gets `contents: write` and no other scope, and every third-party
    action is pinned to a commit SHA with the version in a trailing comment."""
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"^on:\n\s+push:\n\s+tags:\s*\[\s*['\"]?v\*", release, re.MULTILINE), (
        "release.yml does not trigger on pushed v* tags")
    assert sorted(_pytest_commands(RELEASE_WORKFLOW)) == sorted(_ci_pytest_commands()), (
        "release.yml's build steps must be exactly ci.yml's pytest commands")
    assert "tools/package_plugin.py" in release, "release.yml does not package with tools/package_plugin.py"
    missing = [z for z in RELEASE_ZIPS if z not in release]
    assert not missing, f"release.yml does not name {missing}"
    scopes = re.findall(r"^\s+([\w-]+): (read|write|none)$", release, re.MULTILINE)
    assert ("contents", "write") in scopes, "release.yml grants no contents: write"
    other = [f"{scope}: {level}" for scope, level in scopes if scope != "contents"]
    assert not other, f"release.yml grants more than contents: {other}"
    unpinned = [
        line.strip()
        for line in release.splitlines()
        if re.search(r"^\s*-?\s*uses:", line)
        and not re.search(r"uses: \S+@[0-9a-f]{40}\s+# v\d", line)
    ]
    assert not unpinned, f"release.yml actions not pinned to a SHA with a version comment: {unpinned}"


def test_readme_lightroom_section_names_the_release_zips_and_keeps_the_from_source_path():
    """Card #402: a user installs from a release download, one zip per
    platform, through Plug-in Manager; readme.md's Lightroom section must name
    both zips, and keep the copy-from-dist step for a build from source."""
    readme = README.read_text(encoding="utf-8")
    section = re.search(r"^## Reviewing in Lightroom\n(.*?)^## ", readme, re.MULTILINE | re.DOTALL)
    assert section, "readme.md has no ## Reviewing in Lightroom section"
    missing = [z for z in RELEASE_ZIPS if z not in section.group(1)]
    assert not missing, f"readme.md's Lightroom section does not name {missing}"
    assert "cp dist/melampus plugin/Melampus.lrplugin/" in section.group(1), (
        "readme.md's Lightroom section lost the from-source install")
