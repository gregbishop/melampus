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
workflow file that does not exist; and no doc states a test count, because
the suite grows with every card and CI checks no such number (card #437,
Done-when 1).

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
WORKFLOW_SUFFIXES = (".yml", ".yaml")  # GitHub runs both
CI_WORKFLOW = WORKFLOWS / "ci.yml"
LUA_PLUGIN_TESTS = REPO / "service" / "tests" / "test_lua_plugin.py"
PLUGIN_DOC = REPO / "docs" / "plugin.md"
ARCHITECTURE_DOC = REPO / "docs" / "architecture.md"
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
    # Card #440: the build keeps PyInstaller's cache inside the checkout, and the
    # section says so, names the switch, and says a corrupt cache is deleted.
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
    using = {name for name, text in texts.items() if re.search(r"^\s*-?\s*uses:", text, re.MULTILINE)}
    assert CI_WORKFLOW.name in using, f"a workflow uses no action: {sorted(texts)}"
    unpinned = [f"{name}: {line}" for name, text in texts.items() for line in _unpinned_actions(text)]
    assert not unpinned, f"a workflow names an action by tag, not a commit SHA with its version: {unpinned}"


RELEASE_ZIPS = ("Melampus-macOS.zip", "Melampus-Windows.zip")


def _steps(job: str) -> list[str]:
    """That job's steps, each as its text."""
    return re.split(r"^      - ", job, flags=re.MULTILINE)[1:]


def _unpinned_actions(text: str) -> list[str]:
    """The `uses:` lines in that text not pinned to a commit SHA with the
    version in a trailing comment."""
    return [
        line.strip()
        for line in text.splitlines()
        if re.search(r"^\s*-?\s*uses:", line) and not re.search(r"uses: \S+@[0-9a-f]{40}\s+# v\d", line)
    ]


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
    the file; and every action this card adds (the zip uploads and the
    release job's) is pinned to a commit SHA with the version in a trailing
    comment. ci.yml's earlier `uses:` lines are card #439's."""
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

    zip_steps = [
        step for name in packaging for step in _steps(jobs[name]) if any(z in step for z in RELEASE_ZIPS)
    ]
    assert zip_steps, "no step uploads a zip"
    unpinned = _unpinned_actions(release) + [u for step in zip_steps for u in _unpinned_actions(step)]
    assert not unpinned, f"actions added for the release are not pinned to a SHA with a version comment: {unpinned}"


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

def test_docs_name_engine_detection_where_the_default_and_the_refusal_are_described():
    """Card #404: the backend's default is now the first engine that can run
    here, and `--detect-engines` is how a user (and card #405's dialog) sees
    the verdicts. docs/config.md's `backend` row and readme.md's Windows
    section describe the default and the refusal, so both must name the flag,
    and neither may still promise that detection is yet to come."""
    config_doc = CONFIG_DOC.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    backend_row = next(line for line in config_doc.splitlines() if line.startswith("| `backend` |"))
    assert "`--detect-engines`" in backend_row, "docs/config.md's backend row does not name --detect-engines"
    assert "turns that into" not in backend_row, "docs/config.md still says detection is yet to come"
    assert "`--detect-engines`" in readme, "readme.md does not name --detect-engines"


def test_docs_describe_the_engine_picker_and_where_the_key_lives():
    """Card #405: the engine has a control in Settings now. docs/plugin.md's
    engine section must describe the picker (what greys an engine, where the
    key goes: LrPasswords, never a file) instead of promising the dialog is
    yet to come, and readme.md's Lightroom section must show the dialog: the
    screenshot's reference, docs/settings-dialog.png, which the owner takes."""
    plugin_doc = (REPO / "docs" / "plugin.md").read_text(encoding="utf-8")
    engine = re.search(r"^## The engine\n(.*?)^---", plugin_doc, re.MULTILINE | re.DOTALL)
    assert engine, "docs/plugin.md has no ## The engine section"
    prose = " ".join(engine.group(1).split())
    assert "until then the preference is unset" not in prose, (
        "docs/plugin.md still says the engine has no control in Settings")
    for named in ("`--detect-engines`", "LrPasswords", "settings-dialog.png"):
        assert named in prose, f"docs/plugin.md's engine section does not name {named}"
    readme = README.read_text(encoding="utf-8")
    section = re.search(r"^## Reviewing in Lightroom\n(.*?)^## ", readme, re.MULTILINE | re.DOTALL)
    assert section, "readme.md has no ## Reviewing in Lightroom section"
    assert "docs/settings-dialog.png" in section.group(1), (
        "readme.md's Lightroom section does not show the settings dialog")


def test_docs_describe_the_cli_engines_in_the_picker():
    """Card #423: the two subscription CLIs are in the picker. docs/plugin.md's
    engine section names them, says they take no key and bill to the
    subscription, and where their titles come from (the verdict); readme.md's
    Lightroom section lists them among what the picker offers; docs/config.md's
    backend row, readme.md and the engine section no longer defer the picker
    to a card yet to come."""
    plugin_doc = (REPO / "docs" / "plugin.md").read_text(encoding="utf-8")
    engine = re.search(r"^## The engine\n(.*?)^---", plugin_doc, re.MULTILINE | re.DOTALL)
    assert engine, "docs/plugin.md has no ## The engine section"
    prose = " ".join(engine.group(1).split())
    for named in ("`claude-code`", "`codex`", "subscription", "no API key", "`title`"):
        assert named in prose, f"docs/plugin.md's engine section does not say {named}"
    readme = README.read_text(encoding="utf-8")
    section = re.search(r"^## Reviewing in Lightroom\n(.*?)^## ", readme, re.MULTILINE | re.DOTALL)
    assert section, "readme.md has no ## Reviewing in Lightroom section"
    for named in ("claude-code", "codex"):
        assert named in section.group(1), f"readme.md's Lightroom section does not offer {named}"
    config_doc = CONFIG_DOC.read_text(encoding="utf-8")
    backend_row = next(line for line in config_doc.splitlines() if line.startswith("| `backend` |"))
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
                    f"`{CANCELLED}`", f"exit {EXIT_CANCELLED}", "exit 3", "exit 0", "stderr", "resume"):
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


def test_config_doc_quotes_the_claude_code_template_from_its_one_source():
    """Card #421: one place holds Claude Code's template,
    providers.CLAUDE_CODE_COMMAND. docs/config.md quotes it as the
    `[model] command` a user would set to override it, in a TOML block
    that parses to exactly that list, so the doc cannot rot into a second
    copy; and it says what to install, how to sign in, and that runs bill
    to the subscription."""
    import tomllib

    from melampus import providers

    text = (REPO / "docs" / "config.md").read_text(encoding="utf-8")
    blocks = [
        block for block in re.findall(r"```toml\n(.*?)```", text, re.DOTALL)
        if 'backend = "claude-code"' in block
    ]
    assert blocks, "docs/config.md has no ```toml block with backend = \"claude-code\""
    (block,) = blocks
    assert tomllib.loads(block)["model"]["command"] == providers.CLAUDE_CODE_COMMAND
    for said in (providers.CLAUDE_CODE_INSTALL, f"`{providers.CLAUDE_CODE_SIGN_IN}`", "subscription"):
        assert said in text, f"docs/config.md does not say {said!r}"
    readme = (REPO / "readme.md").read_text(encoding="utf-8")
    assert "`claude-code`" in readme and "--backend claude-code" in readme
    architecture = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "claude-code" in architecture


def test_config_doc_quotes_the_codex_template_from_its_one_source():
    """Card #422: one place holds Codex CLI's template,
    providers.CODEX_COMMAND. docs/config.md quotes it as the
    `[model] command` a user would set to override it, in a TOML block
    that parses to exactly that list, so the doc cannot rot into a second
    copy; and it says what to install, how to sign in, that runs bill to
    the plan, and what happens at its usage limit."""
    import tomllib

    from melampus import providers

    text = (REPO / "docs" / "config.md").read_text(encoding="utf-8")
    blocks = [
        block for block in re.findall(r"```toml\n(.*?)```", text, re.DOTALL)
        if 'backend = "codex"' in block
    ]
    assert blocks, "docs/config.md has no ```toml block with backend = \"codex\""
    (block,) = blocks
    assert tomllib.loads(block)["model"]["command"] == providers.CODEX_COMMAND
    for said in (providers.CODEX_INSTALL, f"`{providers.CODEX_SIGN_IN}`", "usage limit"):
        assert said in text, f"docs/config.md does not say {said!r}"
    readme = (REPO / "readme.md").read_text(encoding="utf-8")
    assert "`codex`" in readme and "--backend codex" in readme
    architecture = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "`codex`" in architecture and "CODEX_COMMAND" in architecture


TEST_COUNT_DOCS = (README, BRIEF, PLUGIN_DOC, ARCHITECTURE_DOC)
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


def test_docs_state_no_test_count():
    """Card #437, Done-when 1: the suite grows with every card, so a count
    written into a doc is wrong the day after, and CI checks no such number.
    The docs say what the tests need and how to run them, never how many
    there are, in any of the forms a count has taken: "182 tests", "46 rules
    tests", "410 passed", "26 skipped", "skips 2", and "126 Python, 56 Lua"
    in a sentence about tests."""
    stated = []
    for doc in TEST_COUNT_DOCS:
        for sentence in _sentences(doc.read_text(encoding="utf-8")):
            counted = TEST_COUNT.search(sentence) or (
                "test" in sentence.lower() and LANGUAGE_COUNT.search(sentence)
            )
            if counted:
                stated.append(f"{doc.relative_to(REPO)}: {sentence}")
    assert not stated, f"docs state a test count that CI does not check: {stated}"
