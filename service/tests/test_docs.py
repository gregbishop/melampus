"""Docs drift gates.

Each test here makes one promise a doc carries mechanical, so the doc cannot
silently fall behind the code or the repo again (as happened when [occurrence] and
[quality] shipped undocumented). The promises: docs/config.md names every
implemented setting; AGENTS.md, and not .gitignore, names the install command for
the recorded plugins, and on a clone where an installer has run, that documented
command runs; no doc names a file by an uppercase name it does not have;
docs/brief.md names the pytest command CI actually runs and explains it as
installing from the lockfile; the README's install block and CI both install
from the lockfile (card #425, Done-when 3 and 2); AGENTS.md points at
docs/brief.md without restating its values; and AGENTS.md points at the standard
and names the tracker (card #410, Done-when 3).

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
GITIGNORE = REPO / ".gitignore"
README = REPO / "readme.md"


def _installs_from_the_lockfile(command: str) -> bool:
    """`uv sync --locked` (or `--frozen`) installs exactly uv.lock; anything else
    re-resolves from pyproject.toml's bounds."""
    return bool(re.search(r"\buv sync\b[^&|;]*--(locked|frozen)\b", command))


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
    for doc in [README, AGENTS_MD, *sorted((REPO / "docs").glob("*.md"))]:
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if "README.md" in line or "CONFIG.md" in line:
                stale.append(f"{doc.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not stale, f"docs name uppercase files that do not exist: {stale}"


def _ci_pytest_commands() -> list[str]:
    """The `run:` line of every ci.yml step that invokes pytest."""
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
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


def test_readme_install_block_installs_from_the_lockfile():
    """Card #425, Done-when 3: given a fresh clone, when the README setup runs,
    then the resolved versions match the lockfile. Only `uv sync --locked` (or
    `--frozen`) does that; `uv pip install` never reads uv.lock. Running the
    install here would need the network, so the gate is on the command itself."""
    readme = README.read_text(encoding="utf-8")
    block = re.search(r"^## Install\n.*?```bash\n(.*?)```", readme, re.MULTILINE | re.DOTALL)
    assert block, "readme.md has no bash block under ## Install"
    commands = [line for line in block.group(1).splitlines() if line and not line.startswith("#")]
    locked = [c for c in commands if _installs_from_the_lockfile(c)]
    assert locked and not any("uv pip install" in c for c in commands), (
        "readme.md's ## Install block must install with `uv sync --locked` "
        f"(or --frozen), not re-resolve with `uv pip install`: {commands}"
    )


def test_ci_installs_from_the_lockfile_before_pytest():
    """Card #425, Done-when 2: given CI, when it installs, then it installs from
    the lockfile and fails if the lockfile and pyproject disagree. That is
    `uv sync --locked` (or `--frozen`); `uv pip install` re-resolves instead."""
    not_locked = [c for c in _ci_pytest_commands() if not _installs_from_the_lockfile(c)]
    assert not not_locked, f"CI's pytest step does not install with uv sync --locked/--frozen: {not_locked}"


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
