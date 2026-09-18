"""Docs drift gate.

docs/config.md promises every implemented setting, with rationale. This test makes
that promise mechanical: a config field that does not appear in the doc fails CI, so
the doc cannot silently fall behind the code again (as happened when [occurrence]
and [quality] shipped undocumented).

The check is deliberately dumb — substring presence of the backticked key name — so
it never argues with prose style, only with absence.
"""

import json
from pathlib import Path

from melampus.config import MelampusConfig

CONFIG_DOC = Path(__file__).resolve().parents[2] / "docs" / "config.md"


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


ROOT = CONFIG_DOC.parents[1]
AGENTS_MD = ROOT / "AGENTS.md"
PLUGIN_CHOICE = ROOT / ".agents" / "on-purpose.json"


def test_agents_md_names_the_install_command_for_the_recorded_plugins():
    """The install outputs (.claude/settings.json, .agents/skills, .codex/agents)
    are machine-local and untracked; a fresh clone must be told how to regenerate
    them, with the same plugins .agents/on-purpose.json records."""
    plugins = json.loads(PLUGIN_CHOICE.read_text(encoding="utf-8"))["plugins"]
    command = f"node ~/on-purpose/bin/install.mjs {' '.join(plugins)}"
    assert f"`{command}`" in AGENTS_MD.read_text(encoding="utf-8"), (
        f"AGENTS.md must tell a fresh clone to run `{command}` "
        "(the plugins recorded in .agents/on-purpose.json)"
    )


def test_docs_name_only_the_lowercase_files():
    """The real files are readme.md and docs/config.md. A doc that still says
    README.md or docs/CONFIG.md, or claims another doc does, is stale."""
    stale = []
    for doc in [ROOT / "readme.md", AGENTS_MD, *sorted((ROOT / "docs").glob("*.md"))]:
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if "README.md" in line or "CONFIG.md" in line:
                stale.append(f"{doc.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not stale, f"docs name uppercase files that do not exist: {stale}"
