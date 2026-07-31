"""Docs drift gate.

docs/config.md promises every implemented setting, with rationale. This test makes
that promise mechanical: a config field that does not appear in the doc fails CI, so
the doc cannot silently fall behind the code again (as happened when [occurrence]
and [quality] shipped undocumented).

The check is deliberately dumb — substring presence of the backticked key name — so
it never argues with prose style, only with absence.
"""

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
