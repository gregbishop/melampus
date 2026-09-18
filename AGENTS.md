# melampus

Follow the on-purpose standard: `plugins/standard/standards/ticket.md` and
`plugins/standard/standards/tdd.md` in the on-purpose checkout, installed here as
the `standard` and `python` plugins. Its working rules come first.

The install outputs (`.claude/settings.json`, `.agents/skills`, `.codex/agents`,
the secret pre-commit hook) are machine-local and not tracked. A fresh clone
regenerates them from the repo root with
`node ~/on-purpose/bin/install.mjs standard python`; `.agents/on-purpose.json`
records that plugin choice.

tracker: nextcloud-deck board=Melampus

The project brief, stack contract, and hard rules are in `docs/brief.md`. Read it
before any change. The stack contract there is the source of truth for build,
test, and run commands: `.venv/bin/python -m pytest` from the repo root, with the
lockfile at `service/uv.lock`.
