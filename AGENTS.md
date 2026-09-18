# melampus

Follow the on-purpose standard: `plugins/standard/standards/ticket.md` and
`plugins/standard/standards/tdd.md` in the on-purpose checkout, installed here as
the `standard` and `python` plugins. Its working rules come first.

tracker: nextcloud-deck board=Melampus

The project brief, stack contract, and hard rules are in `docs/brief.md`. Read it
before any change. The stack contract there is the source of truth for build,
test, and run commands: `.venv/bin/python -m pytest` from the repo root, with the
lockfile at `service/uv.lock`.
