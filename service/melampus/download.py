"""Fetch the MLX model into the Hugging Face cache, with progress the plugin can parse (card #407).

The progress protocol is the plugin's contract (card #408): one line per update
on stdout, defined once here in `Update` and documented in docs/config.md
§ Downloading the model. Errors go to stderr, never into the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

PROGRESS = "progress"
DONE = "done"
CANCELLED = "cancelled"


@dataclass(frozen=True)
class Update:
    """One stdout line of the download protocol.

    `progress <bytes_done> <bytes_total>` while bytes arrive, `done <path>` once
    the model is complete (the path is the rest of the line: it may hold
    spaces), `cancelled` when a signal stopped the download.
    """

    state: str
    bytes_done: int = 0
    bytes_total: int = 0
    path: str = ""

    @classmethod
    def progress(cls, bytes_done: int, bytes_total: int) -> Update:
        return cls(PROGRESS, bytes_done, bytes_total)

    @classmethod
    def done(cls, path: str) -> Update:
        return cls(DONE, path=path)

    @classmethod
    def cancelled(cls) -> Update:
        return cls(CANCELLED)

    def line(self) -> str:
        if self.state == PROGRESS:
            return f"{PROGRESS} {self.bytes_done} {self.bytes_total}"
        if self.state == DONE:
            return f"{DONE} {self.path}"
        return CANCELLED

    @classmethod
    def parse(cls, line: str) -> Update:
        """The plugin's side of the protocol. Raises ValueError for any other line."""
        word, _, rest = line.rstrip("\r\n").partition(" ")
        if word == PROGRESS:
            fields = rest.split(" ")
            if len(fields) == 2 and all(f.isdigit() for f in fields):
                return cls.progress(int(fields[0]), int(fields[1]))
        elif word == DONE and rest:
            return cls.done(rest)
        elif word == CANCELLED and not rest:
            return cls.cancelled()
        raise ValueError(f"not a download update: {line!r}")
