"""The model download command (card #407).

Done-when 1: given the download command, when it runs, then it fetches the
configured MLX model, reports bytes done and total on stdout as it goes, and
resumes if interrupted.
Done-when 2: given a cancel signal, when it arrives, then the download stops
cleanly and partial files are kept for resume.
Done-when 3: given the tests, when they run, then a local fake of the model
host stands in and nothing is fetched from the internet.

The progress protocol is the plugin's (card #408) contract, so it is tested in
both directions: the lines the command prints, and the parse the plugin will do.
"""

from __future__ import annotations

import pytest

from melampus.download import Update


@pytest.mark.parametrize(
    ("update", "line"),
    [
        (Update.progress(0, 18_300_000_000), "progress 0 18300000000"),
        (Update.progress(4_096, 4_096), "progress 4096 4096"),
        (Update.done("/hf/hub/models--x--y/snapshots/abc"), "done /hf/hub/models--x--y/snapshots/abc"),
        (Update.done("C:\\Users\\me\\AppData\\Local\\hf hub\\snapshots\\abc"),
         "done C:\\Users\\me\\AppData\\Local\\hf hub\\snapshots\\abc"),
        (Update.cancelled(), "cancelled"),
    ],
)
def test_progress_protocol_prints_and_parses_the_same_line(update: Update, line: str):
    """One line per update, machine-readable and stable: `progress <done> <total>`
    while bytes arrive, `done <path>` once the model is complete, `cancelled`
    when a signal stopped it. A path may hold spaces, so it is the rest of the
    line."""
    assert update.line() == line
    assert Update.parse(line) == update
    assert Update.parse(line + "\n") == update, "a line read from a pipe keeps its newline"


@pytest.mark.parametrize("line", [
    "", "progress", "progress 1", "progress one two", "progress 1 2 3",
    "done", "cancelled now", "Downloading bytes: 100%", "engine: mlx",
])
def test_progress_protocol_rejects_what_is_not_an_update(line: str):
    """The plugin must be able to tell an update from any other line."""
    with pytest.raises(ValueError):
        Update.parse(line)
