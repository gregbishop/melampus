"""Encounter clustering as service code (card #436).

The clusterer began life in tools/ but the plugin's enrichment pass depends on
it, and that pass now lives inside the executable, so the clusterer is part of
the service package. It reads xmp:CreateDate straight out of the file bytes and
never opens the pixels, so stub files carrying only an XMP packet are enough.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from melampus.encounters import Encounter, capture_time, cluster


def stub_frame(folder: Path, name: str, when: str | None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    packet = b""
    if when is not None:
        packet = (
            b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            + f'<rdf:Description xmp:CreateDate="{when}"/>'.encode()
            + b"</x:xmpmeta>"
        )
    path = folder / name
    path.write_bytes(b"\xff\xd8\xff\xe1" + packet)
    return path


def test_capture_time_reads_the_xmp_create_date(tmp_path: Path):
    frame = stub_frame(tmp_path, "a.jpg", "2025-06-01T08:00:00.50-04:00")
    assert capture_time(frame) == datetime.fromisoformat("2025-06-01T08:00:00.50-04:00")
    assert capture_time(stub_frame(tmp_path, "b.jpg", None)) is None


def test_frames_within_the_gap_share_an_encounter_and_a_pause_starts_a_new_one(tmp_path: Path):
    a = stub_frame(tmp_path, "a.jpg", "2025-06-01T08:00:00")
    b = stub_frame(tmp_path, "b.jpg", "2025-06-01T08:00:05")
    c = stub_frame(tmp_path, "c.jpg", "2025-06-01T08:00:30")
    encounters = cluster([c, a, b], gap_seconds=10.0)
    assert [[f.name for f in e.frames] for e in encounters] == [["a.jpg", "b.jpg"], ["c.jpg"]]
    assert [e.index for e in encounters] == [0, 1]
    first = encounters[0]
    assert isinstance(first, Encounter)
    assert first.size == 2 and first.duration_s == 5.0
    assert first.start.month == 6


def test_undated_frames_are_one_trailing_orphan_encounter(tmp_path: Path):
    dated = stub_frame(tmp_path, "a.jpg", "2025-06-01T08:00:00")
    lost = stub_frame(tmp_path, "b.jpg", None)
    encounters = cluster([lost, dated], gap_seconds=10.0)
    assert [[f.name for f in e.frames] for e in encounters] == [["a.jpg"], ["b.jpg"]]
    assert encounters[1].start is None and encounters[1].duration_s == 0.0
    assert encounters[1].representative() == lost
