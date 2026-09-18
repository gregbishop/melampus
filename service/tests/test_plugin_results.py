"""The enrichment pass the Lightroom plugin reads, as service code (card #436).

The plugin gates every write on fields the raw per-frame results do not carry:
`burst_agreement`, `range_flag` and `encounter` (MelampusRules.lua), and it will
not set a star rating without `quality`, `quality_rank` and `encounter_frames`.
Until this card those came from tools/make_plugin_results.py, a second Python
step the shipped executable did not contain. Now `melampus-id --plugin-out`
writes them from inside the executable.

Done-when 1: given the service package, when `melampus-id` is given
`--plugin-out <path>`, then it writes the enriched results the tool wrote
before this card, from the same inputs. The tool is now a thin caller of the
same `enrich`, so the corpus test here proves that the tool and `--plugin-out`
write the same bytes; the comparison against the untouched tool was made out
of tree, at review, against the base branch's tools/make_plugin_results.py.

Unit tests run over stub frames carrying only an XMP packet and a fake quality
scorer; the integration tests run the real CLI over real pixels with the GBIF
client faked at the network edge, as test_occurrence.py does.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from conftest import PHOTO

from melampus import cli
from melampus.cache import ResultCache
from melampus.config import load_config
from melampus.images import content_hash
from melampus.occurrence import GBIFClient
from melampus.plugin_results import PLUGIN_FIELDS, enrich, write_plugin_results
from melampus.schema import ImageResult

from test_encounters import stub_frame
from test_occurrence import FLORIDA, FakeClient

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "fixtures"
TOOL = REPO / "tools" / "make_plugin_results.py"

# What GBIF would say near Florida: the cuckoo is Asian, the rest are residents.
COUNTS = {"Cuculus micropterus": 0, "Egretta tricolor": 3060, "Egretta caerulea": 2954}


def record(file: str, species: list[tuple[str, str, float]] | None, *,
           taxon: str = "bird", abstain: bool = False, content: str | None = None) -> dict:
    """One raw result as the CLI's --json-out writes it."""
    ident = None
    if species is not None:
        ident = {
            "taxon": taxon,
            "candidates": [{"common_name": c, "scientific_name": s, "confidence": conf,
                            "reasoning": ""} for c, s, conf in species],
            "age_sex": "adult", "count": 1, "behavior": [],
            "diagnostic_features_visible": True,
            "abstain": abstain, "abstain_reason": "facing away" if abstain else None,
        }
    return {"file": file, "content_hash": content or file, "status": "ok",
            "identification": ident, "model": "test", "seconds": 1.0}


def fake_scorer(scores: dict[str, float | None]):
    """Stands in for quality.analyze_quality: None means the frame could not be scored."""
    from melampus.quality import QualityResult

    def score(path: Path, config) -> QualityResult:
        result = QualityResult()
        value = scores.get(path.name)
        if value is None:
            result.error = "unscorable"
        else:
            result.composite = value
        return result

    return score


def burst(tmp_path: Path) -> list[Path]:
    """Three frames five seconds apart, then one frame a minute later."""
    return [
        stub_frame(tmp_path, "a.jpg", "2025-06-01T08:00:00"),
        stub_frame(tmp_path, "b.jpg", "2025-06-01T08:00:05"),
        stub_frame(tmp_path, "c.jpg", "2025-06-01T08:00:10"),
        stub_frame(tmp_path, "d.jpg", "2025-06-01T08:01:30"),
    ]


HERON = [("Tricolored Heron", "Egretta tricolor", 0.8), ("Little Blue Heron", "Egretta caerulea", 0.2)]
BLUE = [("Little Blue Heron", "Egretta caerulea", 0.7)]
CUCKOO = [("Long-tailed Cuckoo", "Cuculus micropterus", 0.9)]


# --------------------------------------------------------------------------- #
# unit: the three gates, one encounter at a time
# --------------------------------------------------------------------------- #
def test_burst_agreement_is_the_share_of_frames_that_agree_with_the_majority(tmp_path: Path):
    frames = burst(tmp_path)
    records = [record("a.jpg", HERON), record("b.jpg", HERON), record("c.jpg", BLUE),
               record("d.jpg", HERON)]
    rows = enrich(frames, records, load_config(use_local=False), score=None).rows

    by_file = {r["file"]: r for r in rows}
    assert [r["file"] for r in rows] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert [by_file[f]["encounter"] for f in ("a.jpg", "b.jpg", "c.jpg", "d.jpg")] == [0, 0, 0, 1]
    assert by_file["a.jpg"]["burst_agreement"] == by_file["c.jpg"]["burst_agreement"] == 0.667
    assert by_file["d.jpg"]["burst_agreement"] == 1.0
    assert all(r["range_flag"] is False for r in rows), "no lookup was given, nothing is flagged"


def test_abstentions_carry_no_vote_and_an_all_abstained_burst_has_no_agreement(tmp_path: Path):
    frames = burst(tmp_path)
    records = [record("a.jpg", HERON), record("b.jpg", [], abstain=True),
               record("c.jpg", HERON), record("d.jpg", None, abstain=True)]
    rows = {r["file"]: r for r in enrich(frames, records, load_config(use_local=False), score=None).rows}

    assert rows["a.jpg"]["burst_agreement"] == 1.0, "an abstained frame must not dilute the burst"
    assert "burst_agreement" not in rows["d.jpg"]


def test_range_flag_is_one_lookup_per_encounter_in_the_month_it_was_shot(tmp_path: Path):
    frames = burst(tmp_path)
    records = [record("a.jpg", CUCKOO), record("b.jpg", CUCKOO), record("c.jpg", CUCKOO),
               record("d.jpg", HERON)]
    client = FakeClient(COUNTS)
    outcome = enrich(frames, records, load_config(use_local=False), score=None,
                     lookup=(client, FLORIDA))

    rows = {r["file"]: r for r in outcome.rows}
    assert [rows[f]["range_flag"] for f in ("a.jpg", "b.jpg", "c.jpg", "d.jpg")] == [True, True, True, False]
    assert client.calls == [("Cuculus micropterus", 6), ("Egretta tricolor", 6)], (
        "one lookup per encounter, for the month of the encounter")
    assert outcome.flagged_encounters == 1


def test_range_flag_never_fires_for_a_non_organism_or_a_failed_lookup(tmp_path: Path):
    frames = burst(tmp_path)
    records = [record("a.jpg", [("American Football", "", 0.95)], taxon="football"),
               record("b.jpg", [("American Football", "", 0.95)], taxon="football"),
               record("c.jpg", [("American Football", "", 0.95)], taxon="football"),
               record("d.jpg", [("Some Bird", "Unknownus unknownus", 0.9)])]
    client = FakeClient(COUNTS)
    rows = {r["file"]: r for r in enrich(frames, records, load_config(use_local=False), score=None,
                                          lookup=(client, FLORIDA)).rows}

    assert not any(name == "" for name, _ in client.calls), "a sport was looked up on GBIF"
    assert rows["a.jpg"]["range_flag"] is False
    assert rows["d.jpg"]["range_flag"] is False, "an unavailable lookup (None) is not absence"


def test_quality_is_ranked_within_the_encounter_not_across_the_folder(tmp_path: Path):
    frames = burst(tmp_path)
    records = [record(f"{n}.jpg", HERON) for n in "abcd"]
    scores = {"a.jpg": 30.04, "b.jpg": 50.0, "c.jpg": 40.0, "d.jpg": 10.0}
    rows = {r["file"]: r for r in enrich(frames, records, load_config(use_local=False),
                                          score=fake_scorer(scores)).rows}

    assert rows["a.jpg"]["quality"] == 30.0, "composite is rounded to one decimal"
    assert [rows[f]["quality_rank"] for f in ("a.jpg", "c.jpg", "b.jpg")] == [0.0, 0.5, 1.0]
    assert rows["a.jpg"]["encounter_frames"] == 3
    # The lone frame of the second encounter is its own best and worst; it must
    # not be ranked against the burst before it.
    assert rows["d.jpg"]["quality_rank"] == 0.0 and rows["d.jpg"]["encounter_frames"] == 1


def test_an_unscorable_frame_gets_no_quality_and_is_left_out_of_the_ranking(tmp_path: Path):
    frames = burst(tmp_path)
    records = [record(f"{n}.jpg", HERON) for n in "abcd"]
    scores = {"a.jpg": 30.0, "b.jpg": None, "c.jpg": 40.0, "d.jpg": None}
    rows = {r["file"]: r for r in enrich(frames, records, load_config(use_local=False),
                                          score=fake_scorer(scores)).rows}

    assert "quality" not in rows["b.jpg"] and "quality_rank" not in rows["b.jpg"]
    assert rows["a.jpg"]["encounter_frames"] == 2, "only scored frames count"
    assert "encounter_frames" not in rows["d.jpg"]


def test_frames_without_a_result_and_results_without_a_frame_are_left_out(tmp_path: Path):
    frames = burst(tmp_path)
    records = [record("a.jpg", HERON), record("elsewhere.jpg", HERON)]
    rows = enrich(frames, records, load_config(use_local=False), score=None).rows

    assert [r["file"] for r in rows] == ["a.jpg"]
    assert rows[0]["identification"]["candidates"][0]["common_name"] == "Tricolored Heron", (
        "the raw record travels with its enrichment")


def test_write_plugin_results_is_the_old_tools_json_shape(tmp_path: Path):
    out = tmp_path / "plugin_results.json"
    write_plugin_results(out, [{"file": "a.jpg", "encounter": 0}])
    assert out.read_bytes() == json.dumps([{"file": "a.jpg", "encounter": 0}], indent=1).encode("utf-8")


# --------------------------------------------------------------------------- #
# integration: the CLI, real pixels, GBIF faked at the network edge
# --------------------------------------------------------------------------- #
def seed(cache_path: Path, frames: list[Path], species: dict[str, list]) -> None:
    """Results for these frames as a run would have cached them."""
    cache = ResultCache(cache_path)
    for frame in frames:
        rec = record(frame.name, species[frame.name], content=content_hash(frame))
        cache.put(ImageResult.model_validate(rec))


def _config_file(tmp_path: Path, *, with_location: bool = True) -> Path:
    """An explicit config: Florida as the default location, caches under tmp."""
    location = ("default_latitude = 26.45\ndefault_longitude = -82.11\n" if with_location else "")
    path = tmp_path / "melampus.toml"
    path.write_text(
        "[occurrence]\n" + location
        + f"cache_path = '{(tmp_path / 'occurrence.json').as_posix()}'\n"
        + f"[run]\ncache_path = '{(tmp_path / 'identifications.jsonl').as_posix()}'\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def offline_gbif(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None]]:
    """Every GBIFClient in the process answers from COUNTS and touches no network."""
    calls: list[tuple[str, int | None]] = []

    def count(self, scientific_name, location, month):
        calls.append((scientific_name, month))
        return COUNTS.get(scientific_name)

    monkeypatch.setattr(GBIFClient, "count", count)
    return calls


def _load_tool():
    spec = importlib.util.spec_from_file_location("make_plugin_results", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_out_writes_every_field_the_plugin_reads(photos: Path, tmp_path: Path, offline_gbif, capsys):
    """Done-when 1, on the committed frame: one `melampus-id` run writes the six
    fields MelampusImport.lua reads, next to the raw --json-out."""
    frame = photos / PHOTO
    config = _config_file(tmp_path)
    seed(tmp_path / "identifications.jsonl", [frame], {frame.name: CUCKOO})
    out = tmp_path / "plugin_results.json"

    code = cli.main([str(photos), "--config", str(config), "--report-only",
                     "--json-out", str(tmp_path / "raw.json"), "--plugin-out", str(out)])

    assert code == 0, capsys.readouterr().err
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [r["file"] for r in rows] == [frame.name]
    row = rows[0]
    assert set(PLUGIN_FIELDS) <= set(row), f"missing {set(PLUGIN_FIELDS) - set(row)}"
    assert row["range_flag"] is True and row["burst_agreement"] == 1.0
    assert row["encounter"] == 0 and row["encounter_frames"] == 1 and row["quality_rank"] == 0.0
    assert 0 < row["quality"] <= 100
    assert offline_gbif == [("Cuculus micropterus", None)], "a frame with no capture date has no month"
    assert f"wrote {out}" in capsys.readouterr().err


def test_plugin_out_skips_range_checks_without_a_default_location_and_says_so(
    photos: Path, tmp_path: Path, offline_gbif, capsys
):
    frame = photos / PHOTO
    config = _config_file(tmp_path, with_location=False)
    seed(tmp_path / "identifications.jsonl", [frame], {frame.name: CUCKOO})
    out = tmp_path / "plugin_results.json"

    code = cli.main([str(photos), "--config", str(config), "--report-only", "--plugin-out", str(out)])

    assert code == 0
    assert offline_gbif == []
    assert json.loads(out.read_text(encoding="utf-8"))[0]["range_flag"] is False
    assert "no default location" in capsys.readouterr().err


@pytest.mark.skipif(not CORPUS.is_dir(), reason="corpus fixtures not present")
def test_plugin_out_and_the_thin_tool_write_the_same_bytes_on_the_corpus(
    tmp_path: Path, offline_gbif, monkeypatch: pytest.MonkeyPatch
):
    """Done-when 1 as it holds at HEAD: the same raw results and the same frames
    through `melampus-id --plugin-out` and through the thin
    tools/make_plugin_results.py --occurrence --quality give the same bytes, so
    the plugin's Analyze command and the executable cannot drift."""
    frames = sorted(CORPUS.glob("*.jpg"))[:8]
    species = {}
    for index, frame in enumerate(frames):
        species[frame.name] = [CUCKOO, HERON, BLUE, HERON][index % 4]
    config = _config_file(tmp_path)
    seed(tmp_path / "identifications.jsonl", frames, species)
    folder = tmp_path / "photos"
    folder.mkdir()
    for frame in frames:
        (folder / frame.name).symlink_to(frame)
    raw = tmp_path / "raw.json"
    from_cli = tmp_path / "from_cli.json"
    from_tool = tmp_path / "from_tool.json"

    assert cli.main([str(folder), "--config", str(config), "--report-only",
                     "--json-out", str(raw), "--plugin-out", str(from_cli)]) == 0

    # The tool reads melampus.local.toml only; point it at the same config.
    from melampus import config as config_module
    monkeypatch.setattr(config_module, "_local_config", lambda: config)
    tool = _load_tool()
    assert tool.main([str(TOOL), str(folder), str(raw), str(from_tool), "--occurrence", "--quality"]) == 0

    assert from_cli.read_bytes() == from_tool.read_bytes()
    rows = json.loads(from_cli.read_text(encoding="utf-8"))
    assert len(rows) == len(frames)
    assert sum(r["range_flag"] for r in rows) == 2, "the two cuckoo frames are flagged"
    assert all(set(PLUGIN_FIELDS) <= set(r) for r in rows)


def test_the_tool_is_a_thin_caller_of_the_service_module():
    """Done-when 4: tools/make_plugin_results.py no longer carries its own
    enrichment; it parses arguments and calls melampus.plugin_results. The
    Lightroom plugin still invokes it until card #401 rewires the plugin, so
    it stays, and stays thin."""
    source = TOOL.read_text(encoding="utf-8")
    assert "from melampus.plugin_results import" in source
    for own_logic in ("most_common", "quality_rank", "burst_agreement"):
        assert own_logic not in source, f"the tool still computes {own_logic!r} itself"
