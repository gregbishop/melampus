"""Thin CLI over the library. No interactive prompts, no assumptions about cwd."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backend import MLXBackend
from .cache import ResultCache
from .config import load_config
from .identify import Identifier
from .report import raw_table, score
from .runner import BatchStats, list_images, run_batch, stratify_by_prediction
from .schema import ImageResult


def _humanise(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="melampus-id", description=__doc__)
    ap.add_argument("folder", type=Path, help="folder of JPEGs")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None, help="override model repo")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="process at most N images")
    ap.add_argument("--force", action="store_true", help="reprocess already-cached images")
    ap.add_argument("--per-species", type=int, default=None,
                    help="stratified sample: at most N per previously-predicted species")
    ap.add_argument("--labels", type=Path, default=None, help="reference labels JSON for scoring")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="print tables from cache without running the model")
    args = ap.parse_args(argv)

    overrides: dict = {}
    if args.model:
        overrides.setdefault("model", {})["repo"] = args.model
    if args.cache:
        overrides.setdefault("run", {})["cache_path"] = str(args.cache)
    config = load_config(args.config, **overrides)

    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2

    cache = ResultCache(config.run.cache_path)
    paths = list_images(args.folder)
    if not paths:
        print(f"no images found in {args.folder}", file=sys.stderr)
        return 2

    if args.per_species is not None:
        paths = stratify_by_prediction(paths, cache, args.per_species)
        print(f"stratified selection: {len(paths)} images", file=sys.stderr)

    if not args.report_only:
        backend = MLXBackend(config.model.repo, config.model.temperature)
        print(f"loading {config.model.repo} ...", file=sys.stderr)
        backend.warmup()
        identifier = Identifier(backend, config)

        def progress(result: ImageResult, stats: BatchStats) -> None:
            done = stats.processed
            mean = stats.mean_seconds
            remaining = (stats.total - done - stats.skipped) * mean
            print(
                f"[{done}/{stats.total - stats.skipped}] {result.file} "
                f"{result.status} {result.seconds:.1f}s "
                f"(mean {mean:.1f}s, eta {_humanise(remaining)})",
                file=sys.stderr,
            )

        stats = run_batch(
            paths, identifier, cache,
            force=args.force, limit=args.limit, on_result=progress,
        )
        print(
            f"\nprocessed {stats.processed}  skipped {stats.skipped}  "
            f"ok {stats.ok}  unprocessed {stats.unprocessed}  errors {stats.errors}",
            file=sys.stderr,
        )
        print(
            f"per-image: mean {stats.mean_seconds:.1f}s  median {stats.median_seconds:.1f}s  "
            f"wall {_humanise(stats.seconds)}",
            file=sys.stderr,
        )

    wanted = {p.name for p in paths}
    results = [r for r in cache.results() if r.file in wanted]

    print(raw_table(results))

    if args.labels:
        print()
        _, rendered = score(results, args.labels)
        print(rendered)

    if args.json_out:
        cache.export_json(args.json_out)
        print(f"\nwrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
