"""Thin CLI over the library. No interactive prompts, no assumptions about cwd."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backend import MLXBackend
from .cache import ResultCache
from .config import load_config
from .identify import Identifier
from .report import name_quality, raw_table, score
from .runner import BatchStats, list_images, run_batch, stratify_by_prediction
from .schema import ImageResult


def _humanise(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _run_escalation(paths, local_cache: ResultCache, config, *, dry_run: bool) -> int:
    """Second-opinion pass against the Claude API (CLAUDE.md §6.6)."""
    from .escalate import (
        EscalationRefused,
        build_cloud_identifier,
        escalate,
        merged_results,
        resolve_model,
    )

    cloud_cache = ResultCache(config.escalation.cache_path)
    identifier = None
    if not dry_run:
        try:
            identifier = build_cloud_identifier(config)
        except ImportError:
            extra = "openai" if config.escalation.provider == "openai" else "cloud"
            print(
                f"The {config.escalation.provider} SDK is not installed. Run:\n"
                f'  uv pip install --python .venv/bin/python "./service[{extra}]"',
                file=sys.stderr,
            )
            return 3
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 3

    def progress(result: ImageResult, reason: str) -> None:
        top = result.identification.top() if result.identification else None
        answer = top.common_name if top else result.status
        print(f"  {result.file}: {reason} -> {answer}", file=sys.stderr)

    try:
        run = escalate(
            paths, local_cache, cloud_cache,
            identifier=identifier, config=config,
            dry_run=dry_run, on_result=progress,
        )
    except EscalationRefused as exc:
        print(str(exc), file=sys.stderr)
        return 3

    settings = config.escalation
    verb = "would escalate" if dry_run else "escalated"
    # State the rates in the same breath as the dollars. The rates are config and
    # default to Claude Opus 5's, so an unadjusted estimate for another provider is
    # confidently wrong — better to show the assumption than to hide it.
    print(
        f"\n{verb} {run.selected} frame(s) to {settings.provider}/{resolve_model(settings)}, "
        f"est. ${run.estimated_cost_usd:.2f} "
        f"at ${settings.input_usd_per_mtok:g}/${settings.output_usd_per_mtok:g} per Mtok "
        f"(set escalation.input_usd_per_mtok / output_usd_per_mtok to match your provider)",
        file=sys.stderr,
    )
    if not dry_run:
        print(
            f"  processed {run.processed}  changed {run.changed}  "
            f"resolved {run.resolved}  refused {run.refused}  errors {run.errors}  "
            f"wall {_humanise(run.seconds)}",
            file=sys.stderr,
        )
        print(f"  cloud results: {config.escalation.cache_path}", file=sys.stderr)
    for note in run.notes[:10]:
        print(f"  note: {note}", file=sys.stderr)

    if not dry_run and run.processed:
        # Report on the merged view, so the tables reflect what a consumer of the
        # results would actually see rather than only the local pass.
        merged = merged_results(local_cache, cloud_cache)
        print(f"  merged view holds {len(merged)} result(s)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="melampus-id", description=__doc__)
    ap.add_argument("folder", type=Path, help="folder of JPEGs")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", default=None, help="override model repo")
    ap.add_argument("--profile", choices=("wildlife", "sport"), default=None,
                    help="what kind of shoot this is; picks the routing prompt")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="process at most N images")
    ap.add_argument("--force", action="store_true", help="reprocess already-cached images")
    ap.add_argument("--per-species", type=int, default=None,
                    help="stratified sample: at most N per previously-predicted species")
    ap.add_argument("--labels", type=Path, default=None, help="reference labels JSON for scoring")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="print tables from cache without running the model")

    cloud = ap.add_argument_group(
        "cloud escalation (optional, off by default)",
        "Sends the frames the local model was unsure about to the Claude API for a "
        "second opinion. This is the only path that sends photographs off your "
        "machine, and it costs money. Needs MELAMPUS_ANTHROPIC_KEY.",
    )
    cloud.add_argument("--escalate", action="store_true",
                       help="after the local pass, re-run the uncertain frames in the cloud")
    cloud.add_argument("--escalate-dry-run", action="store_true",
                       help="show what escalation would send and roughly cost, and send nothing")
    cloud.add_argument("--escalate-max", type=int, default=None,
                       help="ceiling on how many frames one run may bill for")
    cloud.add_argument("--escalate-model", default=None, help="override the cloud model")
    cloud.add_argument("--escalate-provider", choices=("anthropic", "openai"), default=None,
                       help="which cloud to ask (default: anthropic)")
    cloud.add_argument("--escalate-base-url", default=None,
                       help="OpenAI-compatible endpoint, for OpenRouter, LM Studio, a proxy, ...")

    args = ap.parse_args(argv)

    overrides: dict = {}
    if args.model:
        overrides.setdefault("model", {})["repo"] = args.model
    if args.profile:
        overrides.setdefault("run", {})["profile"] = args.profile
    if args.cache:
        overrides.setdefault("run", {})["cache_path"] = str(args.cache)
    if args.escalate or args.escalate_dry_run:
        overrides.setdefault("escalation", {})["enabled"] = True
    if args.escalate_max is not None:
        overrides.setdefault("escalation", {})["max_images"] = args.escalate_max
    if args.escalate_model:
        overrides.setdefault("escalation", {})["model"] = args.escalate_model
    if args.escalate_provider:
        overrides.setdefault("escalation", {})["provider"] = args.escalate_provider
    if args.escalate_base_url:
        overrides.setdefault("escalation", {})["base_url"] = args.escalate_base_url
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

    if args.escalate or args.escalate_dry_run:
        code = _run_escalation(paths, cache, config, dry_run=args.escalate_dry_run)
        if code != 0:
            return code

    wanted = {p.name for p in paths}
    results = [r for r in cache.results() if r.file in wanted]

    print(raw_table(results))

    print()
    print(name_quality(results))

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
