"""Thin CLI over the library. No interactive prompts, no assumptions about cwd."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .cache import ResultCache
from .config import load_config
from .identify import Identifier
from .images import content_hash
from .providers import (
    BACKEND_CHOICES,
    KEY_VARIABLES,
    BackendUnavailable,
    apply_cloud_primary_defaults,
    build_primary_backend,
    default_engine,
    detect_engines,
    is_cloud_primary,
)
from .report import name_quality, raw_table, score
from .runner import BatchStats, list_images, run_batch, stratify_by_prediction
from .schema import ImageResult


def _venv_python() -> str:
    """The venv interpreter path for install hints, phrased for this OS."""
    return ".venv\\Scripts\\python.exe" if sys.platform == "win32" else ".venv/bin/python"


def _fail(message: str) -> int:
    """Exit 3, the code for "could not run; the message on stderr names the
    fix", from every site that refuses a run."""
    print(message, file=sys.stderr)
    return 3


def _sdk_missing(backend: str) -> int:
    """The install hint for a cloud backend whose SDK is not here: which
    backend, and the extra that ships its SDK. Exit 3, for both the primary
    backend and the escalation provider."""
    extra = "openai" if backend == "openai" else "cloud"
    return _fail(
        f"The SDK for the {backend} backend is not installed. Run:\n"
        f'  uv pip install --python {_venv_python()} "./service[{extra}]"'
    )


def _humanise(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _confirm(count: int, cost: float, *, yes_flag: str = "--escalate-yes") -> bool:
    """Ask before spending. Returns False rather than guessing when not a terminal.

    A non-interactive caller (cron, the Lightroom plugin, CI) has no way to answer,
    and defaulting to "yes" there would make the confirmation decorative. Such
    callers pass the yes-flag deliberately.
    """
    if not sys.stdin.isatty():
        print(
            f"  {count} frame(s), est. ${cost:.2f}. Not a terminal, so nothing was "
            f"sent — pass {yes_flag} to run non-interactively.",
            file=sys.stderr,
        )
        return False
    try:
        answer = input(f"  Send {count} frame(s) for about ${cost:.2f}? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in {"y", "yes"}


def _run_escalation(paths, local_cache: ResultCache, config, *,
                    dry_run: bool, assume_yes: bool = False) -> int:
    """Second-opinion pass against the Claude API (CLAUDE.md §6.6)."""
    from .escalate import (
        EscalationRefused,
        build_cloud_identifier,
        compute_range_flags,
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
            return _sdk_missing(config.escalation.provider)
        except ValueError as exc:
            return _fail(str(exc))

    # Range flags make the on_range_flag trigger reachable at all; previously
    # nothing computed them, so that documented case never fired.
    range_flagged = frozenset()
    if config.escalation.on_range_flag:
        try:
            range_flagged = compute_range_flags(local_cache.results(), config)
        except Exception as exc:  # noqa: BLE001 - a missing signal must not block the run
            print(f"range flags unavailable ({exc}); continuing without them",
                  file=sys.stderr)

    def progress(result: ImageResult, reason: str) -> None:
        top = result.identification.top() if result.identification else None
        answer = top.common_name if top else result.status
        print(f"  {result.file}: {reason} -> {answer}", file=sys.stderr)

    try:
        # Cost first, spend second. A "dry run" of the selection is free, so there
        # is no excuse for the estimate to appear after the money is gone.
        preview = escalate(
            paths, local_cache, cloud_cache,
            identifier=None, config=config,
            range_flagged=range_flagged, dry_run=True,
        )
    except EscalationRefused as exc:
        return _fail(str(exc))

    settings = config.escalation
    print(
        f"\nwould escalate {preview.selected} frame(s) to "
        f"{settings.provider}/{resolve_model(settings)}, "
        f"est. ${preview.estimated_cost_usd:.2f} "
        f"at ${settings.input_usd_per_mtok:g}/${settings.output_usd_per_mtok:g} per Mtok "
        f"(set escalation.input_usd_per_mtok / output_usd_per_mtok to match your provider)",
        file=sys.stderr,
    )
    for note in preview.notes[:10]:
        print(f"  note: {note}", file=sys.stderr)

    if dry_run:
        return 0
    if preview.selected == 0:
        print("  nothing to escalate", file=sys.stderr)
        return 0
    if not assume_yes and not _confirm(preview.selected, preview.estimated_cost_usd):
        print("  cancelled; nothing was sent", file=sys.stderr)
        return 0

    try:
        run = escalate(
            paths, local_cache, cloud_cache,
            identifier=identifier, config=config,
            range_flagged=range_flagged, dry_run=False, on_result=progress,
        )
    except EscalationRefused as exc:
        return _fail(str(exc))

    print(
        f"  processed {run.processed}  changed {run.changed}  "
        f"resolved {run.resolved}  refused {run.refused}  errors {run.errors}  "
        f"wall {_humanise(run.seconds)}",
        file=sys.stderr,
    )
    if run.errors:
        print(
            f"  {run.errors} frame(s) failed transiently and were NOT cached — "
            "re-run to retry them",
            file=sys.stderr,
        )
    print(f"  cloud results: {config.escalation.cache_path}", file=sys.stderr)
    for note in run.notes[:10]:
        print(f"  note: {note}", file=sys.stderr)

    if run.processed:
        # Report on the merged view, so the tables reflect what a consumer of the
        # results would actually see rather than only the local pass.
        merged = merged_results(local_cache, cloud_cache)
        print(f"  merged view holds {len(merged)} result(s)", file=sys.stderr)
    return 0


def _download_model(repo: str) -> int:
    """--download-model (card #407): fetch the MLX model with progress on
    stdout in the protocol the plugin parses (docs/config.md § Downloading the
    model). Exit 0 once complete, 3 on a failure with the fix on stderr, and
    EXIT_CANCELLED when a signal or the cancel marker (card #408) stopped it
    with the partial file kept: one path for both."""
    from .download import EXIT_CANCELLED, DownloadCancelled, DownloadError, Update, cancel_on_signals, download_model

    def emit(update: Update) -> None:
        print(update.line(), flush=True)

    try:
        with cancel_on_signals():
            path = download_model(repo, on_update=emit)
    except DownloadCancelled:
        emit(Update.cancelled())
        return EXIT_CANCELLED
    except DownloadError as exc:
        return _fail(str(exc))
    emit(Update.done(str(path)))
    return 0


def _model_status(repo: str) -> int:
    """--model-status (card #408): one JSON object on stdout saying whether the
    MLX model is in the cache, its size, and where the plugin writes to
    cancel a download. Never fails for the network: the size is null then.
    Exit 3 with the reason on stderr for a repo that is not a repo id."""
    from .download import DownloadError, model_status

    try:
        status = model_status(repo)
    except DownloadError as exc:
        return _fail(str(exc))
    print(status.json())
    return 0


def _remove_model(repo: str) -> int:
    """--remove-model (card #408): delete the MLX model from the cache, exit 0
    with `removed <path>`; exit 3 with the reason on stderr when nothing is
    installed or a download of it is running."""
    from .download import DownloadError, remove_model

    try:
        path = remove_model(repo)
    except DownloadError as exc:
        return _fail(str(exc))
    print(f"removed {path}")
    return 0


def _write_plugin_results(paths: list[Path], cache: ResultCache, config, destination: Path) -> None:
    """The enrichment pass the Lightroom plugin reads (card #436).

    Quality is always scored: the plugin sets no star rating without it. Range
    checks need a default location and are the one step that touches the
    network; without one they are skipped and the log says so.
    """
    from .occurrence import range_lookup
    from .plugin_results import enrich, progress_printer, write_plugin_results

    lookup = range_lookup(config.occurrence)
    if lookup is None:
        print("no default location configured; skipping range checks", file=sys.stderr)
    outcome = enrich(paths, cache.records(), config, lookup=lookup,
                     on_progress=progress_printer(sys.stderr))
    write_plugin_results(destination, outcome.rows)
    print(f"\nwrote {destination}\n{outcome.summary()}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="melampus-id", description=__doc__)
    ap.add_argument("folder", type=Path, nargs="?", help="folder of JPEGs")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--no-local-config", action="store_true",
                    help="do not read melampus.local.toml beside the data: --config "
                         "alone, over the defaults, is the whole configuration")
    ap.add_argument("--model", default=None, help="override model repo")
    ap.add_argument("--backend", choices=BACKEND_CHOICES, default=None,
                    help="which engine answers: mlx locally on Apple Silicon, "
                         "ollama locally through an Ollama server, openai or "
                         "claude for machines with no local runtime, or scripted "
                         "(a fake that answers nothing; for smoke tests without "
                         "weights). Default: the first that can run here, per "
                         "--detect-engines")
    ap.add_argument("--detect-engines", action="store_true",
                    help="print, as JSON, which engines can run on this machine "
                         "and why or why not, then exit; needs no folder")
    ap.add_argument("--download-model", action="store_true",
                    help="fetch the MLX model ([model] repo, or --model) into the "
                         "Hugging Face cache, one 'progress <bytes done> <bytes total>' "
                         "line per update on stdout and 'done <path>' at the end, "
                         "then exit; resumes an interrupted download; stops, exit 4, "
                         "on a signal or when the cancel file --model-status names "
                         "appears; needs no folder")
    ap.add_argument("--model-status", action="store_true",
                    help="print, as one JSON object, whether the MLX model ([model] repo, "
                         "or --model) is in the Hugging Face cache, its size and path, "
                         "then exit; needs no folder or network")
    ap.add_argument("--remove-model", action="store_true",
                    help="delete the MLX model ([model] repo, or --model) from the "
                         "Hugging Face cache and print 'removed <path>', then exit; "
                         "refused while a download of it is running; needs no folder")
    ap.add_argument("--yes", action="store_true",
                    help="skip the cost confirmation when the primary backend is a "
                         "cloud provider (for non-interactive callers)")
    ap.add_argument("--profile", choices=("wildlife", "sport"), default=None,
                    help="what kind of shoot this is; picks the routing prompt")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="process at most N images")
    ap.add_argument("--force", action="store_true", help="reprocess already-cached images")
    ap.add_argument("--per-species", type=int, default=None,
                    help="stratified sample: at most N per previously-predicted species")
    ap.add_argument("--labels", type=Path, default=None, help="reference labels JSON for scoring")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--plugin-out", type=Path, default=None,
                    help="also write the results enriched for the Lightroom plugin "
                         "(quality and its rank within the burst, burst agreement, "
                         "range flag, encounter); range checks need a default "
                         "location in config and are the only step that uses the network")
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
    cloud.add_argument("--escalate-provider", choices=tuple(KEY_VARIABLES), default=None,
                       help="which cloud to ask (default: claude)")
    cloud.add_argument("--escalate-yes", action="store_true",
                       help="skip the cost confirmation (for non-interactive callers)")
    cloud.add_argument("--escalate-base-url", default=None,
                       help="OpenAI-compatible endpoint, for OpenRouter, LM Studio, a proxy, ...")

    args = ap.parse_args(argv)

    overrides: dict = {}
    if args.model:
        overrides.setdefault("model", {})["repo"] = args.model
    if args.backend:
        overrides.setdefault("model", {})["backend"] = args.backend
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
    config = load_config(args.config, use_local=not args.no_local_config, **overrides)

    if args.detect_engines:
        # The address probed is the configured one, read the way the run
        # reads it (--config and --no-local-config alike), so the verdict
        # cannot disagree with what --backend ollama would talk to.
        print(json.dumps([asdict(v) for v in detect_engines(config.model.ollama_url)], indent=2))
        return 0
    if args.download_model:
        return _download_model(config.model.repo)
    if args.model_status:
        return _model_status(config.model.repo)
    if args.remove_model:
        return _remove_model(config.model.repo)
    if args.folder is None:
        ap.error("the following arguments are required: folder")

    if "backend" not in config.model.model_fields_set:
        # Nothing named an engine: neither --backend nor [model] backend. The
        # first that can run here answers (card #404), before anything reads
        # the choice: the cloud retuning below, the cache file, the refusals.
        config.model.backend = default_engine(config.model.ollama_url)
        print(
            f"engine: {config.model.backend} (the first that can run here; "
            "--backend or [model] backend chooses, --detect-engines explains)",
            file=sys.stderr,
        )

    cloud_primary = is_cloud_primary(config)
    if cloud_primary:
        # Before anything else reads config: the cache opened below must be the
        # cloud file (paid answers never share the local one), and the identifier's
        # fingerprint bakes in the retuned values.
        for change in apply_cloud_primary_defaults(config):
            print(f"  cloud default: {change}", file=sys.stderr)

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
        try:
            backend = build_primary_backend(config)
        except ImportError:
            return _sdk_missing(config.model.backend)
        except (BackendUnavailable, ValueError) as exc:
            return _fail(str(exc))
        identifier = Identifier(backend, config)

        if cloud_primary:
            # A cloud primary bills every frame, not just an escalated tail — so it
            # gets the same courtesy the tail gets: cost first, spend second.
            from .escalate import estimate_cost_usd

            selected = paths[: args.limit if args.limit is not None else len(paths)]
            pending = 0
            for path in selected:
                try:
                    digest = content_hash(path)
                except OSError:
                    continue
                if args.force or not cache.has_success(digest, identifier.fingerprint):
                    pending += 1
            rates = config.escalation  # per-Mtok prices live in [escalation]
            cost = estimate_cost_usd(pending, rates)
            print(
                f"every frame goes to {config.model.backend}/{backend.name}: "
                f"{pending} frame(s) to process, est. ${cost:.2f} "
                f"at ${rates.input_usd_per_mtok:g}/${rates.output_usd_per_mtok:g} per Mtok "
                "(set escalation.input_usd_per_mtok / output_usd_per_mtok to match "
                "your provider)",
                file=sys.stderr,
            )
            # The hard ceiling holds even for a confirmed or --yes run: unlike
            # escalation there is no most-uncertain-first ordering to make a
            # truncated batch meaningful, so refusing outright beats billing an
            # arbitrary subset. Raising it is a config edit, i.e. deliberate.
            if pending > config.model.max_images:
                return _fail(
                    f"  refused: {pending} frame(s) exceed model.max_images = "
                    f"{config.model.max_images}. A cloud primary bills every frame — "
                    "raise model.max_images in config, or narrow the run with --limit."
                )
            if pending and not args.yes and not _confirm(pending, cost, yes_flag="--yes"):
                print("  cancelled; nothing was sent", file=sys.stderr)
                return 0
        else:
            print(f"loading {backend.name} ...", file=sys.stderr)
            backend.warmup()

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
        code = _run_escalation(paths, cache, config,
                               dry_run=args.escalate_dry_run,
                               assume_yes=args.escalate_yes)
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
    if args.plugin_out:
        _write_plugin_results(paths, cache, config, args.plugin_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
