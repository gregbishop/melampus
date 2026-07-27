"""Watch the local model identify a photograph, token by token.

Nothing here is a simulation. The weights are read from the local HuggingFace
cache, inference runs on the Apple Silicon GPU through MLX, and the text appears
as fast as the machine produces it. Run it with the network off if you want proof.

    python tools/watch_live.py fixtures/0A1A4175.jpg
    python tools/watch_live.py --random fixtures_full

Both stages are shown: the cheap taxon routing pass, then the taxon-specialised
species pass. The image handed to the model is the staged copy — resized, stripped
of all metadata, and renamed — exactly as in a real run.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

from melampus.backend import MLXBackend
from melampus.config import load_config
from melampus.images import staged_pixels
from melampus.prompts import ROUTING_PROMPT, PromptLibrary

REPO = Path(__file__).resolve().parents[1]

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
OFF = "\033[0m"


def rule(title: str) -> None:
    width = min(shutil.get_terminal_size((90, 20)).columns, 90)
    print(f"\n{CYAN}{BOLD}── {title} {'─' * max(0, width - len(title) - 4)}{OFF}")


def stream(backend: MLXBackend, image: Path, prompt: str, max_tokens: int) -> str:
    """Print the model's output as it is generated, and return the full text."""
    from mlx_vlm import apply_chat_template, stream_generate

    formatted = apply_chat_template(backend._processor, backend._config, prompt, num_images=1)
    pieces: list[str] = []
    tokens = 0
    started = time.perf_counter()

    for chunk in stream_generate(
        backend._model, backend._processor, formatted,
        image=[str(image)], max_tokens=max_tokens, temperature=0.0,
    ):
        text = getattr(chunk, "text", "") or ""
        if text:
            pieces.append(text)
            tokens += 1
            sys.stdout.write(text)
            sys.stdout.flush()

    elapsed = time.perf_counter() - started
    rate = tokens / elapsed if elapsed else 0.0
    print(f"\n{DIM}  {tokens} chunks in {elapsed:.1f}s  ({rate:.1f}/s){OFF}")
    return "".join(pieces)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="an image, or a folder with --random")
    ap.add_argument("--random", action="store_true", help="pick a random image from the folder")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv[1:])

    if args.random:
        candidates = sorted(args.target.glob("*.jpg"))
        if not candidates:
            print(f"no JPEGs in {args.target}", file=sys.stderr)
            return 2
        image = random.choice(candidates)
    else:
        image = args.target
    if not image.is_file():
        print(f"not a file: {image}", file=sys.stderr)
        return 2

    config = load_config(run={"prompts_dir": str(REPO / "prompts")})
    repo = args.model or config.model.repo
    prompts = PromptLibrary(REPO / "prompts")

    print(f"{BOLD}Melampus — live inference{OFF}")
    print(f"{DIM}model  {repo}{OFF}")
    print(f"{DIM}image  {image}{OFF}")
    print(f"{DIM}device Apple Silicon GPU via MLX — weights from local cache, no network{OFF}")

    rule("loading weights")
    import mlx.core as mx

    t0 = time.perf_counter()
    backend = MLXBackend(repo, 0.0)
    backend.warmup()
    gib = mx.get_active_memory() / 2**30
    print(f"{GREEN}loaded in {time.perf_counter() - t0:.1f}s{OFF}")
    print(f"{DIM}GPU memory now held by MLX: {gib:.1f} GiB "
          f"— that is the model resident on the Apple Silicon GPU{OFF}")

    with staged_pixels(image, config.image.max_edge, config.image.jpeg_quality) as staged:
        from PIL import Image as PILImage

        with PILImage.open(staged) as probe:
            size = probe.size
        print(f"{DIM}staged {staged.name} at {size[0]}x{size[1]} "
              f"— metadata stripped, filename discarded{OFF}")

        rule("stage A — what kind of organism is this?")
        routing_text = stream(
            backend, staged, prompts.render(ROUTING_PROMPT), config.model.routing_max_tokens
        )

        # Route exactly as the real pipeline does, rather than assuming a taxon.
        from melampus.identify import extract_json
        from melampus.schema import Taxon, TaxonRouting

        payload = extract_json(routing_text)
        try:
            routing = TaxonRouting.model_validate(payload or {})
        except Exception:
            print(f"{YELLOW}stage A did not return valid JSON — the real pipeline would "
                  f"retry once, then mark this photo unprocessed{OFF}")
            return 0

        print(f"{GREEN}routed to '{routing.taxon.value}' "
              f"(confidence {routing.confidence:.2f}){OFF}")

        if routing.taxon is Taxon.NONE:
            print(f"\n{YELLOW}No organism present, so the species stage is skipped "
                  f"entirely — there is nothing to ask about, and this is where the "
                  f"two-stage design pays for itself.{OFF}")
            return 0

        rule(f"stage B — species identification (using the {routing.taxon.value} prompt)")
        stream(backend, staged, prompts.prompt_for_taxon(routing.taxon.value),
               config.model.max_tokens)

    peak = mx.get_peak_memory() / 2**30
    print(f"\n{DIM}peak GPU memory this run: {peak:.1f} GiB{OFF}")
    print(f"\n{YELLOW}Both stages are schema-validated before anything is kept; "
          f"a parse failure gets one corrective retry, then the photo is left "
          f"unprocessed rather than guessed at.{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
