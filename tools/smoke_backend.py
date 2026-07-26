"""Minimal end-to-end check that the MLX backend and prompts actually work.

Runs the real two-stage pipeline against a couple of images with whatever model is
given. Intended for use with a small model to validate wiring before committing to a
long batch with a large one.

    python tools/smoke_backend.py mlx-community/Qwen3-VL-2B-Instruct-4bit fixtures/a.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

from melampus.backend import MLXBackend
from melampus.config import load_config
from melampus.identify import Identifier


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    repo, images = argv[1], [Path(p) for p in argv[2:]]

    config = load_config(
        model={"repo": repo},
        run={"prompts_dir": str(Path(__file__).resolve().parents[1] / "prompts")},
    )
    backend = MLXBackend(repo, config.model.temperature)
    print(f"loading {repo} ...", flush=True)
    backend.warmup()
    print("loaded", flush=True)

    identifier = Identifier(backend, config)
    for path in images:
        result = identifier.identify(path)
        print(f"\n=== {path.name}  [{result.status}]  {result.seconds:.1f}s  retries={result.retries}")
        if result.error:
            print(f"    error: {result.error}")
        if result.taxon_routing:
            print(f"    routed: {result.taxon_routing.taxon.value} ({result.taxon_routing.confidence:.2f})")
        ident = result.identification
        if ident:
            print(f"    abstain={ident.abstain} count={ident.count} age_sex={ident.age_sex}")
            for rank, cand in enumerate(ident.ranked(), 1):
                print(f"      {rank}. {cand.common_name} ({cand.scientific_name}) {cand.confidence:.2f}")
                print(f"         {cand.reasoning[:160]}")
            if ident.abstain:
                print(f"      reason: {ident.abstain_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
