"""Build a self-contained local HTML sheet for reviewing identifications.

Review is per ENCOUNTER, not per frame. The corpus is bursts: 1,743 frames are
roughly 43 shooting encounters, and every frame within one is the same individual.
So one judgement per encounter labels the whole corpus, and takes minutes rather
than hours.

The output is a single .html file with thumbnails embedded as data URIs. It opens
from disk, needs no server and no network, and writes nothing anywhere — you press
"Download corrections" and it saves a JSON file you feed back in with
ingest_corrections.py.

    python tools/make_review_sheet.py fixtures_full stage1_full_results.json review.html
"""

from __future__ import annotations

import base64
import html
import io
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_encounters import cluster  # noqa: E402

THUMB_PX = 460


def norm(name: str | None) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def display(name: str | None) -> str:
    return " ".join((name or "").strip().lower().split())


def thumbnail(path: Path) -> str:
    with Image.open(path) as src:
        img = ImageOps.exif_transpose(src).convert("RGB")
        img.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


PAGE_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Melampus review</title>
<style>
 :root { color-scheme: light dark; }
 body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; padding: 24px;
        max-width: 1100px; margin-inline: auto; }
 h1 { font-size: 22px; margin: 0 0 4px; }
 .sub { opacity: .7; margin-bottom: 20px; }
 .bar { position: sticky; top: 0; padding: 12px 0; backdrop-filter: blur(8px);
        background: color-mix(in srgb, Canvas 85%, transparent); z-index: 5;
        border-bottom: 1px solid color-mix(in srgb, CanvasText 15%, transparent); }
 button { font: inherit; padding: 8px 16px; border-radius: 8px; cursor: pointer;
          border: 1px solid color-mix(in srgb, CanvasText 30%, transparent);
          background: color-mix(in srgb, CanvasText 8%, transparent); color: inherit; }
 button.primary { background: #2563eb; color: #fff; border-color: #2563eb; }
 .row { display: grid; grid-template-columns: 300px 1fr; gap: 20px; padding: 18px 0;
        border-bottom: 1px solid color-mix(in srgb, CanvasText 12%, transparent);
        align-items: start; }
 img { width: 100%; border-radius: 8px; display: block; }
 .meta { font-size: 13px; opacity: .7; margin-top: 6px; }
 .call { font-size: 17px; font-weight: 600; margin-bottom: 2px; }
 .alt { font-size: 13px; opacity: .75; }
 .warn { color: #b45309; font-weight: 600; }
 .choices { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
 label.pick { padding: 6px 12px; border-radius: 999px; cursor: pointer;
              border: 1px solid color-mix(in srgb, CanvasText 25%, transparent); }
 input[type=radio] { margin-right: 6px; }
 label.pick:has(input:checked) { background: #2563eb; color: #fff; border-color: #2563eb; }
 input[type=text] { font: inherit; padding: 7px 10px; border-radius: 8px; width: 300px;
                    border: 1px solid color-mix(in srgb, CanvasText 30%, transparent);
                    background: Canvas; color: inherit; }
 .hidden { display: none; }
</style>
<h1>Melampus &mdash; encounter review</h1>
<div class="sub">One judgement per shooting encounter. Every frame in an encounter is the
same individual, so your answer applies to all of them. Nothing leaves this page until
you press Download.</div>
<div class="bar">
  <button class="primary" onclick="save()">Download corrections</button>
  <span id="count" style="margin-left:12px;opacity:.7"></span>
</div>
"""

PAGE_TAIL = """
<script>
function save() {
  const out = [];
  document.querySelectorAll('.row').forEach(row => {
    const verdict = row.querySelector('input[type=radio]:checked');
    if (!verdict) return;
    const typed = row.querySelector('input[type=text]').value.trim();
    out.push({
      encounter: Number(row.dataset.encounter),
      representative: row.dataset.file,
      frames: Number(row.dataset.frames),
      model_call: row.dataset.call,
      verdict: verdict.value,
      corrected_to: verdict.value === 'wrong' ? typed : null
    });
  });
  const blob = new Blob([JSON.stringify({corrections: out}, null, 1)],
                        {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'melampus_corrections.json';
  a.click();
}
function tally() {
  const done = document.querySelectorAll('input[type=radio]:checked').length;
  const total = document.querySelectorAll('.row').length;
  document.getElementById('count').textContent = done + ' of ' + total + ' reviewed';
}
document.addEventListener('change', e => {
  if (e.target.type === 'radio') {
    const box = e.target.closest('.row').querySelector('.correction');
    box.classList.toggle('hidden', e.target.value !== 'wrong');
    if (e.target.value === 'wrong') box.querySelector('input').focus();
  }
  tally();
});
tally();
</script>
"""


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    folder = Path(argv[1])
    results_path = Path(argv[2])
    out_path = Path(argv[3]) if len(argv) > 3 else Path("review.html")
    gap = float(argv[4]) if len(argv) > 4 else 10.0

    records = {r["file"]: r for r in json.loads(results_path.read_text("utf-8"))}
    encounters = cluster(sorted(folder.glob("*.jpg")), gap)

    parts = [PAGE_HEAD]
    shown = 0
    for enc in encounters:
        names: list[str] = []
        alts: Counter = Counter()
        abstains = done = 0
        for frame in enc.frames:
            rec = records.get(frame.name)
            if rec is None or rec.get("status") != "ok":
                continue
            done += 1
            ident = rec.get("identification") or {}
            cands = sorted(ident.get("candidates", []), key=lambda c: -c.get("confidence", 0))
            if ident.get("abstain") or not cands:
                abstains += 1
                continue
            names.append(display(cands[0].get("common_name")))
            for c in cands[1:]:
                alts[display(c.get("common_name"))] += 1
        if not done:
            continue

        counts = Counter(norm(n) for n in names)
        if counts:
            key, top_n = counts.most_common(1)[0]
            call = next(n for n in names if norm(n) == key)
            agreement = top_n / len(names)
        else:
            call, agreement = "(abstained)", 1.0

        rep = enc.representative()
        distinct = len(counts)
        warn = ""
        if distinct > 1:
            others = ", ".join(
                sorted({n for n in names if norm(n) != norm(call)})
            )
            warn = (f'<div class="alt warn">unstable: {distinct} different species called '
                    f'across frames &mdash; also {html.escape(others)}</div>')

        alt_line = ""
        if alts:
            alt_line = ('<div class="alt">runner-up suggestions: '
                        + html.escape(", ".join(n for n, _ in alts.most_common(3))) + "</div>")

        parts.append(f"""
<div class="row" data-encounter="{enc.index}" data-file="{html.escape(rep.name)}"
     data-frames="{enc.size}" data-call="{html.escape(call)}">
  <div>
    <img src="data:image/jpeg;base64,{thumbnail(rep)}" alt="">
    <div class="meta">{html.escape(rep.name)} &middot; encounter {enc.index}
      &middot; {enc.size} frames ({done} processed)</div>
  </div>
  <div>
    <div class="call">{html.escape(call)}</div>
    <div class="alt">frame agreement {agreement:.0%}
      {f"&middot; {abstains} abstained" if abstains else ""}</div>
    {warn}
    {alt_line}
    <div class="choices">
      <label class="pick"><input type="radio" name="v{enc.index}" value="correct">correct</label>
      <label class="pick"><input type="radio" name="v{enc.index}" value="wrong">wrong</label>
      <label class="pick"><input type="radio" name="v{enc.index}" value="unsure">can't tell</label>
      <span class="correction hidden"><input type="text" placeholder="actual species"></span>
    </div>
  </div>
</div>""")
        shown += 1

    parts.append(PAGE_TAIL)
    out_path.write_text("".join(parts), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  ({shown} encounters, {size_mb:.1f} MB)")
    print(f"open it with:  open {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
