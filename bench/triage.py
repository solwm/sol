#!/usr/bin/env python3
"""Triage a sol capture dump: flag frames whose pixels violate scene
invariants, so a human (or an LLM with eyes) only has to look at the
interesting ones.

Usage:
    triage.py /tmp/sol-capture [--clear-tolerance 12] [--min-region 64]

Checks, per frame, using the scene metadata each JSONL row carries:

1. clear-color bleed: inside any opaque-ish window rect (alpha >=
   0.5, kind != backdrop, not a background layer), count pixels close
   to the compositor clear color (#05050A). The clear color must
   never be visible inside a window — if it is, something sampled or
   blended against a cleared target (see doc/nvidia-rendering.md).
2. static-scene flicker: when two consecutive captured frames carry
   identical scene descriptions, diff the images; large changed
   regions mean content changed without any scene change — client
   redraw (fine, report only) or compositor garbage (the thing we
   hunt).

Output: one line per finding with frame file, check, bbox (capture
coords), and severity; summary at the end. Exit code 1 if anything
was flagged, so this can gate CI.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("triage.py needs Pillow: pacman -S python-pillow / pip install pillow")

CLEAR = (5, 5, 10)  # 0.02, 0.02, 0.04 in 8-bit


def load_rows(dump_dir: Path):
    rows = []
    with open(dump_dir / "frames.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bbox_of(mask_pixels):
    xs = [p[0] for p in mask_pixels]
    ys = [p[1] for p in mask_pixels]
    return (min(xs), min(ys), max(xs), max(ys))


def check_clear_bleed(img, row, tol, min_region):
    findings = []
    sx = img.width / row["src"][0]
    sy = img.height / row["src"][1]
    px = img.load()
    for el in row["elements"]:
        if el["kind"] == "backdrop" or el.get("bg") or el["alpha"] < 0.5:
            continue
        x0 = max(0, int(el["x"] * sx))
        y0 = max(0, int(el["y"] * sy))
        x1 = min(img.width, int((el["x"] + el["w"]) * sx))
        y1 = min(img.height, int((el["y"] + el["h"]) * sy))
        # Shrink by the corner radius zone (scaled ~4px) so legitimate
        # rounded corners showing the backdrop don't false-positive.
        inset = 4
        x0, y0, x1, y1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        hits = []
        for y in range(y0, y1, 2):  # stride 2: 4x cheaper, plenty dense
            for x in range(x0, x1, 2):
                r, g, b = px[x, y][:3]
                if (
                    abs(r - CLEAR[0]) <= tol
                    and abs(g - CLEAR[1]) <= tol
                    and abs(b - CLEAR[2]) <= tol
                ):
                    hits.append((x, y))
        if len(hits) * 4 >= min_region:  # *4 compensates the stride
            findings.append(
                {
                    "check": "clear-bleed",
                    "key": el["key"],
                    "bbox": bbox_of(hits),
                    "pixels": len(hits) * 4,
                }
            )
    return findings


def check_static_flicker(img_a, img_b, min_region):
    from PIL import ImageChops

    diff = ImageChops.difference(img_a.convert("RGB"), img_b.convert("RGB"))
    bbox = diff.getbbox()
    if bbox is None:
        return []
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area < min_region:
        return []
    return [{"check": "static-flicker", "bbox": bbox, "pixels": area}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--clear-tolerance", type=int, default=12)
    ap.add_argument("--min-region", type=int, default=64)
    args = ap.parse_args()

    rows = load_rows(args.dump_dir)
    flagged = 0
    prev = None  # (row, img)
    for row in rows:
        path = args.dump_dir / row["file"]
        if not path.exists():
            continue
        img = Image.open(path)
        for f in check_clear_bleed(img, row, args.clear_tolerance, args.min_region):
            flagged += 1
            print(
                f"{row['file']} t={row['t_ms']/1000:.3f}s  {f['check']}  "
                f"window key={f['key']} bbox={f['bbox']} px={f['pixels']}  "
                f"[blur_gpu_ns={row['gpu_ns']['blur']} backdrops={row['n_backdrop_draws']}]"
            )
        if prev is not None and prev[0]["elements"] == row["elements"]:
            for f in check_static_flicker(prev[1], img, args.min_region):
                flagged += 1
                print(
                    f"{row['file']} t={row['t_ms']/1000:.3f}s  {f['check']}  "
                    f"bbox={f['bbox']} px={f['pixels']} (scene identical to {prev[0]['file']})"
                )
        prev = (row, img)

    print(f"-- {len(rows)} frames, {flagged} findings")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
