#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stream a bounded sample from WebNovels-Ja and save candidate rows locally.

By default the analysis population is restricted to rows explicitly marked as original
(`isoriginal=1`) before randomization and lexical retrieval. Qualifying works are then
sampled with reservoir sampling, so the output is not the first N matches in storage
order. Retrieval labels are never used as clustering features.
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset

NONHUMAN_TERMS = [
    "人外","異類","異種族","妖怪","あやかし","もののけ","鬼","妖狐","狐","狸","蛇","龍神","竜神","龍","竜",
    "神","神様","女神","精霊","妖精","天狗","雪女","吸血鬼","ヴァンパイア","悪魔","魔族","獣人","亜人",
    "エルフ","人魚","ラミア","ハーピー","フェンリル","アンデッド","ゾンビ","幽霊","死神","宇宙人","異星人",
    "アンドロイド","ロボット","人工生命","怪物","化け物","モンスター"
]
RELATION_TERMS = [
    "恋","恋愛","好き","愛する","愛され","惹かれ","想い","結婚","婚姻","嫁","嫁入り","花嫁","妻","夫","夫婦",
    "婚約","求婚","伴侶","つがい","恋人","恋仲","同居","暮らす","一緒に暮ら","新婚","溺愛","契約婚","生贄","生け贄"
]


def flatten_strings(obj):
    out = []
    if isinstance(obj, dict):
        for value in obj.values():
            out.extend(flatten_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(flatten_strings(value))
    elif isinstance(obj, (str, int, float, bool)):
        out.append(str(obj))
    return out


def term_hits(text, terms):
    return {w for w in terms if w in text}


def proximity_hits(text, window=500):
    """Return nonhuman/relation terms that co-occur within a local window."""
    nh_pos = []
    rel_pos = []
    for w in NONHUMAN_TERMS:
        start = 0
        while True:
            p = text.find(w, start)
            if p < 0:
                break
            nh_pos.append((p, w))
            start = p + max(1, len(w))
    for w in RELATION_TERMS:
        start = 0
        while True:
            p = text.find(w, start)
            if p < 0:
                break
            rel_pos.append((p, w))
            start = p + max(1, len(w))

    nh = set()
    rel = set()
    pairs = 0
    for p, nw in nh_pos:
        for q, rw in rel_pos:
            if abs(p - q) <= window:
                nh.add(nw)
                rel.add(rw)
                pairs += 1
    return nh, rel, pairs


def score_candidate(meta_text, head, proximity_window=500):
    nhm = term_hits(meta_text, NONHUMAN_TERMS)
    relm = term_hits(meta_text, RELATION_TERMS)
    nhp, relp, pairs = proximity_hits(head, proximity_window)

    meta_pair = bool(nhm and relm)
    text_pair = bool(nhp and relp)
    if not (meta_pair or text_pair):
        return 0.0

    score = 0.0
    if meta_pair:
        score += 2.0 * min(len(nhm), 3) + 2.0 * min(len(relm), 3)
    if text_pair:
        score += 1.5 + 0.75 * min(len(nhp), 4) + 0.75 * min(len(relp), 4)
        score += 0.25 * min(pairs, 4)
    return score


def is_original(meta):
    """Return True only for rows explicitly marked original by the source metadata."""
    if not isinstance(meta, dict):
        return False
    value = meta.get("isoriginal", None)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def row_is_original(row):
    return is_original((row or {}).get("meta", {}) or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="OmniAICreator/WebNovels-Ja")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default="data/pilot_candidates.jsonl")
    ap.add_argument("--max-rows", type=int, default=50000,
                    help="Maximum rows in the selected population to inspect (original rows by default).")
    ap.add_argument("--max-candidates", type=int, default=100)
    ap.add_argument("--head-chars", type=int, default=12000)
    ap.add_argument("--min-score", type=float, default=3.5)
    ap.add_argument("--proximity-window", type=int, default=500)
    ap.add_argument("--shuffle-buffer", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-non-original", action="store_true",
                    help="Use all rows instead of restricting the population to isoriginal=1.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Add a read token to repository Actions secrets.")

    kwargs = dict(path=args.repo, split=args.split, streaming=True, token=token)
    if args.config:
        kwargs["name"] = args.config
    ds = load_dataset(**kwargs)

    # Define the sampling population first. With the default settings, every downstream
    # operation sees original works only. The filter is lazy for the streaming dataset.
    if not args.include_non_original:
        ds = ds.filter(row_is_original)

    # Shuffle *within the selected population*. Reservoir sampling below is still the
    # mechanism that makes the final candidate sample uniform among qualifying rows.
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    rng = random.Random(args.seed)
    reservoir = []
    population_rows = 0
    qualifying_rows = 0

    for row in ds:
        if population_rows >= args.max_rows:
            break
        population_rows += 1

        text = row.get("text", "") or ""
        meta = row.get("meta", {}) or {}
        meta_text = " ".join(flatten_strings(meta))
        score = score_candidate(meta_text, text[:args.head_chars], args.proximity_window)
        if score < args.min_score:
            continue

        qualifying_rows += 1
        item = {"text": text, "meta": meta}
        if len(reservoir) < args.max_candidates:
            reservoir.append(item)
        else:
            # Algorithm R: after q qualifying rows, every qualifying row has probability
            # max_candidates/q of being represented in the final reservoir.
            j = rng.randrange(qualifying_rows)
            if j < args.max_candidates:
                reservoir[j] = item

    # Randomize output order independently of stream/storage position.
    rng.shuffle(reservoir)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in reservoir:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"population={'all' if args.include_non_original else 'original_only'}")
    print(f"population_rows_scanned={population_rows}")
    print(f"qualifying_rows={qualifying_rows}")
    print(f"selected={len(reservoir)}")
    print(f"sampling=reservoir")
    print(f"seed={args.seed}")
    print(f"output={out}")
    if len(reservoir) < 10:
        print("WARNING: very few candidates; broaden max_rows before relaxing proximity filtering.")


if __name__ == "__main__":
    main()
