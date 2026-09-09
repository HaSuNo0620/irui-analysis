#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stream a bounded pilot sample from WebNovels-Ja and save candidate rows locally.

Retrieval is deliberately high-recall, but requires a nonhuman term and an intimate-
relationship term to occur near each other. Retrieval labels are never used as
clustering features.
"""

import argparse
import json
import os
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

    # Metadata co-occurrence is strong evidence; otherwise require local textual co-occurrence.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="OmniAICreator/WebNovels-Ja")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default="data/pilot_candidates.jsonl")
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--max-candidates", type=int, default=100)
    ap.add_argument("--head-chars", type=int, default=12000)
    ap.add_argument("--min-score", type=float, default=3.5)
    ap.add_argument("--proximity-window", type=int, default=500)
    ap.add_argument("--shuffle-buffer", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Add a read token to repository Actions secrets.")

    kwargs = dict(path=args.repo, split=args.split, streaming=True, token=token)
    if args.config:
        kwargs["name"] = args.config
    ds = load_dataset(**kwargs)
    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scanned = 0
    selected = 0

    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            scanned += 1
            text = row.get("text", "") or ""
            meta = row.get("meta", {}) or {}
            meta_text = " ".join(flatten_strings(meta))
            score = score_candidate(meta_text, text[:args.head_chars], args.proximity_window)
            if score >= args.min_score:
                # Local handoff only; candidate text is never uploaded as an artifact.
                f.write(json.dumps({"text": text, "meta": meta}, ensure_ascii=False) + "\n")
                selected += 1
                if selected >= args.max_candidates:
                    break
            if scanned >= args.max_rows:
                break

    print(f"scanned={scanned}")
    print(f"selected={selected}")
    print(f"output={out}")
    if selected < 10:
        print("WARNING: very few candidates; broaden max_rows before relaxing proximity filtering.")


if __name__ == "__main__":
    main()
