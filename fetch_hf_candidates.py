#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stream a bounded pilot sample from WebNovels-Ja and save only candidate rows locally.

This step is retrieval, not typology. The broad lexical rules only reduce the corpus to
likely human/nonhuman intimate-relationship narratives. These retrieval labels are not
fed to embeddings or clustering.
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
    "婚約","求婚","伴侶","番","つがい","恋人","恋仲","同居","暮らす","一緒に暮ら","新婚","溺愛","契約婚","生贄","生け贄"
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


def score_candidate(meta_text, head):
    nhm = {w for w in NONHUMAN_TERMS if w in meta_text}
    relm = {w for w in RELATION_TERMS if w in meta_text}
    nhh = {w for w in NONHUMAN_TERMS if w in head}
    relh = {w for w in RELATION_TERMS if w in head}
    if not (nhm or nhh) or not (relm or relh):
        return 0.0
    return 2 * min(len(nhm), 3) + 2 * min(len(relm), 3) + 0.75 * min(len(nhh), 4) + 0.75 * min(len(relh), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="OmniAICreator/WebNovels-Ja")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default="data/pilot_candidates.jsonl")
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--max-candidates", type=int, default=150)
    ap.add_argument("--head-chars", type=int, default=8000)
    ap.add_argument("--min-score", type=float, default=4.0)
    ap.add_argument("--shuffle-buffer", type=int, default=10000)
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
            score = score_candidate(meta_text, text[:args.head_chars])
            if score >= args.min_score:
                # Save the original row locally for the next stage. This file is never uploaded as an artifact.
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
        print("WARNING: very few candidates; broaden max_rows or lower min_score for the next pilot.")


if __name__ == "__main__":
    main()
