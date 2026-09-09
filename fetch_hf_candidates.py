#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stream a bounded sample from WebNovels-Ja and save candidate rows locally.

The population is restricted to original works (isoriginal=1) by default. A work
qualifies only when the sampled text contains explicit local evidence of a human side,
a nonhuman side, and a romantic/marital relationship in the same or adjacent sentence
block. Final candidates are selected uniformly with reservoir sampling.
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from event_windows import relation_evidence_blocks


def is_original(meta):
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
    ap.add_argument("--max-sentences", type=int, default=2,
                    help="Require human/nonhuman/relation evidence within this many adjacent sentences.")
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

    if not args.include_non_original:
        ds = ds.filter(row_is_original)

    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    rng = random.Random(args.seed)
    reservoir = []
    population_rows = 0
    qualifying_rows = 0
    evidence_blocks_total = 0

    for row in ds:
        if population_rows >= args.max_rows:
            break
        population_rows += 1

        text = row.get("text", "") or ""
        if not text:
            continue
        evidence = relation_evidence_blocks(text[:args.head_chars], max_sentences=args.max_sentences)
        if not evidence:
            continue

        qualifying_rows += 1
        evidence_blocks_total += len(evidence)
        item = {"text": text, "meta": row.get("meta", {}) or {}}
        if len(reservoir) < args.max_candidates:
            reservoir.append(item)
        else:
            j = rng.randrange(qualifying_rows)
            if j < args.max_candidates:
                reservoir[j] = item

    rng.shuffle(reservoir)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in reservoir:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"population={'all' if args.include_non_original else 'original_only'}")
    print(f"population_rows_scanned={population_rows}")
    print(f"qualifying_rows={qualifying_rows}")
    print(f"evidence_blocks_total={evidence_blocks_total}")
    print(f"selected={len(reservoir)}")
    print("qualification=explicit_human_nonhuman_relationship")
    print(f"max_sentences={args.max_sentences}")
    print("sampling=reservoir")
    print(f"seed={args.seed}")
    print(f"output={out}")
    if len(reservoir) < 10:
        print("WARNING: very few candidates; broaden max_rows before relaxing relationship evidence.")


if __name__ == "__main__":
    main()
