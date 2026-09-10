#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sample work-level candidates from RyokoAI_Syosetu711K.

Each JSONL row in this corpus is one Syosetu novel and carries its N-code in
meta.id. We deliberately bypass datasets.load_dataset because the repository's
viewer/schema can disagree across rows; raw JSONL files are streamed directly
through Hugging Face's filesystem interface.

The human/nonhuman romantic-or-marital evidence is used ONLY to decide corpus
membership. Explicit secondary/fan works are excluded BEFORE qualification.
Downstream semantic analysis uses the selected work's full novel text.
"""

import argparse
import json
import os
import random
from pathlib import Path

from huggingface_hub import HfFileSystem

from event_windows import relation_evidence_blocks


def flatten_text_values(obj):
    """Yield scalar metadata values as strings without imposing genre semantics."""
    if isinstance(obj, dict):
        for value in obj.values():
            yield from flatten_text_values(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from flatten_text_values(value)
    elif obj is not None:
        yield str(obj)


def is_explicit_secondary_work(meta):
    """High-precision exclusion: only metadata explicitly marked 二次創作.

    We intentionally do not infer secondary status from genre/fandom codes or
    named franchises, so corpus membership is not altered by speculative rules.
    """
    if not isinstance(meta, dict):
        return False
    keywords = meta.get("keywords", [])
    values = list(flatten_text_values(keywords))
    return any("二次創作" in value for value in values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="botp/RyokoAI_Syosetu711K")
    ap.add_argument("--output", default="data/pilot_candidates.jsonl")
    ap.add_argument("--max-rows", type=int, default=200000,
                    help="Maximum work-level JSONL rows to inspect.")
    ap.add_argument("--max-candidates", type=int, default=500)
    ap.add_argument("--scan-chars", type=int, default=60000,
                    help="Characters from the beginning of each work used only for strict membership evidence.")
    ap.add_argument("--max-sentences", type=int, default=2,
                    help="Require human/nonhuman/relation evidence within this many adjacent sentences.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    fs = HfFileSystem(token=token)
    root = f"datasets/{args.repo}"
    files = sorted(fs.glob(f"{root}/syosetu711k-*.jsonl"))
    if not files:
        raise SystemExit(f"No Syosetu711K JSONL shards found under {root}")

    rng = random.Random(args.seed)
    rng.shuffle(files)  # reduce shard-order bias before bounded scanning

    reservoir = []
    rows_scanned = 0
    qualifying_rows = 0
    evidence_blocks_total = 0
    bad_rows = 0
    missing_ncode = 0
    excluded_secondary = 0

    stop = False
    for shard in files:
        if stop:
            break
        print(f"scanning_shard={Path(shard).name}")
        with fs.open(shard, "r", encoding="utf-8") as f:
            for line in f:
                if rows_scanned >= args.max_rows:
                    stop = True
                    break
                rows_scanned += 1
                try:
                    row = json.loads(line)
                except Exception:
                    bad_rows += 1
                    continue

                text = row.get("text", "") or ""
                meta = row.get("meta", {}) or {}
                if not isinstance(meta, dict) or not text.strip():
                    bad_rows += 1
                    continue
                ncode = str(meta.get("id", "") or "").strip().upper()
                if not ncode:
                    missing_ncode += 1
                    continue

                # Sampling restriction only: remove works explicitly labelled as
                # secondary creations before any human/nonhuman relation test.
                if is_explicit_secondary_work(meta):
                    excluded_secondary += 1
                    continue

                evidence = relation_evidence_blocks(
                    text[:args.scan_chars], max_sentences=args.max_sentences
                )
                if not evidence:
                    continue

                qualifying_rows += 1
                evidence_blocks_total += len(evidence)
                item = {"text": text, "meta": meta}
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

    unique_ncodes = len({str(x.get("meta", {}).get("id", "")).upper() for x in reservoir})
    selected_secondary = sum(is_explicit_secondary_work(x.get("meta", {})) for x in reservoir)

    print(f"source_repo={args.repo}")
    print("sampling_unit=novel")
    print("work_id=meta.id_ncode")
    print(f"shards_available={len(files)}")
    print(f"work_rows_scanned={rows_scanned}")
    print(f"excluded_explicit_secondary_works={excluded_secondary}")
    print(f"qualifying_works={qualifying_rows}")
    print(f"evidence_blocks_total={evidence_blocks_total}")
    print(f"selected_works={len(reservoir)}")
    print(f"unique_selected_ncodes={unique_ncodes}")
    print(f"selected_explicit_secondary_works={selected_secondary}")
    print(f"bad_rows={bad_rows}")
    print(f"missing_ncode={missing_ncode}")
    print("corpus_restriction=exclude_explicit_secondary_creation_keyword")
    print("qualification=explicit_human_nonhuman_relationship")
    print(f"membership_scan_chars={args.scan_chars}")
    print(f"max_sentences={args.max_sentences}")
    print("sampling=reservoir_over_scanned_works")
    print(f"seed={args.seed}")
    print(f"output={out}")

    if selected_secondary:
        raise SystemExit("Invariant failed: explicit secondary works remained in selected corpus")


if __name__ == "__main__":
    main()
