#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit WebNovels-Ja row/schema structure without saving source text.

Outputs only schema/statistics and short hashed continuity diagnostics. No novel text is
written to artifacts.
"""

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

ID_WORDS = ("id", "novel", "work", "story", "author", "user", "url", "ncode", "episode", "chapter", "part", "title", "name", "date", "time", "publish")


def flatten(obj, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten(v, key))
    elif isinstance(obj, list):
        out.append((prefix, "list", len(obj), None))
    else:
        out.append((prefix, type(obj).__name__, None, obj))
    return out


def safe_summary(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    if re.search(r"https?://", s):
        return "<URL>"
    if len(s) <= 40 and re.fullmatch(r"[A-Za-z0-9_.:/@+-]+", s):
        return s
    return f"<str len={len(s)}>"


def digest(s):
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def common_prefix_len(a, b, limit=500):
    n = min(len(a), len(b), limit)
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def common_suffix_prefix(a, b, max_overlap=500):
    m = min(len(a), len(b), max_overlap)
    for k in range(m, 19, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="OmniAICreator/WebNovels-Ja")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--output-dir", default="structure_audit")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.repo, split=args.split, streaming=True, token=token)

    top_keys = Counter()
    flat_keys = Counter()
    flat_types = defaultdict(Counter)
    examples = defaultdict(list)
    candidate_values = defaultdict(Counter)
    text_lengths = []
    meta_key_counts = Counter()
    row_count = 0
    prev_text = None
    pair_stats = []

    for row in ds:
        if row_count >= args.rows:
            break
        row_count += 1
        for k in row.keys():
            top_keys[k] += 1

        text = row.get("text", "") or ""
        text_lengths.append(len(text))

        for key, typ, list_len, value in flatten(row):
            flat_keys[key] += 1
            flat_types[key][typ] += 1
            if len(examples[key]) < 5:
                examples[key].append(safe_summary(value if typ != "list" else f"list_len={list_len}"))
            low = key.lower()
            if any(w in low for w in ID_WORDS) and value is not None and not isinstance(value, (dict, list)):
                s = str(value)
                if len(s) <= 200:
                    candidate_values[key][s] += 1

        meta = row.get("meta", {}) or {}
        if isinstance(meta, dict):
            for k in meta.keys():
                meta_key_counts[k] += 1

        if prev_text is not None:
            pair_stats.append({
                "pair_index": row_count - 1,
                "prev_len": len(prev_text),
                "curr_len": len(text),
                "prefix_equal_chars": common_prefix_len(prev_text, text),
                "suffix_to_prefix_overlap": common_suffix_prefix(prev_text, text),
                "prev_head_hash": digest(prev_text[:200]),
                "curr_head_hash": digest(text[:200]),
            })
        prev_text = text

    lengths = np.asarray(text_lengths, dtype=float)
    q = [0, .01, .1, .25, .5, .75, .9, .99, 1]
    length_stats = [{"quantile": x, "chars": float(np.quantile(lengths, x)) if len(lengths) else None} for x in q]

    with (out / "field_schema.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["field", "present_rows", "presence_rate", "types", "examples"])
        w.writeheader()
        for key in sorted(flat_keys):
            w.writerow({
                "field": key,
                "present_rows": flat_keys[key],
                "presence_rate": flat_keys[key] / max(row_count, 1),
                "types": json.dumps(flat_types[key], ensure_ascii=False),
                "examples": json.dumps(examples[key], ensure_ascii=False),
            })

    with (out / "meta_keys.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["meta_key", "rows", "presence_rate"])
        w.writeheader()
        for k, n in meta_key_counts.most_common():
            w.writerow({"meta_key": k, "rows": n, "presence_rate": n / max(row_count, 1)})

    with (out / "candidate_identifier_fields.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["field", "observed", "unique", "duplicate_values", "max_frequency", "uniqueness_ratio"])
        w.writeheader()
        for key, cnt in sorted(candidate_values.items()):
            observed = sum(cnt.values())
            unique = len(cnt)
            w.writerow({
                "field": key,
                "observed": observed,
                "unique": unique,
                "duplicate_values": sum(1 for v in cnt.values() if v > 1),
                "max_frequency": max(cnt.values()) if cnt else 0,
                "uniqueness_ratio": unique / max(observed, 1),
            })

    with (out / "text_length_quantiles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quantile", "chars"])
        w.writeheader(); w.writerows(length_stats)

    with (out / "adjacent_row_continuity.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(pair_stats[0].keys()) if pair_stats else ["pair_index"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(pair_stats[:1000])

    summary = {
        "repo": args.repo,
        "split": args.split,
        "rows_audited": row_count,
        "top_level_keys": dict(top_keys),
        "meta_keys": dict(meta_key_counts),
        "text_length": {
            "mean": float(lengths.mean()) if len(lengths) else None,
            "median": float(np.median(lengths)) if len(lengths) else None,
            "min": float(lengths.min()) if len(lengths) else None,
            "max": float(lengths.max()) if len(lengths) else None,
        },
        "adjacent_pairs_with_20plus_overlap": sum(r["suffix_to_prefix_overlap"] >= 20 for r in pair_stats),
        "adjacent_pairs_with_same_head_hash": sum(r["prev_head_hash"] == r["curr_head_hash"] for r in pair_stats),
        "note": "No source novel text is stored in audit outputs.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
