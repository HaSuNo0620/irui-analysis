#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One-command local downstream analysis runner.

Usage:
    python run_analysis.py
    python run_analysis.py --config local_analysis.yaml

The runner never downloads the corpus or recomputes embeddings. It requires the
cached semantic base in results/ and executes only lightweight downstream steps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_cache(results: Path, names: list[str]) -> None:
    missing = [str(results / n) for n in names if not (results / n).exists()]
    if missing:
        msg = "Missing cached semantic-base files:\n  " + "\n  ".join(missing)
        msg += "\nPlace the run-33 cache in results/ before running downstream analysis."
        raise SystemExit(msg)


def compact_summary(results: Path, primary_remove: int) -> dict:
    summary: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_remove_common_pcs": primary_remove,
        "database_access": False,
        "embedding_recomputed": False,
    }

    p = results / "residual_raw_vs_anonymized_robustness.csv"
    if p.exists():
        df = pd.read_csv(p)
        row = df[df["removed_common_pcs"] == primary_remove]
        if not row.empty:
            r = row.iloc[0]
            summary["primary_raw_anonymized"] = {
                "distance_spearman": float(r["distance_spearman_raw_vs_anonymized"]),
                "knn_overlap_k10": float(r["knn_overlap_k10_raw_vs_anonymized"]),
            }

    p = results / "residual_confound_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df[df["removed_common_pcs"] == primary_remove]
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "variant": str(r["variant"]),
                "measure": str(r["measure"]),
                "max_abs_or_max_value": float(r["max_abs_or_max_value"]),
                "pc": str(r["pc"]),
                "confound": str(r["confound"]),
            })
        summary["primary_confound_maxima"] = rows

    for variant in ("raw", "anonymized"):
        p = results / f"residual_axis_extreme_works_{variant}_remove{primary_remove}.csv"
        if p.exists():
            df = pd.read_csv(p)
            summary[f"{variant}_axis_extremes_file"] = p.name
            summary[f"{variant}_n_extreme_rows"] = int(len(df))

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="local_analysis.yaml")
    ap.add_argument("--skip", action="append", default=[],
                    choices=["work_space", "geometry", "residual", "interpret_axes"])
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    results = Path(cfg.get("paths", {}).get("results_dir", "results"))
    results.mkdir(parents=True, exist_ok=True)
    require_cache(results, cfg.get("required_cache_files", []))

    stages = cfg.get("stages", {})
    py = sys.executable

    if stages.get("work_space", True) and "work_space" not in args.skip:
        run([py, "analyze_work_space.py"])
    if stages.get("geometry", True) and "geometry" not in args.skip:
        run([py, "analyze_work_geometry.py"])
    if stages.get("residual", True) and "residual" not in args.skip:
        run([py, "analyze_residual_space.py"])

    rcfg = cfg.get("residual", {})
    primary = int(rcfg.get("primary_remove_common_pcs", 3))
    pcs = int(rcfg.get("pcs_to_interpret", 10))
    extremes = int(rcfg.get("extreme_works_per_side", 10))
    variants = rcfg.get("variants", ["raw", "anonymized"])

    if stages.get("interpret_axes", True) and "interpret_axes" not in args.skip:
        for variant in variants:
            run([
                py, "interpret_residual_axes.py",
                "--remove", str(primary),
                "--pcs", str(pcs),
                "--extremes", str(extremes),
                "--variant", str(variant),
            ])

    summary = compact_summary(results, primary)
    if cfg.get("runtime", {}).get("write_summary_json", True):
        with (results / "local_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nLocal downstream analysis completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
