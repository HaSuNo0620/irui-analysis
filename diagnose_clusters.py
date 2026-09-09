#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
import hdbscan


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    out = Path("results")
    X = np.load(out / "embeddings.npy")
    manifest = pd.read_csv(out / "chunk_manifest.csv")

    ncomp = min(cfg["pca_components"], X.shape[1], max(2, X.shape[0] - 1))
    Xp = PCA(n_components=ncomp, random_state=cfg["random_state"]).fit_transform(X)

    mcs_values = cfg.get("hdbscan_diagnostic_min_cluster_sizes", [5, 8, 10, 15, 20])
    ms_values = cfg.get("hdbscan_diagnostic_min_samples", [1, 2, 3, 5])

    rows = []
    assignments = []
    for mcs in mcs_values:
        for ms in ms_values:
            labels = hdbscan.HDBSCAN(
                min_cluster_size=int(mcs),
                min_samples=int(ms),
            ).fit_predict(Xp)

            non_noise = labels != -1
            clusters = sorted(set(labels) - {-1})
            rows.append({
                "min_cluster_size": int(mcs),
                "min_samples": int(ms),
                "n_clusters": len(clusters),
                "noise_fraction": float((labels == -1).mean()),
                "n_non_noise_chunks": int(non_noise.sum()),
                "n_works_with_non_noise": int(manifest.loc[non_noise, "candidate_id"].nunique()) if non_noise.any() else 0,
            })

            tmp = manifest[["candidate_id", "chunk_idx", "position"]].copy()
            tmp["min_cluster_size"] = int(mcs)
            tmp["min_samples"] = int(ms)
            tmp["cluster"] = labels
            assignments.append(tmp)

    pd.DataFrame(rows).to_csv(out / "hdbscan_diagnostics.csv", index=False)
    pd.concat(assignments, ignore_index=True).to_csv(out / "hdbscan_assignments_grid.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
