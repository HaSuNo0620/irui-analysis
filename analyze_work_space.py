#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Construct work-level semantic representations from cached chunk embeddings.

The literary work, not the chunk, is the analysis unit.  Each work is represented
in two complementary hypothesis-light ways inside the shared Ruri space:

1. centroid (mean direction of its chunk embeddings), and
2. a low-rank local subspace estimated from centered chunk embeddings.

No folklore/literary labels or retrieval lexicons enter these representations.
They are only used upstream for corpus membership.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import umap
from scipy.stats import spearmanr


OUT = Path("results")
SUBSPACE_RANK = 3
KNN_K = 10


def unit_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def cosine_distance_matrix(x):
    u = unit_rows(x)
    d = 1.0 - np.clip(u @ u.T, -1.0, 1.0)
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32)


def work_representations(X, manifest, rank=SUBSPACE_RANK):
    work_ids = manifest["candidate_id"].drop_duplicates().tolist()
    means = []
    stats = []
    bases = {}

    for wid in work_ids:
        idx = np.flatnonzero(manifest["candidate_id"].to_numpy() == wid)
        Xi = X[idx]
        mu = Xi.mean(axis=0)
        means.append(mu)

        centered = Xi - mu
        # Mean pairwise cosine distance is a simple within-work semantic spread.
        if len(Xi) >= 2:
            U = unit_rows(Xi)
            sim = U @ U.T
            tri = np.triu_indices(len(Xi), 1)
            spread = float(np.mean(1.0 - sim[tri]))
        else:
            spread = 0.0

        singular_values = np.array([], dtype=float)
        if len(Xi) >= rank + 1:
            _, s, vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = s
            # Columns are orthonormal basis vectors in the original embedding space.
            bases[wid] = vt[:rank].T.astype(np.float32)

        total_var = float(np.sum(singular_values ** 2)) if singular_values.size else 0.0
        top_var = float(np.sum(singular_values[:rank] ** 2)) if singular_values.size else 0.0
        stats.append({
            "candidate_id": wid,
            "n_chunks": int(len(Xi)),
            "mean_pairwise_cosine_distance": spread,
            "subspace_rank_available": int(min(rank, max(0, len(Xi) - 1))),
            "top_rank_variance_fraction": (top_var / total_var) if total_var > 0 else np.nan,
        })

    means = unit_rows(np.asarray(means, dtype=np.float32))
    return work_ids, means, pd.DataFrame(stats), bases


def subspace_distance_matrix(work_ids, bases, rank=SUBSPACE_RANK):
    eligible = [w for w in work_ids if w in bases]
    n = len(eligible)
    D = np.zeros((n, n), dtype=np.float32)
    angle_mean = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        Ui = bases[eligible[i]]
        for j in range(i + 1, n):
            Uj = bases[eligible[j]]
            s = np.linalg.svd(Ui.T @ Uj, compute_uv=False)
            s = np.clip(s, 0.0, 1.0)
            theta = np.arccos(s)
            # RMS principal angle: 0 means identical local semantic directions.
            d = float(np.sqrt(np.mean(theta ** 2)))
            m = float(np.mean(theta))
            D[i, j] = D[j, i] = d
            angle_mean[i, j] = angle_mean[j, i] = m
    return eligible, D, angle_mean


def embed_2d_from_vectors(X):
    return umap.UMAP(
        n_components=2,
        n_neighbors=min(30, max(2, len(X) - 1)),
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    ).fit_transform(X)


def embed_2d_from_distance(D):
    return umap.UMAP(
        n_components=2,
        n_neighbors=min(30, max(2, len(D) - 1)),
        min_dist=0.05,
        metric="precomputed",
        random_state=42,
    ).fit_transform(D)


def save_matrix(path, ids, D):
    df = pd.DataFrame(D, index=ids, columns=ids)
    df.index.name = "candidate_id"
    df.to_csv(path)


def knn_overlap(Da, Db, k=KNN_K):
    n = len(Da)
    k = min(k, max(1, n - 1))
    vals = []
    for i in range(n):
        a = set(np.argsort(Da[i])[1:k + 1])
        b = set(np.argsort(Db[i])[1:k + 1])
        vals.append(len(a & b) / k)
    return float(np.mean(vals))


def analyze_variant(name, X, manifest, candidate_meta):
    ids, means, stats, bases = work_representations(X, manifest)
    mean_D = cosine_distance_matrix(means)
    coords = embed_2d_from_vectors(means)

    base = candidate_meta.set_index("candidate_id").reindex(ids).reset_index()
    out = base[[c for c in ["candidate_id", "ncode", "title", "author", "chapters", "body_text_chars"] if c in base.columns]].copy()
    out = out.merge(stats, on="candidate_id", how="left")
    out["umap_mean_x"] = coords[:, 0]
    out["umap_mean_y"] = coords[:, 1]
    out.to_csv(OUT / f"work_semantic_map_{name}.csv", index=False)
    np.save(OUT / f"work_mean_embeddings_{name}.npy", means)
    save_matrix(OUT / f"work_mean_cosine_distance_{name}.csv", ids, mean_D)

    eligible, sub_D, angle_mean = subspace_distance_matrix(ids, bases)
    if len(eligible) >= 3:
        sub_xy = embed_2d_from_distance(sub_D)
        sub_df = pd.DataFrame({
            "candidate_id": eligible,
            "umap_subspace_x": sub_xy[:, 0],
            "umap_subspace_y": sub_xy[:, 1],
        })
        sub_df = sub_df.merge(out, on="candidate_id", how="left")
        sub_df.to_csv(OUT / f"work_subspace_map_{name}.csv", index=False)
        save_matrix(OUT / f"work_subspace_rms_principal_angle_{name}.csv", eligible, sub_D)
        save_matrix(OUT / f"work_subspace_mean_principal_angle_{name}.csv", eligible, angle_mean)

    print(f"{name}: works={len(ids)}, subspace_rank{SUBSPACE_RANK}_works={len(eligible)}")
    return ids, mean_D, eligible, sub_D


def main():
    manifest = pd.read_csv(OUT / "chunk_manifest.csv")
    candidates = pd.read_csv(OUT / "candidates.csv")
    Xraw = np.load(OUT / "embeddings_raw.npy")
    Xanon = np.load(OUT / "embeddings_anonymized.npy")

    if len(manifest) != len(Xraw) or len(manifest) != len(Xanon):
        raise SystemExit("Embedding rows do not match chunk_manifest.csv")

    ids_r, Dr, sub_ids_r, Sr = analyze_variant("raw", Xraw, manifest, candidates)
    ids_a, Da, sub_ids_a, Sa = analyze_variant("anonymized", Xanon, manifest, candidates)

    if ids_r != ids_a:
        raise SystemExit("Raw/anonymized work order mismatch")

    tri = np.triu_indices(len(Dr), 1)
    rho, p = spearmanr(Dr[tri], Da[tri])
    rows = [
        {"metric": "mean_distance_spearman", "value": float(rho), "detail": f"p={p:.3g}"},
        {"metric": f"mean_knn_overlap_k{KNN_K}", "value": knn_overlap(Dr, Da, KNN_K), "detail": "raw vs anonymized"},
    ]

    common = [w for w in sub_ids_r if w in set(sub_ids_a)]
    if len(common) >= 3:
        ir = {w:i for i,w in enumerate(sub_ids_r)}
        ia = {w:i for i,w in enumerate(sub_ids_a)}
        rr = np.array([ir[w] for w in common])
        aa = np.array([ia[w] for w in common])
        Src = Sr[np.ix_(rr, rr)]
        Sac = Sa[np.ix_(aa, aa)]
        tri2 = np.triu_indices(len(common), 1)
        rho_s, p_s = spearmanr(Src[tri2], Sac[tri2])
        rows.extend([
            {"metric": "subspace_distance_spearman", "value": float(rho_s), "detail": f"rank={SUBSPACE_RANK}; p={p_s:.3g}; n={len(common)}"},
            {"metric": f"subspace_knn_overlap_k{KNN_K}", "value": knn_overlap(Src, Sac, KNN_K), "detail": f"rank={SUBSPACE_RANK}; n={len(common)}"},
        ])

    pd.DataFrame(rows).to_csv(OUT / "work_space_raw_vs_anonymized_robustness.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Work-level representations use only full-body chunk embeddings; no literary labels enter the space.")


if __name__ == "__main__":
    main()
