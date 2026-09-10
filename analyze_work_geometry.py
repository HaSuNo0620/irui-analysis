#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Diagnose whether work-level semantic structure is continuous or island-like.

Uses only cached work mean embeddings. No retrieval lexicons or literary labels
enter the geometry. Reports PCA spectrum/scores, kNN connectivity, nearest-neighbor
scales, and Euclidean MST edge gaps in the PCA space.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

OUT = Path("results")
MAX_PC = 30


def participation_ratio(eig):
    eig = np.asarray(eig, float)
    s = eig.sum()
    return float(s*s / np.sum(eig*eig)) if s > 0 else np.nan


def analyze(name):
    X = np.load(OUT / f"work_mean_embeddings_{name}.npy")
    meta = pd.read_csv(OUT / f"work_semantic_map_{name}.csv")
    ncomp = min(MAX_PC, X.shape[0]-1, X.shape[1])
    pca = PCA(n_components=ncomp, random_state=42)
    Z = pca.fit_transform(X)

    eig = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    spectrum = pd.DataFrame({
        "pc": np.arange(1, ncomp+1),
        "eigenvalue": eig,
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": np.cumsum(evr),
    })
    spectrum.to_csv(OUT / f"work_pca_spectrum_{name}.csv", index=False)

    keep = [c for c in ["candidate_id","ncode","title","author","chapters","body_text_chars"] if c in meta.columns]
    scores = meta[keep].copy()
    for j in range(min(10, ncomp)):
        scores[f"PC{j+1}"] = Z[:, j]
    scores.to_csv(OUT / f"work_pca_scores_{name}.csv", index=False)

    # Quantiles reveal gaps without imposing bins or clusters.
    qrows = []
    probs = np.linspace(0, 1, 101)
    for j in range(min(10, ncomp)):
        vals = Z[:, j]
        qs = np.quantile(vals, probs)
        for p, q in zip(probs, qs):
            qrows.append({"pc":j+1,"quantile":p,"score":q})
    pd.DataFrame(qrows).to_csv(OUT / f"work_pca_quantiles_{name}.csv", index=False)

    # Work in PCA coordinates retaining >=90% variance if possible, otherwise all computed PCs.
    d90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    d90 = min(d90, ncomp)
    Y = Z[:, :d90]
    D = pairwise_distances(Y, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = np.min(D, axis=1)
    kth10 = np.partition(D, min(9, len(D)-2), axis=1)[:, min(9, len(D)-2)]
    pd.DataFrame({
        "candidate_id": meta["candidate_id"],
        "nearest_neighbor_distance": nn,
        "tenth_neighbor_distance": kth10,
    }).to_csv(OUT / f"work_local_distance_{name}.csv", index=False)

    connectivity = []
    for k in [2,3,5,10,20,30]:
        k = min(k, len(X)-1)
        G = kneighbors_graph(Y, n_neighbors=k, mode="connectivity", include_self=False)
        G = G.maximum(G.T)
        ncc, labels = connected_components(G, directed=False)
        sizes = np.bincount(labels)
        connectivity.append({
            "k":k,
            "n_components":int(ncc),
            "largest_component_fraction":float(sizes.max()/len(X)),
            "second_largest_component_fraction":float(np.sort(sizes)[-2]/len(X)) if len(sizes)>1 else 0.0,
        })
    pd.DataFrame(connectivity).to_csv(OUT / f"work_knn_connectivity_{name}.csv", index=False)

    # MST: a genuine island separation tends to require unusually long bridge edges.
    Dfinite = D.copy(); np.fill_diagonal(Dfinite, 0.0)
    mst = minimum_spanning_tree(Dfinite)
    edges = np.asarray(mst.data, float)
    edges.sort()
    median_edge = float(np.median(edges)) if len(edges) else np.nan
    max_edge = float(edges[-1]) if len(edges) else np.nan
    max_jump_ratio = float(max_edge/median_edge) if median_edge > 0 else np.nan
    edge_df = pd.DataFrame({"rank":np.arange(1,len(edges)+1),"edge_length":edges})
    edge_df.to_csv(OUT / f"work_mst_edges_{name}.csv", index=False)

    summary = pd.DataFrame([{
        "variant":name,
        "n_works":len(X),
        "pc1_evr":float(evr[0]),
        "pc2_evr":float(evr[1]) if ncomp>1 else np.nan,
        "pc3_evr":float(evr[2]) if ncomp>2 else np.nan,
        "cum_evr_pc3":float(np.sum(evr[:3])),
        "cum_evr_pc10":float(np.sum(evr[:10])),
        "pcs_for_50pct":int(np.searchsorted(np.cumsum(evr),0.50)+1) if np.cumsum(evr)[-1]>=0.5 else -1,
        "pcs_for_80pct":int(np.searchsorted(np.cumsum(evr),0.80)+1) if np.cumsum(evr)[-1]>=0.8 else -1,
        "pcs_for_90pct":d90 if np.cumsum(evr)[-1]>=0.9 else -1,
        "participation_ratio":participation_ratio(eig),
        "median_nn_distance":float(np.median(nn)),
        "median_10nn_distance":float(np.median(kth10)),
        "mst_median_edge":median_edge,
        "mst_max_edge":max_edge,
        "mst_max_to_median_ratio":max_jump_ratio,
    }])
    summary.to_csv(OUT / f"work_geometry_summary_{name}.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(connectivity).to_string(index=False))


def main():
    analyze("raw")
    analyze("anonymized")
    print("Geometry uses cached work centroids only; no literary labels or clustering assumptions are used.")

if __name__ == "__main__":
    main()
