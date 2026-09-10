#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rank the extremes of residual work-level semantic PCs.

The corpus-wide chunk PCs are removed first (default m=3).  PCA is then fit to
work centroids in that residual space.  For each work-space PC, this script
reports:

- works at the positive and negative extremes;
- the chunk in each extreme work with the largest/smallest projection on the
  same work-space PC direction;
- chunk position so a small number of passages can later be inspected without
  re-scanning the whole corpus.

No retrieval lexicons or literary labels enter the construction of the axes.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

OUT = Path("results")


def unit_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def residualize_chunks(X, m):
    # Reproduce the residual-space construction: learn common directions from
    # all selected-work chunks, then project them out.
    pca = PCA(n_components=max(10, m), random_state=42)
    pca.fit(X)
    Xc = X - pca.mean_
    if m > 0:
        V = pca.components_[:m]
        Xr = Xc - (Xc @ V.T) @ V
    else:
        Xr = Xc
    return Xr


def build_work_means(Xr, manifest):
    work_ids = manifest["candidate_id"].drop_duplicates().tolist()
    means=[]
    index_map={}
    ids_arr=manifest["candidate_id"].astype(str).to_numpy()
    for wid in work_ids:
        idx=np.flatnonzero(ids_arr == str(wid))
        index_map[str(wid)] = idx
        means.append(Xr[idx].mean(axis=0))
    means=unit_rows(np.asarray(means, dtype=np.float32))
    return [str(x) for x in work_ids], means, index_map


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--remove", type=int, default=3)
    ap.add_argument("--pcs", type=int, default=10)
    ap.add_argument("--extremes", type=int, default=10)
    ap.add_argument("--variant", choices=["raw","anonymized"], default="anonymized")
    args=ap.parse_args()

    X=np.load(OUT / f"embeddings_{args.variant}.npy")
    manifest=pd.read_csv(OUT / "chunk_manifest.csv")
    candidates=pd.read_csv(OUT / "candidates.csv")
    manifest["candidate_id"]=manifest["candidate_id"].astype(str)
    candidates["candidate_id"]=candidates["candidate_id"].astype(str)

    if len(X) != len(manifest):
        raise SystemExit("embedding/manifest row mismatch")

    Xr=residualize_chunks(X, args.remove)
    work_ids, means, idx_map=build_work_means(Xr, manifest)

    ncomp=min(args.pcs, len(work_ids)-1, means.shape[1])
    wpca=PCA(n_components=ncomp, random_state=42)
    scores=wpca.fit_transform(means)
    meta=candidates.drop_duplicates("candidate_id").set_index("candidate_id")

    rows=[]
    chunk_rows=[]
    for k in range(ncomp):
        pc=k+1
        order=np.argsort(scores[:,k])
        selections=[("negative", order[:args.extremes]),
                    ("positive", order[-args.extremes:][::-1])]
        direction=wpca.components_[k]

        for side, inds in selections:
            for rank, wi in enumerate(inds, start=1):
                wid=work_ids[wi]
                mrow=meta.loc[wid] if wid in meta.index else pd.Series(dtype=object)
                rows.append({
                    "variant":args.variant,
                    "removed_common_pcs":args.remove,
                    "pc":pc,
                    "side":side,
                    "rank":rank,
                    "candidate_id":wid,
                    "ncode":mrow.get("ncode", ""),
                    "title":mrow.get("title", ""),
                    "author":mrow.get("author", ""),
                    "genre":mrow.get("genre", ""),
                    "biggenre":mrow.get("biggenre", ""),
                    "body_text_chars":mrow.get("body_text_chars", np.nan),
                    "work_pc_score":float(scores[wi,k]),
                })

                idx=idx_map[wid]
                proj=Xr[idx] @ direction
                # Pick the chunk that is maximally aligned with the same side.
                local=int(np.argmax(proj) if side=="positive" else np.argmin(proj))
                gi=idx[local]
                mr=manifest.iloc[gi]
                chunk_rows.append({
                    "variant":args.variant,
                    "removed_common_pcs":args.remove,
                    "pc":pc,
                    "side":side,
                    "work_rank":rank,
                    "candidate_id":wid,
                    "ncode":mrow.get("ncode", ""),
                    "title":mrow.get("title", ""),
                    "chunk_idx":int(mr.get("chunk_idx", local)),
                    "position":float(mr.get("position", np.nan)),
                    "start_char":mr.get("start_char", np.nan),
                    "end_char":mr.get("end_char", np.nan),
                    "chunk_pc_projection":float(proj[local]),
                    "work_pc_score":float(scores[wi,k]),
                })

    pd.DataFrame(rows).to_csv(OUT / f"residual_axis_extreme_works_{args.variant}_remove{args.remove}.csv", index=False)
    pd.DataFrame(chunk_rows).to_csv(OUT / f"residual_axis_extreme_chunks_{args.variant}_remove{args.remove}.csv", index=False)

    spec=pd.DataFrame({
        "pc":np.arange(1,ncomp+1),
        "explained_variance_ratio":wpca.explained_variance_ratio_,
        "cumulative_explained_variance":np.cumsum(wpca.explained_variance_ratio_),
    })
    spec.to_csv(OUT / f"residual_axis_spectrum_{args.variant}_remove{args.remove}.csv", index=False)

    print(f"variant={args.variant} remove={args.remove} works={len(work_ids)}")
    print("Top residual-axis extremes:")
    preview=pd.DataFrame(rows)
    print(preview[preview.pc<=3][["pc","side","rank","ncode","title","work_pc_score"]].to_string(index=False))
    print("No literary labels or selection lexicons enter residual-axis fitting.")


if __name__ == "__main__":
    main()
