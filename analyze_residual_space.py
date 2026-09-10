#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Remove corpus-wide common semantic directions from chunk embeddings and
reconstruct work-level residual spaces.

This is deliberately hypothesis-light: common directions are learned from all
selected-work chunks without literary labels.  We then test m in {0,1,2,3,5,10}
and compare geometry plus metadata confounds.  Corpus-membership lexicons never
enter this stage.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

OUT = Path("results")
REMOVE_MS = [0,1,2,3,5,10]
TOP_WORK_PCS = 10
KNN_K = 10


def unit_rows(x):
    n=np.linalg.norm(x,axis=1,keepdims=True)
    return x/np.maximum(n,1e-12)


def work_centroids(X, manifest):
    ids=manifest["candidate_id"].drop_duplicates().tolist()
    arr=[]
    mid=manifest["candidate_id"].to_numpy()
    for wid in ids:
        arr.append(X[mid==wid].mean(axis=0))
    return ids, unit_rows(np.asarray(arr,dtype=np.float32))


def cosine_D(X):
    X=unit_rows(X)
    D=1-np.clip(X@X.T,-1,1)
    np.fill_diagonal(D,0)
    return D.astype(np.float32)


def knn_overlap(Da,Db,k=10):
    k=min(k,len(Da)-1)
    vals=[]
    for i in range(len(Da)):
        a=set(np.argsort(Da[i])[1:k+1]); b=set(np.argsort(Db[i])[1:k+1])
        vals.append(len(a&b)/k)
    return float(np.mean(vals))


def parse_numeric(s):
    return pd.to_numeric(s,errors="coerce")


def eta_squared(values, groups):
    tmp=pd.DataFrame({"v":values,"g":groups}).dropna()
    if len(tmp)<10 or tmp.g.nunique()<2: return np.nan
    counts=tmp.g.value_counts()
    keep=counts[counts>=5].index
    tmp=tmp[tmp.g.isin(keep)]
    if len(tmp)<10 or tmp.g.nunique()<2: return np.nan
    grand=tmp.v.mean()
    ss_total=((tmp.v-grand)**2).sum()
    if ss_total<=0: return np.nan
    ss_between=sum(len(g)*(g.v.mean()-grand)**2 for _,g in tmp.groupby("g"))
    return float(ss_between/ss_total)


def metadata_confound_rows(scores, meta, variant, m):
    joined=scores.merge(meta,on="candidate_id",how="left")
    rows=[]
    numeric=[]
    for col in ["body_text_chars","chapters","points"]:
        if col in joined:
            vals=parse_numeric(joined[col])
            if vals.notna().sum()>=20:
                numeric.append((col,vals))
    categorical=[c for c in ["genre","biggenre"] if c in joined]
    for pc in [f"PC{i}" for i in range(1,TOP_WORK_PCS+1) if f"PC{i}" in joined]:
        for name,vals in numeric:
            mask=vals.notna() & joined[pc].notna()
            rho,p=spearmanr(vals[mask],joined.loc[mask,pc]) if mask.sum()>=20 else (np.nan,np.nan)
            rows.append({"variant":variant,"removed_common_pcs":m,"pc":pc,"confound":name,"measure":"spearman_rho","value":rho,"p":p,"n":int(mask.sum())})
        for name in categorical:
            eta=eta_squared(joined[pc],joined[name].astype(str))
            rows.append({"variant":variant,"removed_common_pcs":m,"pc":pc,"confound":name,"measure":"eta_squared","value":eta,"p":np.nan,"n":int(joined[pc].notna().sum())})
    return rows


def analyze_variant(name,X,manifest,candidates):
    # Center chunks once, then learn common semantic directions from all chunks.
    center=X.mean(axis=0,keepdims=True)
    Xc=X-center
    pca_common=PCA(n_components=max(REMOVE_MS),random_state=42).fit(Xc)
    V=pca_common.components_
    common=pd.DataFrame({"component":np.arange(1,len(V)+1),"explained_variance_ratio":pca_common.explained_variance_ratio_})
    common.to_csv(OUT/f"common_chunk_pca_spectrum_{name}.csv",index=False)

    geometry=[]; confounds=[]; distance_mats={}; score_tables={}
    base_ids=None
    for m in REMOVE_MS:
        if m==0:
            Xr=X.copy()
        else:
            # Remove only the directions, not the corpus mean, so the operation is a projection
            # in the original embedding coordinate system.
            proj=(X@V[:m].T)@V[:m]
            Xr=X-proj
        ids,W=work_centroids(Xr,manifest)
        if base_ids is None: base_ids=ids
        D=cosine_D(W); distance_mats[m]=D
        np.save(OUT/f"work_mean_embeddings_{name}_remove{m}.npy",W)

        ncomp=min(50,W.shape[0]-1,W.shape[1])
        wpca=PCA(n_components=ncomp,random_state=42).fit(W)
        Z=wpca.transform(W)
        spec=pd.DataFrame({"pc":np.arange(1,ncomp+1),"explained_variance_ratio":wpca.explained_variance_ratio_,"cumulative_explained_variance":np.cumsum(wpca.explained_variance_ratio_)})
        spec.to_csv(OUT/f"residual_work_pca_spectrum_{name}_remove{m}.csv",index=False)
        scores=pd.DataFrame({"candidate_id":ids})
        for j in range(min(TOP_WORK_PCS,Z.shape[1])): scores[f"PC{j+1}"]=Z[:,j]
        scores.to_csv(OUT/f"residual_work_pca_scores_{name}_remove{m}.csv",index=False)
        score_tables[m]=scores

        ev=wpca.explained_variance_ratio_
        pr=float((ev.sum()**2)/np.sum(ev**2)) if np.sum(ev**2)>0 else np.nan
        geometry.append({
            "variant":name,"removed_common_pcs":m,"n_works":len(ids),
            "pc1_evr":ev[0],"pc2_evr":ev[1],"pc3_evr":ev[2],
            "cum_evr_pc3":ev[:3].sum(),"cum_evr_pc10":ev[:10].sum(),
            "participation_ratio_50pc":pr,
            "median_nn_distance":float(np.median(np.sort(D,axis=1)[:,1])),
            "median_10nn_distance":float(np.median(np.sort(D,axis=1)[:,min(10,len(ids)-1)])),
        })
        confounds += metadata_confound_rows(scores,candidates,name,m)

    # Stability relative to full-space m=0, and successive residualizations.
    stab=[]
    tri=np.triu_indices(len(base_ids),1)
    for m in REMOVE_MS[1:]:
        rho,p=spearmanr(distance_mats[0][tri],distance_mats[m][tri])
        stab.append({"variant":name,"comparison":"vs_full","removed_common_pcs":m,"distance_spearman":rho,"p":p,"knn_overlap_k10":knn_overlap(distance_mats[0],distance_mats[m],KNN_K)})
    for a,b in zip(REMOVE_MS[:-1],REMOVE_MS[1:]):
        rho,p=spearmanr(distance_mats[a][tri],distance_mats[b][tri])
        stab.append({"variant":name,"comparison":f"remove{a}_vs_remove{b}","removed_common_pcs":b,"distance_spearman":rho,"p":p,"knn_overlap_k10":knn_overlap(distance_mats[a],distance_mats[b],KNN_K)})

    pd.DataFrame(geometry).to_csv(OUT/f"residual_geometry_{name}.csv",index=False)
    pd.DataFrame(confounds).to_csv(OUT/f"residual_metadata_confounds_{name}.csv",index=False)
    pd.DataFrame(stab).to_csv(OUT/f"residual_space_stability_{name}.csv",index=False)
    return base_ids,distance_mats,geometry,confounds


def main():
    manifest=pd.read_csv(OUT/"chunk_manifest.csv")
    candidates=pd.read_csv(OUT/"candidates.csv")
    Xr=np.load(OUT/"embeddings_raw.npy")
    Xa=np.load(OUT/"embeddings_anonymized.npy")
    if len(Xr)!=len(manifest) or len(Xa)!=len(manifest): raise SystemExit("embedding/manifest mismatch")

    ids_r,Dr,gr,cr=analyze_variant("raw",Xr,manifest,candidates)
    ids_a,Da,ga,ca=analyze_variant("anonymized",Xa,manifest,candidates)
    if ids_r!=ids_a: raise SystemExit("work order mismatch")

    # Raw-vs-anonymized robustness after each amount of common-component removal.
    rows=[]; tri=np.triu_indices(len(ids_r),1)
    for m in REMOVE_MS:
        rho,p=spearmanr(Dr[m][tri],Da[m][tri])
        rows.append({"removed_common_pcs":m,"distance_spearman_raw_vs_anonymized":rho,"p":p,"knn_overlap_k10_raw_vs_anonymized":knn_overlap(Dr[m],Da[m],KNN_K)})
    pd.DataFrame(rows).to_csv(OUT/"residual_raw_vs_anonymized_robustness.csv",index=False)

    # Compact summary of maximum measured confounding per m.
    cdf=pd.DataFrame(cr+ca)
    summ=[]
    for (variant,m),g in cdf.groupby(["variant","removed_common_pcs"]):
        for measure,h in g.groupby("measure"):
            h=h.dropna(subset=["value"])
            if h.empty: continue
            idx=h["value"].abs().idxmax() if measure=="spearman_rho" else h["value"].idxmax()
            r=h.loc[idx]
            summ.append({"variant":variant,"removed_common_pcs":m,"measure":measure,"max_abs_or_max_value":abs(r.value) if measure=="spearman_rho" else r.value,"pc":r.pc,"confound":r.confound})
    pd.DataFrame(summ).to_csv(OUT/"residual_confound_summary.csv",index=False)

    print(pd.DataFrame(gr+ga).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(summ).to_string(index=False))
    print("Residualization uses only corpus-wide chunk PCs; no literary labels or retrieval lexicons enter the projection.")

if __name__=="__main__":
    main()
