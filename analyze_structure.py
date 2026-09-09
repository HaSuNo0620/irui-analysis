#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, re
from collections import Counter, defaultdict
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import yaml
from janome.tokenizer import Tokenizer
from sklearn.decomposition import PCA

STOP = {
    "こと","もの","ため","よう","ところ","そう","これ","それ","あれ","ここ","そこ","どこ",
    "私","俺","僕","彼","彼女","自分","人","一つ","二つ","今日","今","時","方","何",
    "する","いる","ある","なる","れる","られる","いう","言う","思う","見る","来る","行く",
}


def read_jsonl(path):
    rows=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: rows.append({})
    return rows


def tokenize(text, tokenizer):
    out=[]
    for t in tokenizer.tokenize(text):
        pos=t.part_of_speech.split(",")[0]
        if pos not in {"名詞","動詞","形容詞"}: continue
        base=t.base_form if t.base_form != "*" else t.surface
        base=base.strip()
        if len(base) < 2 or base in STOP: continue
        if re.fullmatch(r"[\W_\d]+", base): continue
        out.append(base)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="pilot JSONL used by run_all.py")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results", default="results")
    args=ap.parse_args()

    cfg=yaml.safe_load(open(args.config, encoding="utf-8"))
    out=Path(args.results)
    X=np.load(out/"embeddings.npy")
    mdf=pd.read_csv(out/"chunk_manifest.csv")
    cdf=pd.read_csv(out/"candidates.csv")

    # Recluster at the explicitly selected interpretation working point.
    ncomp=min(cfg["pca_components"], X.shape[1], max(2, X.shape[0]-1))
    Xp=PCA(n_components=ncomp, random_state=cfg["random_state"]).fit_transform(X)
    labels=hdbscan.HDBSCAN(
        min_cluster_size=min(cfg["hdbscan_min_cluster_size"], max(2, len(X)//5)),
        min_samples=cfg.get("hdbscan_min_samples", None),
    ).fit_predict(Xp)

    res=mdf.copy()
    res["cluster_interpret"]=labels
    res.to_csv(out/"clusters_interpret.csv", index=False)

    # Cluster summary with cross-work coverage.
    rows=[]
    for k,g in res.groupby("cluster_interpret"):
        rows.append({
            "cluster":int(k),
            "n_chunks":len(g),
            "n_works":g.candidate_id.nunique(),
            "mean_position":g.position.mean(),
            "median_position":g.position.median(),
        })
    pd.DataFrame(rows).sort_values("n_chunks", ascending=False).to_csv(out/"cluster_interpret_summary.csv", index=False)

    # Representative chunks: nearest to PCA-space centroid. No source text is exported.
    reps=[]
    nrep=int(cfg.get("representatives_per_cluster",5))
    for k in sorted(set(labels)):
        if k < 0: continue
        idx=np.flatnonzero(labels==k)
        centroid=Xp[idx].mean(axis=0)
        d=np.linalg.norm(Xp[idx]-centroid, axis=1)
        for rank,jj in enumerate(idx[np.argsort(d)[:nrep]],1):
            r=res.iloc[jj]
            reps.append({
                "cluster":int(k),"rank":rank,"candidate_id":r.candidate_id,
                "chunk_idx":int(r.chunk_idx),"position":float(r.position),
                "start_char":int(r.start_char),"end_char":int(r.end_char),
                "distance_to_centroid":float(np.linalg.norm(Xp[jj]-centroid)),
            })
    pd.DataFrame(reps).to_csv(out/"cluster_representatives.csv", index=False)

    # Work-level occupancy p_i(k), including noise as a diagnostic state.
    occ=(res.groupby(["candidate_id","cluster_interpret"]).size().rename("n_chunks").reset_index())
    totals=res.groupby("candidate_id").size().rename("n_total").reset_index()
    occ=occ.merge(totals,on="candidate_id")
    occ["occupancy"]=occ.n_chunks/occ.n_total
    occ.to_csv(out/"work_cluster_occupancy.csv", index=False)

    # Position profile p(k | normalized narrative bin).
    nb=int(cfg.get("position_bins",10))
    res["position_bin"]=np.minimum((res.position.clip(0,0.999999)*nb).astype(int), nb-1)
    prof=(res.groupby(["position_bin","cluster_interpret"]).size().rename("n_chunks").reset_index())
    btot=res.groupby("position_bin").size().rename("bin_total").reset_index()
    prof=prof.merge(btot,on="position_bin")
    prof["p_cluster_given_position"]=prof.n_chunks/prof.bin_total
    prof.to_csv(out/"position_cluster_profile.csv", index=False)

    # Ordered transitions within each work.
    trans=Counter(); state_tot=Counter()
    for cid,g in res.sort_values(["candidate_id","position"]).groupby("candidate_id"):
        seq=g.cluster_interpret.astype(int).tolist()
        for a,b in zip(seq[:-1],seq[1:]):
            trans[(a,b)] += 1; state_tot[a] += 1
    trows=[]
    for (a,b),n in sorted(trans.items()):
        trows.append({"from_cluster":a,"to_cluster":b,"count":n,
                      "probability":n/state_tot[a] if state_tot[a] else 0.0})
    pd.DataFrame(trows).to_csv(out/"cluster_transitions.csv", index=False)

    # Reconstruct selected chunks only in memory for derived lexical statistics.
    source=read_jsonl(args.input)
    text_by_cid={}
    for _,r in cdf.iterrows():
        ri=int(r.row_idx)
        if 0 <= ri < len(source):
            text_by_cid[r.candidate_id]=source[ri].get("text","") or ""

    tok=Tokenizer()
    cluster_counts=defaultdict(Counter); totals=Counter()
    for _,r in res.iterrows():
        k=int(r.cluster_interpret)
        if k < 0: continue
        text=text_by_cid.get(r.candidate_id,"")
        chunk=text[int(r.start_char):int(r.end_char)]
        terms=tokenize(chunk,tok)
        cluster_counts[k].update(terms)
        totals.update(terms)

    vocab=set(totals)
    V=max(len(vocab),1)
    topn=int(cfg.get("top_terms_per_cluster",25))
    termrows=[]
    grand=sum(totals.values())
    for k,cnt in sorted(cluster_counts.items()):
        nk=sum(cnt.values()); bg=totals-cnt; nbg=max(grand-nk,0)
        scored=[]
        for term,n in cnt.items():
            # Smoothed log-frequency ratio against all other non-noise clusters.
            score=math.log((n+0.5)/(nk+0.5*V)) - math.log((bg[term]+0.5)/(nbg+0.5*V))
            scored.append((score,term,n,bg[term]))
        for rank,(score,term,n,nbgterm) in enumerate(sorted(scored,reverse=True)[:topn],1):
            termrows.append({"cluster":k,"rank":rank,"term":term,"score":score,
                             "cluster_count":n,"other_count":nbgterm})
    pd.DataFrame(termrows).to_csv(out/"cluster_top_terms.csv", index=False)

    print(f"Interpretation clustering: {len(set(labels)-{-1})} clusters; noise={(labels==-1).mean():.3f}")
    print("Saved derived interpretation and trajectory tables without exporting source text.")

if __name__ == "__main__":
    main()
