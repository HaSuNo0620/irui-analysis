#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import umap
import hdbscan
import matplotlib.pyplot as plt

from event_windows import lexical_hits, extract_relation_event_windows, anonymize_proper_nouns


def flatten_strings(obj):
    out=[]
    if isinstance(obj, dict):
        for v in obj.values(): out += flatten_strings(v)
    elif isinstance(obj, list):
        for v in obj: out += flatten_strings(v)
    elif isinstance(obj, (str,int,float,bool)):
        out.append(str(obj))
    return out


def iter_jsonl(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try: obj=json.loads(line)
                except Exception: continue
                yield p, idx, obj


def stable_id(path, idx, meta):
    raw=f"{path}:{idx}:"+json.dumps(meta,ensure_ascii=False,sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def pick_meta(meta, needles):
    if not isinstance(meta, dict):
        return ""
    flat=[]
    def walk(obj, key=""):
        if isinstance(obj, dict):
            for k,v in obj.items(): walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(obj, (str,int,float,bool)):
            flat.append((key.lower(), str(obj)))
    walk(meta)
    for needle in needles:
        for k,v in flat:
            if needle in k:
                return v
    return ""


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Preselected candidate JSONL glob")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None)
    args=ap.parse_args()

    cfg=yaml.safe_load(open(args.config,encoding="utf-8"))
    paths=sorted(glob.glob(args.input))
    if not paths: raise SystemExit("No input files matched")
    out=Path("results"); out.mkdir(exist_ok=True)

    candidates=[]; texts={}
    for path, idx, obj in iter_jsonl(paths):
        text=obj.get("text","") or ""; meta=obj.get("meta",{}) or {}
        if not text.strip():
            continue
        meta_text=" ".join(flatten_strings(meta)); head=text[:cfg["head_chars"]]
        nh, rel=lexical_hits(meta_text + "\n" + head)
        cid=stable_id(path,idx,meta)
        candidates.append({
            "candidate_id":cid,"source_file":path,"row_idx":idx,
            "title":pick_meta(meta,["title","name"]),
            "date_raw":pick_meta(meta,["firstup","publish","date","created"]),
            "retrieval_nonhuman_hits":"|".join(sorted(nh)),
            "retrieval_relation_hits":"|".join(sorted(rel)),
            "text_chars":len(text),"meta_json":json.dumps(meta,ensure_ascii=False)
        })
        texts[cid]=text
    cdf=pd.DataFrame(candidates)
    cdf.to_csv(out/"candidates.csv",index=False)
    if cdf.empty: raise SystemExit("No candidate rows were present in the preselected input.")

    model=SentenceTransformer(cfg["model"], device=args.device)
    manifest=[]; vecs=[]
    proximity=int(cfg.get("relation_event_proximity",500))
    window_chars=int(cfg.get("relation_event_window_chars",1200))
    max_windows=int(cfg.get("max_event_windows_per_work",25))
    works_with_events=0

    for row in candidates:
        cid=row["candidate_id"]; text=texts[cid]
        windows=extract_relation_event_windows(
            text, proximity=proximity, window_chars=window_chars, max_windows=max_windows
        )
        if not windows:
            continue
        works_with_events += 1
        normalized=[anonymize_proper_nouns(raw) for _,_,raw in windows]
        batch=["トピック: "+c for c in normalized]
        emb=model.encode(batch,normalize_embeddings=True,batch_size=16,show_progress_bar=False)
        for j,((st,en,_),v) in enumerate(zip(windows,emb)):
            vecs.append(v.astype(np.float32))
            manifest.append({
                "candidate_id":cid,"title":row.get("title",""),"date_raw":row.get("date_raw",""),
                "chunk_idx":j,"event_idx":j,"start_char":st,"end_char":en,
                "position":((st+en)/2)/max(len(text),1),
                "event_chars":en-st,
            })
    if not vecs:
        raise SystemExit("No relation-event windows were generated from selected candidates.")

    X=np.stack(vecs)
    np.save(out/"embeddings.npy",X)
    mdf=pd.DataFrame(manifest); mdf.to_csv(out/"chunk_manifest.csv",index=False)

    ncomp=min(cfg["pca_components"], X.shape[1], max(2,X.shape[0]-1))
    Xp=PCA(n_components=ncomp, random_state=cfg["random_state"]).fit_transform(X)
    Xu=umap.UMAP(
        n_components=2,
        n_neighbors=min(cfg["umap_neighbors"], max(2, len(X)-1)),
        min_dist=cfg["umap_min_dist"],metric="cosine",
        random_state=cfg["random_state"]
    ).fit_transform(Xp)
    labels=hdbscan.HDBSCAN(
        min_cluster_size=min(cfg["hdbscan_min_cluster_size"], max(2, len(X)//5)),
        min_samples=cfg.get("hdbscan_min_samples", None)
    ).fit_predict(Xp)

    res=mdf.copy(); res["umap_x"]=Xu[:,0]; res["umap_y"]=Xu[:,1]; res["cluster"]=labels
    res.to_csv(out/"clusters.csv",index=False)
    summary=res.groupby("cluster").size().rename("n_chunks").reset_index().sort_values("n_chunks",ascending=False)
    summary.to_csv(out/"cluster_summary.csv",index=False)

    plt.figure(figsize=(9,7))
    plt.scatter(res.umap_x,res.umap_y,s=6,alpha=.55,c=res.cluster)
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.title("Anonymized relation-event semantic map")
    plt.tight_layout(); plt.savefig(out/"umap.png",dpi=180); plt.close()

    print(f"Candidates: {len(cdf)}")
    print(f"Works with relation-event windows: {works_with_events}")
    print(f"Relation-event windows: {len(res)}")
    print(f"Clusters (excluding noise): {len(set(labels)-{-1})}")
    print("Saved to results/")

if __name__=="__main__": main()
