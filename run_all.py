#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, json, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap
import hdbscan
import matplotlib.pyplot as plt

from event_windows import lexical_hits, anonymize_proper_nouns


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
    if not isinstance(meta, dict): return ""
    flat=[]
    def walk(obj, key=""):
        if isinstance(obj, dict):
            for k,v in obj.items(): walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(obj, (str,int,float,bool)):
            flat.append((key.lower(), str(obj)))
    walk(meta)
    for needle in needles:
        for k,v in flat:
            if needle in k: return v
    return ""


def full_text_chunks(text, chunk_chars=1800, overlap_chars=300, max_chunks=25):
    """Cover the whole work with fixed windows; cap long works by uniform position sampling."""
    text=text or ""
    if not text.strip(): return []
    chunk_chars=max(int(chunk_chars),200)
    overlap_chars=max(0,min(int(overlap_chars),chunk_chars-1))
    step=max(1,chunk_chars-overlap_chars)
    starts=list(range(0,max(len(text)-chunk_chars+1,1),step))
    last=max(0,len(text)-chunk_chars)
    if not starts or starts[-1] != last: starts.append(last)
    starts=sorted(set(starts))
    chunks=[(st,min(len(text),st+chunk_chars),text[st:min(len(text),st+chunk_chars)]) for st in starts]
    if max_chunks and len(chunks)>max_chunks:
        if max_chunks==1:
            idx=[len(chunks)//2]
        else:
            idx=[round(i*(len(chunks)-1)/(max_chunks-1)) for i in range(max_chunks)]
        chunks=[chunks[i] for i in sorted(set(idx))]
    return chunks


def cluster_embedding(X, cfg, seed_offset=0):
    ncomp=min(int(cfg["pca_components"]),X.shape[1],max(2,X.shape[0]-1))
    seed=int(cfg["random_state"])+seed_offset
    Xp=PCA(n_components=ncomp,random_state=seed).fit_transform(X)
    Xu=umap.UMAP(
        n_components=2,
        n_neighbors=min(int(cfg["umap_neighbors"]),max(2,len(X)-1)),
        min_dist=float(cfg["umap_min_dist"]),metric="cosine",random_state=seed
    ).fit_transform(Xp)
    labels=hdbscan.HDBSCAN(
        min_cluster_size=min(int(cfg["hdbscan_min_cluster_size"]),max(2,len(X)//5)),
        min_samples=cfg.get("hdbscan_min_samples",None)
    ).fit_predict(Xp)
    return Xp,Xu,labels


def save_variant(out, name, manifest, X, Xu, labels):
    np.save(out/f"embeddings_{name}.npy",X)
    res=manifest.copy()
    res["umap_x"]=Xu[:,0]; res["umap_y"]=Xu[:,1]; res["cluster"]=labels
    res.to_csv(out/f"clusters_{name}.csv",index=False)
    (res.groupby("cluster").size().rename("n_chunks").reset_index()
       .sort_values("n_chunks",ascending=False).to_csv(out/f"cluster_summary_{name}.csv",index=False))
    plt.figure(figsize=(9,7))
    plt.scatter(res.umap_x,res.umap_y,s=6,alpha=.55,c=res.cluster)
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.title(f"Full-text semantic map ({name})")
    plt.tight_layout(); plt.savefig(out/f"umap_{name}.png",dpi=180); plt.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,help="Strictly preselected candidate JSONL glob")
    ap.add_argument("--config",default="config.yaml")
    ap.add_argument("--device",default=None)
    args=ap.parse_args()

    cfg=yaml.safe_load(open(args.config,encoding="utf-8"))
    paths=sorted(glob.glob(args.input))
    if not paths: raise SystemExit("No input files matched")
    out=Path("results"); out.mkdir(exist_ok=True)

    candidates=[]; texts={}
    for path,idx,obj in iter_jsonl(paths):
        text=obj.get("text","") or ""; meta=obj.get("meta",{}) or {}
        if not text.strip(): continue
        meta_text=" ".join(flatten_strings(meta)); head=text[:int(cfg["head_chars"])]
        nh,rel=lexical_hits(meta_text+"\n"+head)
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
    cdf=pd.DataFrame(candidates); cdf.to_csv(out/"candidates.csv",index=False)
    if cdf.empty: raise SystemExit("No candidate rows were present in the preselected input.")

    chunk_chars=int(cfg.get("chunk_chars",1800))
    overlap=int(cfg.get("overlap_chars",300))
    max_chunks=int(cfg.get("max_chunks_per_work",25))
    manifest=[]; raw_chunks=[]; anon_chunks=[]
    for row in candidates:
        cid=row["candidate_id"]; text=texts[cid]
        chunks=full_text_chunks(text,chunk_chars,overlap,max_chunks)
        for j,(st,en,raw) in enumerate(chunks):
            manifest.append({
                "candidate_id":cid,"title":row.get("title",""),"date_raw":row.get("date_raw",""),
                "chunk_idx":j,"start_char":st,"end_char":en,
                "position":((st+en)/2)/max(len(text),1),"chunk_chars":en-st,
            })
            raw_chunks.append(raw)
            anon_chunks.append(anonymize_proper_nouns(raw))
    if not raw_chunks: raise SystemExit("No full-text chunks were generated.")
    mdf=pd.DataFrame(manifest); mdf.to_csv(out/"chunk_manifest.csv",index=False)

    model=SentenceTransformer(cfg["model"],device=args.device)
    Xraw=model.encode(["トピック: "+x for x in raw_chunks],normalize_embeddings=True,batch_size=16,show_progress_bar=True).astype(np.float32)
    Xanon=model.encode(["トピック: "+x for x in anon_chunks],normalize_embeddings=True,batch_size=16,show_progress_bar=True).astype(np.float32)

    Xp_raw,Xu_raw,lab_raw=cluster_embedding(Xraw,cfg,0)
    Xp_anon,Xu_anon,lab_anon=cluster_embedding(Xanon,cfg,0)
    save_variant(out,"raw",mdf,Xraw,Xu_raw,lab_raw)
    save_variant(out,"anonymized",mdf,Xanon,Xu_anon,lab_anon)

    # Legacy files point to raw full-text analysis so existing diagnostics keep working.
    np.save(out/"embeddings.npy",Xraw)
    raw_res=mdf.copy(); raw_res["umap_x"]=Xu_raw[:,0]; raw_res["umap_y"]=Xu_raw[:,1]; raw_res["cluster"]=lab_raw
    raw_res.to_csv(out/"clusters.csv",index=False)
    raw_res.groupby("cluster").size().rename("n_chunks").reset_index().sort_values("n_chunks",ascending=False).to_csv(out/"cluster_summary.csv",index=False)

    comp=pd.DataFrame({
        "metric":["adjusted_rand_index","normalized_mutual_info"],
        "value":[adjusted_rand_score(lab_raw,lab_anon),normalized_mutual_info_score(lab_raw,lab_anon)]
    })
    comp.to_csv(out/"raw_vs_anonymized_cluster_agreement.csv",index=False)
    assign=mdf[["candidate_id","chunk_idx","position"]].copy()
    assign["cluster_raw"]=lab_raw; assign["cluster_anonymized"]=lab_anon
    assign.to_csv(out/"raw_vs_anonymized_assignments.csv",index=False)

    per_work=mdf.groupby("candidate_id").size()
    print(f"Candidates: {len(cdf)}")
    print(f"Full-text chunks: {len(mdf)}")
    print(f"Mean chunks per work: {per_work.mean():.2f}")
    print(f"Median chunks per work: {per_work.median():.1f}")
    print(f"Works at chunk cap ({max_chunks}): {(per_work>=max_chunks).sum()}")
    print(f"Raw clusters (excluding noise): {len(set(lab_raw)-{-1})}; noise={(lab_raw==-1).mean():.3f}")
    print(f"Anonymized clusters (excluding noise): {len(set(lab_anon)-{-1})}; noise={(lab_anon==-1).mean():.3f}")
    print(f"Raw/anonymized ARI: {adjusted_rand_score(lab_raw,lab_anon):.3f}")
    print(f"Raw/anonymized NMI: {normalized_mutual_info_score(lab_raw,lab_anon):.3f}")
    print("Selection evidence is used only for corpus membership; downstream analysis uses full text.")

if __name__=="__main__": main()
