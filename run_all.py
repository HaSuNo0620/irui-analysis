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

NONHUMAN_TERMS = [
    "人外","異類","異種族","妖怪","あやかし","もののけ","鬼","妖狐","狐","狸","蛇","龍神","竜神","龍","竜",
    "神","神様","女神","精霊","妖精","天狗","雪女","吸血鬼","ヴァンパイア","悪魔","魔族","獣人","亜人",
    "エルフ","人魚","ラミア","ハーピー","フェンリル","アンデッド","ゾンビ","幽霊","死神","宇宙人","異星人",
    "アンドロイド","ロボット","人工生命","怪物","化け物","モンスター"
]
RELATION_TERMS = [
    "恋","恋愛","好き","愛する","愛され","惹かれ","想い","結婚","婚姻","嫁","嫁入り","花嫁","妻","夫","夫婦",
    "婚約","求婚","伴侶","番","つがい","恋人","恋仲","同居","暮らす","一緒に暮ら","新婚","溺愛","契約婚","生贄","生け贄"
]


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


def score_candidate(meta_text, head):
    nhm={w for w in NONHUMAN_TERMS if w in meta_text}; relm={w for w in RELATION_TERMS if w in meta_text}
    nhh={w for w in NONHUMAN_TERMS if w in head}; relh={w for w in RELATION_TERMS if w in head}
    nh=nhm|nhh; rel=relm|relh
    if not nh or not rel: return 0.0, nh, rel
    score=2*min(len(nhm),3)+2*min(len(relm),3)+0.75*min(len(nhh),4)+0.75*min(len(relh),4)
    return score, nh, rel


def chunk_text(text, size, overlap):
    step=size-overlap
    chunks=[]
    for st in range(0,len(text),step):
        en=min(st+size,len(text)); c=text[st:en]
        if len(c.strip())>=200: chunks.append((st,en,c))
        if en>=len(text): break
    return chunks


def stable_id(path, idx, meta):
    raw=f"{path}:{idx}:"+json.dumps(meta,ensure_ascii=False,sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL glob, e.g. 'data/*.jsonl'")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None)
    args=ap.parse_args()

    cfg=yaml.safe_load(open(args.config,encoding="utf-8"))
    paths=sorted(glob.glob(args.input))
    if not paths: raise SystemExit("No input files matched")
    out=Path("results"); out.mkdir(exist_ok=True)

    # Stage 1: high-recall retrieval only. Retrieval labels are NOT features downstream.
    candidates=[]; texts={}
    for path, idx, obj in iter_jsonl(paths):
        text=obj.get("text","") or ""; meta=obj.get("meta",{}) or {}
        meta_text=" ".join(flatten_strings(meta)); head=text[:cfg["head_chars"]]
        score, nh, rel=score_candidate(meta_text,head)
        if score < cfg["min_candidate_score"]: continue
        cid=stable_id(path,idx,meta)
        candidates.append({"candidate_id":cid,"source_file":path,"row_idx":idx,"score":score,
                           "retrieval_nonhuman_hits":"|".join(sorted(nh)),
                           "retrieval_relation_hits":"|".join(sorted(rel)),
                           "text_chars":len(text),"meta_json":json.dumps(meta,ensure_ascii=False)})
        texts[cid]=text
    cdf=pd.DataFrame(candidates)
    cdf.to_csv(out/"candidates.csv",index=False)
    if cdf.empty: raise SystemExit("No candidates. Lower min_candidate_score or inspect corpus schema.")

    # Stage 2: chunk + unlabeled embeddings
    model=SentenceTransformer(cfg["model"], device=args.device)
    manifest=[]; vecs=[]
    for row in candidates:
        cid=row["candidate_id"]; text=texts[cid]
        chunks=chunk_text(text,cfg["chunk_chars"],cfg["overlap_chars"])
        batch=["トピック: "+c for _,_,c in chunks]
        if not batch: continue
        emb=model.encode(batch,normalize_embeddings=True,batch_size=16,show_progress_bar=False)
        for j,((st,en,_),v) in enumerate(zip(chunks,emb)):
            vecs.append(v.astype(np.float32))
            manifest.append({"candidate_id":cid,"chunk_idx":j,"start_char":st,"end_char":en,
                             "position":((st+en)/2)/max(len(text),1)})
    X=np.stack(vecs)
    np.save(out/"embeddings.npy",X)
    mdf=pd.DataFrame(manifest); mdf.to_csv(out/"chunk_manifest.csv",index=False)

    # Stage 3: unsupervised structure
    ncomp=min(cfg["pca_components"], X.shape[1], max(2,X.shape[0]-1))
    Xp=PCA(n_components=ncomp, random_state=cfg["random_state"]).fit_transform(X)
    Xu=umap.UMAP(n_components=2,n_neighbors=cfg["umap_neighbors"],min_dist=cfg["umap_min_dist"],
                 metric="cosine",random_state=cfg["random_state"]).fit_transform(Xp)
    labels=hdbscan.HDBSCAN(min_cluster_size=cfg["hdbscan_min_cluster_size"]).fit_predict(Xp)

    res=mdf.copy(); res["umap_x"]=Xu[:,0]; res["umap_y"]=Xu[:,1]; res["cluster"]=labels
    res.to_csv(out/"clusters.csv",index=False)
    summary=res.groupby("cluster").size().rename("n_chunks").reset_index().sort_values("n_chunks",ascending=False)
    summary.to_csv(out/"cluster_summary.csv",index=False)

    plt.figure(figsize=(9,7))
    plt.scatter(res.umap_x,res.umap_y,s=5,alpha=.5,c=res.cluster)
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.title("Unsupervised semantic map of candidate text chunks")
    plt.tight_layout(); plt.savefig(out/"umap.png",dpi=180); plt.close()

    print(f"Candidates: {len(cdf)}")
    print(f"Chunks: {len(res)}")
    print(f"Clusters (excluding noise): {len(set(labels)-{-1})}")
    print("Saved to results/")

if __name__=="__main__": main()
