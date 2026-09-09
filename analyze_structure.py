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

from event_windows import anonymize_proper_nouns
from run_all import novel_body

STOP={
    "こと","もの","ため","よう","ところ","そう","これ","それ","あれ","ここ","そこ","どこ",
    "私","俺","僕","彼","彼女","自分","人","一つ","二つ","今日","今","時","方","何",
    "する","いる","ある","なる","れる","られる","いう","言う","思う","見る","来る","行く","PROPER",
}


def read_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: rows.append({})
    return rows


def tokenize(text,tokenizer):
    out=[]
    for t in tokenizer.tokenize(text):
        pos=t.part_of_speech.split(",")[0]
        if pos not in {"名詞","動詞","形容詞"}: continue
        base=t.base_form if t.base_form!="*" else t.surface
        base=base.strip()
        if len(base)<2 or base in STOP: continue
        if re.fullmatch(r"[\W_\d]+",base): continue
        out.append(base)
    return out


def analyze_variant(name,X,mdf,cdf,source,cfg,out):
    ncomp=min(int(cfg["pca_components"]),X.shape[1],max(2,X.shape[0]-1))
    Xp=PCA(n_components=ncomp,random_state=int(cfg["random_state"])).fit_transform(X)
    labels=hdbscan.HDBSCAN(
        min_cluster_size=min(int(cfg["hdbscan_min_cluster_size"]),max(2,len(X)//5)),
        min_samples=cfg.get("hdbscan_min_samples",None),
    ).fit_predict(Xp)

    res=mdf.copy(); res["cluster_interpret"]=labels
    res.to_csv(out/f"clusters_interpret_{name}.csv",index=False)

    rows=[]
    for k,g in res.groupby("cluster_interpret"):
        rows.append({"cluster":int(k),"n_chunks":len(g),"n_works":g.candidate_id.nunique(),
                     "mean_position":g.position.mean(),"median_position":g.position.median()})
    pd.DataFrame(rows).sort_values("n_chunks",ascending=False).to_csv(out/f"cluster_interpret_summary_{name}.csv",index=False)

    reps=[]; nrep=int(cfg.get("representatives_per_cluster",5))
    for k in sorted(set(labels)):
        if k<0: continue
        idx=np.flatnonzero(labels==k); centroid=Xp[idx].mean(axis=0); d=np.linalg.norm(Xp[idx]-centroid,axis=1)
        for rank,jj in enumerate(idx[np.argsort(d)[:nrep]],1):
            r=res.iloc[jj]
            reps.append({"cluster":int(k),"rank":rank,"candidate_id":r.candidate_id,
                         "ncode":getattr(r,"ncode",""),"title":getattr(r,"title",""),
                         "chunk_idx":int(r.chunk_idx),"position":float(r.position),
                         "start_char":int(r.start_char),"end_char":int(r.end_char),
                         "distance_to_centroid":float(np.linalg.norm(Xp[jj]-centroid))})
    pd.DataFrame(reps).to_csv(out/f"cluster_representatives_{name}.csv",index=False)

    occ=res.groupby(["candidate_id","cluster_interpret"]).size().rename("n_chunks").reset_index()
    totals=res.groupby("candidate_id").size().rename("n_total").reset_index()
    occ=occ.merge(totals,on="candidate_id"); occ["occupancy"]=occ.n_chunks/occ.n_total
    occ.to_csv(out/f"work_cluster_occupancy_{name}.csv",index=False)

    nb=int(cfg.get("position_bins",10))
    tmp=res.copy(); tmp["position_bin"]=np.minimum((tmp.position.clip(0,0.999999)*nb).astype(int),nb-1)
    prof=tmp.groupby(["position_bin","cluster_interpret"]).size().rename("n_chunks").reset_index()
    btot=tmp.groupby("position_bin").size().rename("bin_total").reset_index()
    prof=prof.merge(btot,on="position_bin"); prof["p_cluster_given_position"]=prof.n_chunks/prof.bin_total
    prof.to_csv(out/f"position_cluster_profile_{name}.csv",index=False)

    trans=Counter(); state_tot=Counter()
    for _,g in res.sort_values(["candidate_id","position"]).groupby("candidate_id"):
        seq=g.cluster_interpret.astype(int).tolist()
        for a,b in zip(seq[:-1],seq[1:]): trans[(a,b)]+=1; state_tot[a]+=1
    trows=[{"from_cluster":a,"to_cluster":b,"count":n,"probability":n/state_tot[a] if state_tot[a] else 0.0}
           for (a,b),n in sorted(trans.items())]
    pd.DataFrame(trows).to_csv(out/f"cluster_transitions_{name}.csv",index=False)

    text_by_cid={}
    for _,r in cdf.iterrows():
        ri=int(r.row_idx)
        if 0<=ri<len(source):
            obj=source[ri]
            text_by_cid[r.candidate_id]=novel_body(obj.get("text","") or "",obj.get("meta",{}) or {})

    tok=Tokenizer(); cluster_counts=defaultdict(Counter); totals_terms=Counter()
    for _,r in res.iterrows():
        k=int(r.cluster_interpret)
        if k<0: continue
        text=text_by_cid.get(r.candidate_id,"")
        raw=text[int(r.start_char):int(r.end_char)]
        lexical=anonymize_proper_nouns(raw) if name=="anonymized" else raw
        terms=tokenize(lexical,tok); cluster_counts[k].update(terms); totals_terms.update(terms)

    V=max(len(totals_terms),1); grand=sum(totals_terms.values()); topn=int(cfg.get("top_terms_per_cluster",25)); termrows=[]
    for k,cnt in sorted(cluster_counts.items()):
        nk=sum(cnt.values()); bg=totals_terms-cnt; nbg=max(grand-nk,0); scored=[]
        for term,n in cnt.items():
            score=math.log((n+0.5)/(nk+0.5*V))-math.log((bg[term]+0.5)/(nbg+0.5*V))
            scored.append((score,term,n,bg[term]))
        for rank,(score,term,n,nbgterm) in enumerate(sorted(scored,reverse=True)[:topn],1):
            termrows.append({"cluster":k,"rank":rank,"term":term,"score":score,
                             "cluster_count":n,"other_count":nbgterm})
    pd.DataFrame(termrows).to_csv(out/f"cluster_top_terms_{name}.csv",index=False)
    print(f"{name}: {len(set(labels)-{-1})} clusters; noise={(labels==-1).mean():.3f}")
    return labels


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--config",default="config.yaml")
    ap.add_argument("--results",default="results")
    args=ap.parse_args()

    cfg=yaml.safe_load(open(args.config,encoding="utf-8")); out=Path(args.results)
    mdf=pd.read_csv(out/"chunk_manifest.csv"); cdf=pd.read_csv(out/"candidates.csv"); source=read_jsonl(args.input)
    Xraw=np.load(out/"embeddings_raw.npy"); Xanon=np.load(out/"embeddings_anonymized.npy")
    analyze_variant("raw",Xraw,mdf,cdf,source,cfg,out)
    analyze_variant("anonymized",Xanon,mdf,cdf,source,cfg,out)

    pd.read_csv(out/"clusters_interpret_raw.csv").to_csv(out/"clusters_interpret.csv",index=False)
    pd.read_csv(out/"cluster_interpret_summary_raw.csv").to_csv(out/"cluster_interpret_summary.csv",index=False)
    pd.read_csv(out/"cluster_representatives_raw.csv").to_csv(out/"cluster_representatives.csv",index=False)
    pd.read_csv(out/"work_cluster_occupancy_raw.csv").to_csv(out/"work_cluster_occupancy.csv",index=False)
    pd.read_csv(out/"position_cluster_profile_raw.csv").to_csv(out/"position_cluster_profile.csv",index=False)
    pd.read_csv(out/"cluster_transitions_raw.csv").to_csv(out/"cluster_transitions.csv",index=False)
    pd.read_csv(out/"cluster_top_terms_raw.csv").to_csv(out/"cluster_top_terms.csv",index=False)
    print("Saved work-level full-text raw and anonymized interpretation/trajectory tables.")

if __name__=="__main__": main()
