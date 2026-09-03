"""Validate the new (rob.ipynb-style) raw-engagement MAE metric on the KNN path.

Rebuilds the feature matrix exactly as notebooks/alden.ipynb cells 2-10 do and
runs the 5-fold grouped CV with raw (engagement, views) target, scoring MAE in
raw engagement units vs the median baseline. Mirrors cell 10.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse, parse_qs

repo = Path.cwd()
while not (repo / "data" / "processed" / "processed.csv").exists() and repo != repo.parent:
    repo = repo.parent

df = pd.read_csv(repo / "data" / "processed" / "processed.csv")
js = df["description_json"].apply(json.loads)
for i, p in js.items():
    if "play_by-play" in p and "play_by_play" not in p:
        p["play_by_play"] = p.pop("play_by-play")

LIST_FIELDS = ["content_theme", "format_access", "tone", "context",
               "overall_team", "audio_format", "event", "people", "brands"]
MIN_FREQ = 10
COLLAPSE = {"people", "brands"}

def toks(p, k):
    v = p.get(k, [])
    return v if isinstance(v, list) else []

val_counts = {k: {} for k in LIST_FIELDS}
for p in js:
    for k in LIST_FIELDS:
        for v in toks(p, k):
            val_counts[k][v] = val_counts[k].get(v, 0) + 1

# build the proper vocab + feature columns
feature_cols = []
for k in LIST_FIELDS:
    if k in COLLAPSE:
        cols = sorted(v for v, c in val_counts[k].items() if c >= MIN_FREQ) + ["_other"]
    else:
        cols = sorted(val_counts[k].keys())
    feature_cols.append((k, cols))
rec = {}
for idx, p in js.items():
    d = {}
    for k, cols in feature_cols:
        tt = set(toks(p, k))
        for c in cols:
            if c == "_other":
                keep = {v for v in tt if val_counts[k].get(v, 0) >= MIN_FREQ}
                d[f"{k}_{c}"] = int(bool(tt - keep))
            else:
                d[f"{k}_{c}"] = int(c in tt)
    rec[idx] = d
feat_df = pd.DataFrame.from_dict(rec, orient="index"); feat_df.index = df.index

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb_pbp = embedder.encode(js.apply(lambda p: p.get(PROSE_FIELD := "play_by_play", "") or "").tolist(), batch_size=256)
emb_cap = embedder.encode(df["content"].fillna("").astype(str).tolist(), batch_size=256)
emb_df = pd.DataFrame(np.hstack([emb_pbp, emb_cap]),
                      columns=[f"pbp_e{i}" for i in range(emb_pbp.shape[1])] +
                              [f"cap_e{i}" for i in range(emb_cap.shape[1])], index=df.index)

page_oh = pd.get_dummies(df["page"], prefix="page").astype(int)
X = pd.concat([feat_df, emb_df, page_oh, df[["duration_seconds"]]], axis=1).reset_index(drop=True)

def video_id(url):
    u = str(url); p = urlparse(u)
    if "youtube" in p.netloc or "youtu.be" in p.netloc:
        return (parse_qs(p.query).get("v") or [""])[0] or u.rstrip("/").split("/")[-1]
    segs = [s for s in u.split("?")[0].split("#")[0].rstrip("/").split("/") if s]
    return segs[-1]
df["video_id"] = df["url"].apply(video_id)
groups = df["video_id"].values
Y = df[["engagement", "views"]].to_numpy()

print("feature matrix:", X.shape, "| Y:", Y.shape, "| groups:", len(groups))

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

gkf = GroupKFold(n_splits=5)
X_arr = X.values
mae_raw_all, base_all = [], []
for fold, (tr, te) in enumerate(gkf.split(X_arr, groups=groups), 1):
    fsc = RobustScaler(); Xtr = fsc.fit_transform(X_arr[tr]); Xte = fsc.transform(X_arr[te])
    tsc = RobustScaler(); ytr = tsc.fit_transform(Y[tr]); yte = tsc.transform(Y[te])
    knn = KNeighborsRegressor(n_neighbors=20, weights="distance")
    knn.fit(Xtr, ytr)
    pred = tsc.inverse_transform(knn.predict(Xte))
    mae = mean_absolute_error(Y[te][:, 0], pred[:, 0])
    base = np.mean(np.abs(Y[te][:, 0] - np.median(Y[:, 0])))
    mae_raw_all.append(mae); base_all.append(base)
    print(f"fold {fold}: MAE(raw engagement)={mae:,.0f}  median-baseline={base:,.0f}")

mn, bs = np.mean(mae_raw_all), np.mean(base_all)
print("\n=== KNN raw-engagement MAE (5-fold grouped CV) ===")
print(f"MAE (raw engagement): mean={mn:,.0f}  std={np.std(mae_raw_all):,.0f}")
print(f"median-baseline MAE:  {bs:,.0f}")
print(f"Model beats baseline by {bs-mn:,.0f}  ({(bs-mn)/bs*100:.0f}% better)")
