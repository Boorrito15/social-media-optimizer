"""Plot KNN cross-validation results from notebooks/alden.ipynb.

Reads the executed KNN metrics and the underlying data to draw:
  1. Per-fold Spearman + MAE (+RMSE) grouped bar chart.
  2. A scatter of predicted vs actual engagement-rank, aggregated from the
     notebook's 5-fold CV, with a per-platform color to show model-agnostic behavior.
"""
from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Pull the KNN metrics straight from the executed notebook outputs.
# ---------------------------------------------------------------------------
nb = json.loads(Path("notebooks/alden.ipynb").read_text())
def find_text(pred: str) -> list[str]:
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if "text" in o and pred in "".join(o["text"]):
                out.append("".join(o["text"]))
    return out

folds_text = find_text("fold 1:")[0]
fold_rows = []
for line in folds_text.strip().splitlines():
    line = line.strip()
    if not line.startswith("fold"):
        continue
    parts = line.split()
    def kv(pref):
        for p in parts:
            if p.startswith(pref):
                return float(p.split("=")[1])
    fold_rows.append({
        "fold": int(parts[1].rstrip(":")),
        "spearman": kv("Spearman="),
        "mae": kv("MAE="),
        "rmse": kv("RMSE="),
    })
fold = pd.DataFrame(fold_rows)
import re
mean = find_text("=== KNN (5-fold")[-1].strip()
mean_rows = {}
for m in re.finditer(r"(Spearman|MAE|RMSE)\s*:\s*mean=([0-9.]+)\s+std=([0-9.]+)", mean):
    mean_rows[m.group(1).lower()] = {"mean": float(m.group(2)), "std": float(m.group(3))}
means = pd.DataFrame(mean_rows).T

# ---------------------------------------------------------------------------
# 2. Rebuild the per-fold predictions for the scatter (mirror of notebook logic).
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/processed.csv")

LIST_FIELDS = ["content_theme", "format_access", "tone", "context",
               "overall_team", "audio_format", "event", "people", "brands"]
MIN_FREQ = 10
COLLAPSE = {"people", "brands"}
js = df["description_json"].apply(json.loads)
for i, p in js.items():
    if "play_by-play" in p and "play_by_play" not in p:
        p["play_by_play"] = p.pop("play_by-play")
def toks(p, k):
    v = p.get(k, [])
    return v if isinstance(v, list) else []
val_counts = {k: {} for k in LIST_FIELDS}
for p in js:
    for k in LIST_FIELDS:
        for v in toks(p, k):
            val_counts[k][v] = val_counts[k].get(v, 0) + 1
import collections
rec = collections.defaultdict(dict)
feature_cols = []
for k in LIST_FIELDS:
    if k in COLLAPSE:
        cols = sorted(v for v, c in val_counts[k].items() if c >= MIN_FREQ) + ["_other"]
    else:
        cols = sorted(val_counts[k].keys())
    feature_cols.append((k, cols))
for idx, p in js.items():
    for k, cols in feature_cols:
        tt = set(toks(p, k))
        for c in cols:
            if c == "_other":
                keep = {v for v in tt if val_counts[k].get(v, 0) >= MIN_FREQ}
                val = int(bool(tt - keep))
            else:
                val = int(c in tt)
            rec[idx][f"{k}_{c}"] = val
feat_df = pd.DataFrame.from_dict(rec, orient="index"); feat_df.index = df.index

page_oh = pd.get_dummies(df["page"], prefix="page").astype(int)
X = pd.concat([feat_df, page_oh, df[["duration_seconds"]]], axis=1)

from urllib.parse import urlparse, parse_qs
def video_id(url):
    u = str(url); p = urlparse(u)
    if "youtube" in p.netloc or "youtu.be" in p.netloc:
        return (parse_qs(p.query).get("v") or [""])[0] or u.rstrip("/").split("/")[-1]
    segs = [s for s in u.split("?")[0].split("#")[0].rstrip("/").split("/") if s]
    return segs[-1]
df["video_id"] = df["url"].apply(video_id)
df["engagement_rate"] = df["engagement"] / df["views"]
g = df.groupby("platform")["engagement_rate"].transform(lambda s: (s - s.mean()) / s.std())
y = (g.rank(pct=True) * 100).values
groups = df["video_id"].values

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
scaler = StandardScaler(); Xs = scaler.fit_transform(X.values)
gkf = GroupKFold(n_splits=5)
acts, preds, plats = [], [], []
for tr, te in gkf.split(Xs, groups=groups):
    knn = KNeighborsRegressor(n_neighbors=20, weights="distance")
    knn.fit(Xs[tr], y[tr])
    preds += list(knn.predict(Xs[te]))
    acts  += list(y[te])
    plats += list(df["platform"].iloc[te])

out = Path("data/plots"); out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1 — per-fold metrics (Spearman, MAE, RMSE)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
plots = [("spearman", "Spearman correlation", "higher is better", "Greens", True),
         ("mae", "MAE (rank units)", "lower is better", "Reds", False),
         ("rmse", "RMSE (rank units)", "lower is better", "Oranges", False)]
for ax, (col, title, note, cmap, rev) in zip(axes, plots):
    vals = fold[col].values
    bars = ax.bar(range(1, 6), vals, color=plt.get_cmap(cmap)(np.linspace(0.35, 0.85, 5)))
    ax.axhline(means.loc[col, "mean"], color="black", ls="--", lw=1)
    ax.text(5.15, means.loc[col, "mean"], f"mean={means.loc[col,'mean']:.3f}\nstd±{means.loc[col,'std']:.3f}",
            va="center", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.plot(range(7), [means.loc[col,'mean']]*7, "--", color="gray", lw=0.6)
    ax.set_title(f"{title}\n{note}")
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([f"fold {i}" for i in range(1, 6)])
    ax.set_ylim(min(vals)*0.9, max(vals)*1.08)
fig.suptitle("KNN (k=20, distance-weighted) — 5-fold grouped CV, target = engagement-rate rank (0-100)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(out / "knn_cv_metrics.png", dpi=130)
print("saved", out / "knn_cv_metrics.png")

# ---------------------------------------------------------------------------
# Figure 2 — predicted vs actual engagement rank (per-platform color)
# ---------------------------------------------------------------------------
acts = np.array(acts); preds = np.array(preds); plats = np.array(plats)
from scipy.stats import spearmanr, pearsonr
rho, _ = spearmanr(preds, acts)
pear, _ = pearsonr(preds, acts)
fig, ax = plt.subplots(figsize=(7, 6.5))
colors = {"IG": "#E1306C", "FB": "#1877F2", "TT": "#010101", "YT": "#FF0000"}
for p in ["IG", "FB", "TT", "YT"]:
    m = plats == p
    ax.scatter(preds[m], acts[m], s=8, alpha=0.25, color=colors[p], label=f"{p} (n={int(m.sum())})")
ax.plot([0, 100], [0, 100], ls="--", color="gray", lw=1, label="perfect")
ax.set_xlabel("predicted rank (0-100)")
ax.set_ylabel("actual rank (0-100)")
ax.set_title(f"KNN predicted vs actual engagement rank\nSpearman={rho:.3f}  Pearson={pear:.3f}")
ax.legend(markerscale=3, fontsize=9)
fig.tight_layout()
fig.savefig(out / "knn_pred_vs_actual.png", dpi=130)
print("saved", out / "knn_pred_vs_actual.png")

# ---------------------------------------------------------------------------
# Print the final table
# ---------------------------------------------------------------------------
print("\nKNN 5-fold grouped CV results:")
pd.set_option("display.width", 120)
print(fold.round(4).to_string(index=False))
print("\nmeans:")
print(means[["mean", "std"]].round(3))
