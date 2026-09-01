"""Build notebooks/alden.ipynb — KNN-first engagement-rank prototype.

Writes a valid Jupyter notebook whose executed path is the KNN model
(flatten description_json -> embed text -> build features -> 5-fold grouped CV
-> Spearman + MAE). LightGBM / RandomForest (and XGBoost, commented-out note)
cells are present but NOT executed so we can run them later after reviewing KNN.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "alden.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(src: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})


# -----------------------------------------------------------------------------
md("# Alden — engagement-rank prototype (KNN first)\n\n"
   "Goal: given a **text-only** description of a video idea (caption + auto-generated "
   "description), predict a single **0-100 'worth making' rank** — how well the idea "
   "will perform on engagement, platform-bias removed.\n\n"
   "This notebook builds the feature pipeline and runs **KNN** first so we can review "
   "a real number. LightGBM / RandomForest / XGBoost cells are in the notebook but "
   "left unexecuted until we look at KNN's results.\n\n"
   "Input: `data/processed/processed.csv` (12,105 short-video posts, 100% have `description_json`).")

# -----------------------------------------------------------------------------
md("## 1. Load & sanity-check the data\n\n"
   "10-input `data/processed/processed.csv` (12,105 → 11,306 rows after removing "
   "the `views <= 0` rows, which had no valid engagement rate). All remaining rows "
   "have `views > 0` and a `description_json`.")

code("""import pandas as pd
import numpy as np
from pathlib import Path

# Anchor to the repo root (nbconvert runs from notebooks/, so walk up).
repo = Path.cwd()
while not (repo / 'data' / 'processed' / 'processed.csv').exists() and repo != repo.parent:
    repo = repo.parent
DATA = repo / 'data' / 'processed' / 'processed.csv'
print('loading', DATA)

pd.set_option('display.width', 160)
df = pd.read_csv(DATA)
print('shape:', df.shape)
print(df[['platform','views','engagement']].head())
print('\\nplatform counts:')
print(df['platform'].value_counts())
print('\\nrows with views<=0 (should be 0 after cleaning):', int((df['views']<=0).sum()))
print('  of those, by platform:'); print(df.loc[df['views']<=0,'platform'].value_counts())""")

# -----------------------------------------------------------------------------
md("## 2. Flatten `description_json` → numeric features\n\n"
   "The JSON has 10 fields. 7 are small vocab lists → binary multi-hot; "
   "`people`/`brands` are huge (2,439 / 1,227) → keep values seen in ≥10 posts, "
   "collapse the rest to `_other`; `event` (126) multi-hots with a min-frequency cap. "
   "The 3-row `play_by-play` hyphen quirk is normalized to `play_by_play`.")

code("""import json

js = df['description_json'].apply(json.loads)

LIST_FIELDS = ['content_theme','format_access','tone','context','overall_team',
               'audio_format','event','people','brands']
PROSE_FIELD = 'play_by_play'

# Normalize the 3-row hyphen quirk
for i, p in js.items():
    if 'play_by-play' in p and 'play_by_play' not in p:
        p['play_by_play'] = p.pop('play_by-play')

def tokens(p, k):
    v = p.get(k, [])
    return v if isinstance(v, list) else []

# Build vocabulary for each list field
recode = []
VOCAB = {}
for k in LIST_FIELDS:
    vals = set()
    for p in js:
        vals.update(tokens(p, k))
    VOCAB[k] = sorted(vals)
    recode.append(f'{k}: {len(vals)} unique')
print('vocabulary sizes:'); [print(' ', r) for r in recode]

# Multi-hot feature names with min-frequency collapse for high-cardinality fields
MIN_FREQ = 10
COLLAPSE = {'people', 'brands'}   # very high cardinality -> top-values + '_other'
feature_cols = []
mark_records = []
val_counts = {k: {} for k in LIST_FIELDS}
for p in js:
    for k in LIST_FIELDS:
        for v in tokens(p, k):
            val_counts[k][v] = val_counts[k].get(v, 0) + 1

for k in LIST_FIELDS:
    if k in COLLAPSE:
        keep = {v for v, c in val_counts[k].items() if c >= MIN_FREQ}
        cols = sorted(keep) + ['_other']
    else:
        cols = list(VOCAB[k])
    feature_cols.append((k, cols))

from collections import defaultdict
rec = defaultdict(dict)
for idx, p in js.items():
    for k, cols in feature_cols:
        toks = set(tokens(p, k))
        for c in cols:
            if c == '_other':
                keep = {v for v in toks if v in val_counts[k] and val_counts[k][v] >= MIN_FREQ}
                notable = {v for v in toks if v not in keep}
                val = int(bool(notable))
            else:
                val = int(c in toks)
            rec[idx][f'{k}_{c}'] = val

feat_df = pd.DataFrame.from_dict(rec, orient='index')
feat_df.index = df.index
print('\\nmulti-hot feature matrix:', feat_df.shape)
print('sample cols:', list(feat_df.columns[:8]))""")

# -----------------------------------------------------------------------------
md("## 3. Embed the two prose fields\n\n"
   "`play_by_play` (verbal description of the video) and the caption `content` are "
   "each embedded into a 384-dim vector using `all-MiniLM-L6-v2` (same embedder as "
   "rob.ipynb). This captures *meaning*, which a keyword one-hot cannot.")

code("""from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
texts_playbyplay = js.apply(lambda p: p.get(PROSE_FIELD, '') or '')
texts_caption   = df['content'].fillna('').astype(str)

print(f'embedding {len(df)} play_by_play strings...')
emb_pbp  = embedder.encode(texts_playbyplay.tolist(), show_progress_bar=True, batch_size=256)
print(f'embedding {len(df)} caption strings...')
emb_cap  = embedder.encode(texts_caption.tolist(),   show_progress_bar=True, batch_size=256)
print('emb_pbp', emb_pbp.shape, ' emb_cap', emb_cap.shape)

EMB_COLS = [f'pbp_e{i}' for i in range(emb_pbp.shape[1])] + [f'cap_e{i}' for i in range(emb_cap.shape[1])]
emb_df = pd.DataFrame(np.hstack([emb_pbp, emb_cap]), columns=EMB_COLS, index=df.index)""")

# -----------------------------------------------------------------------------
md("## 4. Assemble features + compute the target\n\n"
   "**Target (rob.ipynb metric):** predict **RAW engagement AND views** together "
   "(2 outputs), **not** a normalized rank.\n\n"
   "**Features:** description multi-hots + 2×384 embeddings + `page` one-hot + `duration_seconds`. "
   "We deliberately do **not** feed `platform` as a feature, so the model can't learn "
   "'YouTube = high' and must learn content quality.")

code("""# --- features ---
page_oh = pd.get_dummies(df['page'], prefix='page').astype(int)
num = df[['duration_seconds']].copy()
X = pd.concat([feat_df, emb_df, page_oh, num], axis=1)
print('full feature matrix:', X.shape, '| #NaN:', int(X.isna().sum().sum()))

# --- video_id grouping key (leakage-safe CV) ---
from urllib.parse import urlparse, parse_qs
def video_id(url):
    u = str(url); p = urlparse(u)
    if 'youtube' in p.netloc or 'youtu.be' in p.netloc:
        return (parse_qs(p.query).get('v') or [''])[0] or u.rstrip('/').split('/')[-1]
    segs = [s for s in u.split('?')[0].split('#')[0].rstrip('/').split('/') if s]
    return segs[-1]
df['video_id'] = df['url'].apply(video_id)

# --- TARGET: RAW engagement AND views (rob.ipynb success metric) ---
Y = df[['engagement', 'views']].reset_index(drop=True).to_numpy()

X_tr = X.reset_index(drop=True)
Y_tr = Y
groups = df['video_id'].values
print('rows:', len(df), '| target shape:', Y.shape)
print(df[['engagement','views']].describe())""")

# -----------------------------------------------------------------------------
md("## 5. 5-fold grouped CV — KNN (raw-engagement MAE metric)\n\n"
   "`GroupedKFold` by `video_id` so a video never appears in both train and test. "
   "Per fold we fit a `RobustScaler` on the **train targets only** (no leakage), "
   "predict both (engagement, views), and score **MAE converted back to raw "
   "engagement units**, compared against a **median baseline** — exactly the "
   "rob.ipynb success metric.")

code("""from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

gkf = GroupKFold(n_splits=5)
X_arr = X_tr.values

mae_raw_all, baseline_raw_all = [], []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_arr, groups=groups), 1):
    fsc = RobustScaler()
    Xtr = fsc.fit_transform(X_arr[tr_idx]); Xte = fsc.transform(X_arr[te_idx])
    tsc = RobustScaler()
    ytr = tsc.fit_transform(Y_tr[tr_idx]); yte = tsc.transform(Y_tr[te_idx])
    knn = KNeighborsRegressor(n_neighbors=20, weights='distance')
    knn.fit(Xtr, ytr)
    pred = tsc.inverse_transform(knn.predict(Xte))
    yte_raw = Y_tr[te_idx]
    mae_raw = mean_absolute_error(yte_raw[:, 0], pred[:, 0])
    base = np.mean(np.abs(yte_raw[:, 0] - np.median(Y_tr[:, 0])))
    mae_raw_all.append(mae_raw); baseline_raw_all.append(base)
    print(f'fold {fold}: MAE(raw engagement)={mae_raw:,.0f}  median-baseline={base:,.0f}')

print('\\n=== KNN (5-fold grouped CV, target = RAW engagement, per rob.ipynb metric) ===')
mn = np.mean(mae_raw_all); bs = np.mean(baseline_raw_all)
print(f'MAE (raw engagement): mean={mn:,.0f}  std={np.std(mae_raw_all):,.0f}')
print(f'median-baseline MAE:  {bs:,.0f}')
print(f'Model beats baseline by {bs - mn:,.0f}  ({(bs-mn)/bs*100:.0f}% better)')""")

# -----------------------------------------------------------------------------
md("## 6. (Not run yet) LightGBM, RandomForest, XGBoost\n\n"
   "Success metric **mirrors rob.ipynb**: predict RAW engagement AND views, score by "
   "**MAE in raw engagement units**, vs a **median baseline**. `run_cv_raw` is defined "
   "in the first code cell below; then **run each model in its own cell**:\n"
   "- **LightGBM** (seconds)\n"
   "- **RandomForest** (100 trees, a few minutes)\n"
   "- **XGBoost**: skipped — needs `brew install libomp`, which the sandbox can't do.")

# shared helper
code("""from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
import numpy as np

def run_cv_raw(model, Xm, Ym, grp, name):
    \"\"\"5-fold grouped CV; predicts (engagement, views); reports raw-engagement MAE vs median baseline.\"\"\"
    gkf = GroupKFold(n_splits=5)
    mae_raw_all, base_all = [], []
    for tr_idx, te_idx in gkf.split(Xm, groups=grp):
        tsc = RobustScaler()
        ytr = tsc.fit_transform(Ym[tr_idx]); yte = tsc.transform(Ym[te_idx])
        model.fit(Xm[tr_idx], ytr)
        pred = tsc.inverse_transform(model.predict(Xm[te_idx]))
        yte_raw = Ym[te_idx]
        mae_raw_all.append(mean_absolute_error(yte_raw[:, 0], pred[:, 0]))
        base_all.append(np.mean(np.abs(yte_raw[:, 0] - np.median(Ym[:, 0]))))
    m, b = np.mean(mae_raw_all), np.mean(base_all)
    print(f'{name}: MAE(raw engagement)={m:,.0f} +/-{np.std(mae_raw_all):,.0f} | '
          f'median-baseline={b:,.0f} | beats baseline by {b-m:,.0f} ({(b-m)/b*100:.0f}%)')""")

# LightGBM (fast)
code("""# LightGBM — fast, seconds. Predicts (engagement, views); metric = raw-engagement MAE.
from lightgbm import LGBMRegressor
lgb = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
run_cv_raw(lgb, X_tr.values, Y_tr, groups, 'LightGBM')""")

# RandomForest (slower)
code("""# RandomForest — 100 trees + n_jobs=-1; a few minutes. Same raw-target metric.
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
run_cv_raw(rf, X_tr.values, Y_tr, groups, 'RandomForest')

# XGBoost — uncomment once 'brew install libomp' has been run on this machine
# import xgboost as xgb
# xgbm = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
# run_cv_raw(xgbm, X_tr.values, Y_tr, groups, 'XGBoost')""")

# -----------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (venv)", "language": "python", "name": "venv"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "cells": cells,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH} with {len(cells)} cells")
