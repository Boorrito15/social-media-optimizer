# Social Media Optimizer

All Blacks short-form content optimisation. Ingests the master social-media
funnel dataset, cleans and de-dupes it, extracts video content across all platforms,
re-encodes to 480p H.264, and streams deliverables directly to Google Cloud Storage (GCS).

---

## ⚡️ Quickstart & Setup

```bash
# 1) Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure credentials & environment
cp .env.example .env
# - Place your GCP service account JSON key inside config/ (e.g. config/service-account.json)
# - Set GOOGLE_APPLICATION_CREDENTIALS in .env
```

---

## ☁️ GCP Architecture & Storage

All outputs are uploaded to regional standard buckets in `asia-southeast2`:

| Bucket | Purpose | Object Path |
| --- | --- | --- |
| `sm-optimizer-raw` | Raw source datasets | `gs://sm-optimizer-raw/` |
| `sm-optimizer-processed` | Cleaned datasets & 480p videos | `gs://sm-optimizer-processed/videos/<platform>/<id>.mp4` |
| `sm-optimizer-processed` | Parquet manifests & index shards | `gs://sm-optimizer-processed/manifests/` |

---

## 📂 Code Layout

- **[`Dockerfile.scraper`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/Dockerfile.scraper)**: Lightweight Dockerfile for Cloud Run Jobs scraper.
- **[`requirements-scraper.txt`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/requirements-scraper.txt)**: Minimal scraper runtime dependencies.
- **[`src/video/main.py`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/src/video/main.py)**: Scraper entrypoint with ADC fallback and task sharding.
- **[`src/video/upload.py`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/src/video/upload.py)**: Resolution, download, transcode, GCS upload, and disk cleanup.
- **[`src/video/index.py`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/src/video/index.py)**: Index shard emission to GCS.
- **[`src/video/web_dashboard.py`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/src/video/web_dashboard.py)**: MinionsScout live web dashboard server.
- **[`src/ml/`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/src/ml)`**: ML feature pipeline (`features.py`), training (`train.py`), serving (`predict.py`), and metadata auto-inference (`infer.py`).
- **[`src/api/app.py`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/src/api/app.py)**: FastAPI prediction service (see "Predictor UI & API" below).
- **[`streamlit_app.py`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/streamlit_app.py)**: Streamlit dashboard — a professional social-media-manager "Studio" with six sections (see *Dashboard sections* below).
- **[`run.sh`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/run.sh)**: One-command launcher for the API + UI.

---

## 🧠 Predictor UI & API

A Streamlit dashboard + FastAPI service with a clean, Gemini-like landing page.
The user types one free-text **description**; every other piece of metadata
(title, platform, brand/page, themes, format, tone, duration) is
**auto-inferred** from that text. The app returns a **make / skip
verdict**, a **high/low** views & engagement prediction, a **demo revenue**
projection, and similar historical posts.

> ⚠️ **Money is a demo.** `cost_nzd` is empty for every row (0 / 11,306) in the
> processed data, so `revenue = views × RPM / 1000` is a rule-of-thumb
> placeholder, clearly labelled in the UI — not a trained model.

### Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt -r requirements-api.txt   # adds fastapi/uvicorn & co.

# (optional) retrain models from data/processed/processed.csv
./run.sh train

# run it all (API on :8000, Streamlit on :8501)
./run.sh all     # open http://127.0.0.1:8501
```

### Dashboard sections

The Streamlit UI is organised as a sidebar-navigated dashboard with six sections:

| Section | What it does |
| --- | --- |
| 🎬 **Idea Studio** | Type a description (or pick a quick-example chip), get a make/skip verdict, Go-score gauge, typical views / engagement / demo revenue KPIs, probability bars, auto-inferred metadata, and similar historical posts. |
| ⚖️ **Compare** | Score 2–4 ideas side-by-side. Side-by-side bar chart, best-idea highlight, full verdict + KPIs per scenario. |
| 🕘 **History** | In-session log of every analysis (timestamp, description, verdict, Go score). Re-open any past idea with one click, or export the whole history as JSON. |
| 🔬 **Advanced Insights** | Aggregated findings from `data/processed/processed.csv`: top content themes, formats, tones, what themes drive views, platform leaders (median views vs. engagement), year-over-year trend. |
| 📚 **Explore** | Representative posts from the model index — scatter + box plots, full sample table. |
| ℹ️ **About** | What the models do, the verdict maths, honest limits, and the architecture map. |

The sidebar also carries a **⚙️ Manual override** panel: toggle it on to manually
pick platform, page, duration, themes, formats, tones and money assumptions
before scoring — instead of trusting auto-inference blindly. When the toggle
is off (default), the app sends just the description and lets the model
auto-fill the rest.

Or run the two parts separately:

```bash
./run.sh api     # FastAPI docs at http://127.0.0.1:8000/docs
./run.sh app     # Streamlit UI at http://127.0.0.1:8501
```

### Inputs & auto-inference

The UI ships a single text box. Because the prediction API accepts a bare
description and auto-fills the rest, a "huge try in the final minute" vs "an
emotional retirement tribute" produce different platform/page, themes, format,
tone and duration automatically.

### What the models do

The runtime trains **reproducible XGBoost** models on the same intent as
`notebooks/rob.ipynb` (TensorFlow isn't part of this environment), on a
**compact** feature set that the app can actually reconstruct from a free-text
description (platform, page, content-theme / format / tone multi-hots,
duration, and hashtag/mention/emoji counts). Training and serving share the
same feature space — this matters: a user's plain sentence doesn't populate the
dataset's rich token columns (specific players, campaigns, categories), so a
model trained on those would collapse every sparse input to a single answer.
The compact model keeps that from happening:

| Target | Model | Hold-out accuracy | Majority baseline |
| --- | --- | --- | --- |
| High/low **views** | XGBoost classifier | ~0.81 | 0.50 |
| High/low **engagement** | XGBoost classifier | ~0.84 | 0.50 |

The displayed "typical views / engagement" are the **continuers regression
estimate, anchored to the historical median** of the predicted bucket — so they
respond to the input while staying within a realistic range of the historical
data. Similar posts come from a `sentence-transformers` semantic index of
historical captions.

> ⚠️ **Honest limitation:** discrimination mainly comes from a handful of
> high-signal ideas ("try", "highlight", "celebration"). Two moderately-low
> ideas can still land on the same borderline score, because NZ-rugby short
> form mostly performs well. The verdict and estimates vary with input, but
> don't expect fine-grained ranking of two similar weak ideas.

### API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness + model load state |
| `POST` | `/infer` | Turn a free-text description into full metadata |
| `POST` | `/predict` | Verdict (auto-infers metadata from a bare description) |
| `GET` | `/explore/peers` | Representative posts for the Explore tab |
| `GET` | `/schema` | Canonical platform / page options |

Example (description-only — everything else is inferred):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"description":"A huge try in the final minute breaks the deadlock as the crowd erupts"}'
```

### Regenerating artifacts

Models are written to `data/models/` (git-ignored). Re-run
`./run.sh train` after updating `data/processed/processed.csv` to refresh
`bundle.joblib` (pipeline + models + thresholds) and `similar.joblib`
(embeddings + peer rows).

---

## 🚀 Video Pipeline & Automated Multi-Platform Scraping

The pipeline automatically handles short-form video extraction, format normalization, 480p H.264/AAC transcoding via ffmpeg, and GCS ingestion across **all 4 platforms**:

| Platform | Extraction Strategy | Recommended Concurrency | Status |
| --- | --- | --- | --- |
| **YouTube Shorts** | Direct `yt-dlp` stream extraction | `--concurrency 1` (Serial to avoid rate-limits) | ✅ Automated |
| **TikTok** | Direct `yt-dlp` stream extraction | `--concurrency 3` | ✅ Automated |
| **Instagram Reels** | Enhanced extractor + sanitized post IDs | `--concurrency 5` | ✅ Automated |
| **Facebook Reels** | Direct stream extraction + fallback handlers | `--concurrency 5` | ✅ Automated |

### Running the Scrapers

```bash
# Run Instagram & Facebook scraping concurrently (5 threads each)
python -m src.video.main --platforms instagram --concurrency 5
python -m src.video.main --platforms facebook --concurrency 5

# Authenticated scraping with browser cookies (e.g. chrome/safari/firefox)
python -m src.video.main --cookies-from-browser chrome

# Scrape all 4 platforms in a single command
python -m src.video.main --platforms youtube,tiktok,instagram,facebook

# Dry-run test (5 items, no downloads or GCS uploads)
python -m src.video.main --limit 5 --dry-run
```

### CLI Flags

| Flag | Description |
| ---- | ----------- |
| `--platforms <a,b>` | Comma-separated or space-separated platforms to process (youtube, tiktok, instagram, facebook) |
| `--limit N` | Process at most `N` posts (dry-run/scratch work) |
| `--dry-run` | Trace the pipeline without downloading/transcoding/uploading |
| `--no-transcode` | Upload the source (original resolution) instead of the 480p |
| `--concurrency N` | Number of parallel workers (default `VIDEO_CONCURRENCY`) |
| `--task-index N` | Task index for Cloud Run job sharding (default `CLOUD_RUN_TASK_INDEX` or 0) |
| `--task-count N` | Total task count for Cloud Run job sharding (default `CLOUD_RUN_TASK_COUNT` or 1) |
| `--run-id <id>` | Stable id used to name index shards + logs |
| `--log <file>` | Also upload the run's stdout to `logs/<run_id>.log` |
| `--consolidate` | Rebuild the cumulative `videos_index.parquet` afterwards |
| `--cookies <file>` | Path to a Netscape `cookies.txt` for authenticated scraping |
| `--cookies-from-browser <browser>` | Load cookies from an installed browser (chrome, safari, ...) |

---

## ☁️ Serverless Batch Ingestion with Google Cloud Run Jobs

For high-throughput, parallel scraping without keeping local machines running, the pipeline supports deployment as a serverless **Cloud Run Job**:

### Architecture & Sharding
- **Deterministic Modulo Sharding**: When deployed with `N` tasks (`--task-count N`), each task processes rows where `index % CLOUD_RUN_TASK_COUNT == CLOUD_RUN_TASK_INDEX` (specifically filtering `df[df.index % CLOUD_RUN_TASK_COUNT == CLOUD_RUN_TASK_INDEX]`). Task indices are determined by the standard Cloud Run environment variables `CLOUD_RUN_TASK_INDEX` (0..N-1) and `CLOUD_RUN_TASK_COUNT` (N).
- **Application Default Credentials (ADC)**: Automatically authenticates via GCP metadata server inside Cloud Run or falls back to service account JSON keys.
- **Index Shards & Destination**: Writes 15-column Snappy Parquet shards to `gs://sm-optimizer-processed/manifests/index_shard_<run_id>_<seq:06d>.parquet`.
- **Co-located Storage**: Deployed to `asia-southeast2` (Jakarta) alongside the GCS bucket `gs://sm-optimizer-processed` for $0 network egress cost.
- **Scraper Packaging**: Deployed as a lightweight Docker container built using [`Dockerfile.scraper`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/Dockerfile.scraper) and [`requirements-scraper.txt`](file:///Users/LFH/code/leonhelfinger/project/social-media-optimizer/requirements-scraper.txt) (based on `python:3.11-slim` + system `ffmpeg`, ~220 MB).

### Building & Deploying

```bash
# 1) Build container image via Cloud Build / Docker
gcloud builds submit --config cloudbuild.yaml

# 2) Deploy Cloud Run Job with 10 parallel tasks & 1h timeout limit
gcloud run jobs create meta-video-scraper-job \
  --image asia-southeast2-docker.pkg.dev/$PROJECT_ID/sm-optimizer-repo/meta-video-scraper:latest \
  --region asia-southeast2 \
  --tasks 10 \
  --cpu 1 \
  --memory 2Gi \
  --max-retries 1 \
  --task-timeout 3600s \
  --set-env-vars GCS_PROCESSED_BUCKET=sm-optimizer-processed,VIDEO_CONCURRENCY=5

# 3) Execute the job
gcloud run jobs execute meta-video-scraper-job --region asia-southeast2
```

---

## 🍌 MinionsScout — Live Command Center & Monitoring Dashboard

**MinionsScout** is a high-performance, real-time command center for monitoring ingestion progress, throughput, and estimated completion times across all platforms.

### Features
- **⚡️ Instant Load (< 15ms)**: Powered by an in-memory & disk cache (`data/.gcs_stats_cache.json`) for zero-delay page loads.
- **🎯 Targeted GCS Prefix Scanning**: Optimized scan queries targeting only active prefixes (`videos/facebook/`, `videos/instagram/`), reducing API latency from ~7s to ~1s.
- **🔄 On-Demand Sync Button**: Interactive `⚡️ Refresh Data` button with live spinner feedback and smooth 60 FPS numeric lerp animations.
- **🌍 Dynamic Client Timezone Detection**: Automatically formats all ETAs, sync timestamps, and live clocks in the viewer's local timezone (`Intl.DateTimeFormat`).
- **📈 Stabilized EMA Speed & ETA**: Exponential Moving Average ($\\alpha = 0.15$) filter over a rolling window to eliminate abrupt spikes from parallel thread completions.
- **🎬 Hardware-Accelerated 480p Video Preview**: Live embedded preview of the latest processed video deliverables (`/api/video/latest`) with zero Cumulative Layout Shift (CLS).
- **🌐 Global Live Sharing**: Instant public HTTPS tunnel integration via Cloudflare Tunnel.

### Launching the Dashboard

```bash
# 1) Start the high-performance MinionsScout Web Server (Port 8505)
python -m src.video.web_dashboard 8505

# 2) Optional: Create an instant public live HTTPS link
cloudflared tunnel --url http://localhost:8505

# 3) Optional: Launch the Streamlit dashboard
streamlit run src/video/dashboard.py
```

---

## 🤖 Automated Telegram & Lark Notifiers

To run automated background status reports to Telegram or Lark channels with rolling ETA updates:

```bash
# Set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID or LARK_WEBHOOK_URL in .env
python -m src.video.monitor --interval 300
```

---

## 🧪 Testing

Run the test suite with pytest:

```bash
python -m pytest tests/ -v
```
