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
| `--platforms <a,b>` | Comma-separated platforms to process (youtube, tiktok, instagram, facebook) |
| `--limit N` | Process at most `N` posts (dry-run/scratch work) |
| `--dry-run` | Trace the pipeline without downloading/transcoding/uploading |
| `--no-transcode` | Upload the source (original resolution) instead of the 480p |
| `--concurrency N` | Number of parallel workers (default `VIDEO_CONCURRENCY`) |
| `--run-id <id>` | Stable id used to name index shards + logs |
| `--log <file>` | Also upload the run's stdout to `logs/<run_id>.log` |
| `--consolidate` | Rebuild the cumulative `videos_index.parquet` afterwards |
| `--cookies <file>` | Path to a Netscape `cookies.txt` for authenticated scraping |
| `--cookies-from-browser <browser>` | Load cookies from an installed browser (chrome, safari, ...) |

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
