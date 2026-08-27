# Social Media Optimizer

All Blacks short-form content optimisation. Ingests the master social-media
funnel dataset, cleans and de-dupes it, then labels targets for downstream
modelling.

## Setup

```bash
# 1) Create a virtualenv and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure environment
cp .env.example .env
#    - set GEMINI_API_KEY if using Gemini direct
#    - drop your service-account JSON into config/ and point
#      GOOGLE_APPLICATION_CREDENTIALS at it (e.g. config/service-account.json)
```

Environment variables are read through `utils/config.py`; see `.env.example`
for every supported variable.

## GCP resources

Configured via the service-account key in `config/` (git-ignored). The project
uses two regional (standard) buckets in `asia-southeast2`:

| Bucket                   | Purpose                        |
| ------------------------ | ------------------------------ |
| `sm-optimizer-raw`      | raw source data                |
| `sm-optimizer-processed`| cleaned / labelled outputs     |

## Phase 2 — Data ingestion & cleaning

Cleans the master funnel CSV:

- standardises the vendor column names (`NZR - ...`) into a stable schema
- keeps short-form (`Short Video`) posts on the target platforms (FB, IG, TT, YT)
- removes duplicate links — for rows sharing a post URL it keeps the single
  most-engaged representative and drops the rest
- normalises string / numeric types and drops fully-blank rows

```bash
# defaults come from RAW_DATA_PATH / CLEAN_DATA_PATH in .env
python -m src.ingestion.main

# explicit input + output (the base path determines stem + directory)
python -m src.ingestion.main --input data/raw/example.csv --output data/processed/out

# skip the duplicate-link removal (inspection only)
python -m src.ingestion.main --no-dedupe
```

Every run writes **both** a Parquet and a CSV of the processed data, derived
from the same base path: `posts_clean.parquet` and `posts_clean.csv` under
`data/processed/` by default.

### Run summary

Each run prints a summary of how many rows were removed at every stage
(platform filter, media-type filter, blank rows, duplicate links, etc.), and
writes the same data to a JSON file next to the output
(`<output>.summary.json`), or to a path given with `--report`:

```
Input rows                                   76,673
  after platform filter                     56,292
  after media-type filter                   15,464
  after dropping blank URL+content rows      15,464
  after removing duplicate links            13,142
Output rows                                 13,142

Rows removed by stage:
  - after_platform_filter               20,381 removed
  - after_media_type_filter             40,828 removed
  - after_dedupe                         2,322 removed
```

The same counts are available programmatically via
`df.attrs["summary"]` after calling `clean_dataframe` (recoverable with
`CleanSummary.from_dict(...)`).

## Phase 3 — Video pipeline (scrape → 480p → GCS)

Downloads each short-form post's source video, re-encodes it to a consistent
**480p** H.264/AAC profile and uploads it to GCS. It is modular: shared GCS
and ffmpeg helpers live in `utils/`, and the `src/video/` package orchestrates
`resolve → download → transcode → upload`.

Scraping strategy per platform (see `src/video/download.py`):

| Platform | Status       | Method                          |
| -------- | ------------ | ------------------------------- |
| YouTube  | ✅ automated | `yt-dlp` (no browser)           |
| TikTok   | ✅ automated | `yt-dlp` (no browser)           |
| IG / FB  | ⏳ queued    | stubbed; needs browser/login later |

```bash
# Dry-run trace on 5 posts (no network / GCS writes)
python -m src.video.main --limit 5 --dry-run

# Process only YouTube, limit to 10
python -m src.video.main --platforms youtube --limit 10

# Run for real (downloads + 480p transcode + upload to GCS)
python -m src.video.main
```

- Output object layout: `gs://sm-optimizer-processed/<VIDEO_GCS_PREFIX>/<platform>/<post_id>.mp4`.

### 480p transcode profile

`utils/ffmpeg.py` re-encodes every video to a consistent profile (configurable
via env):

| Setting            | Value               | Env var            |
| ------------------ | ------------------- | ------------------ |
| Height             | 480 (auto width)    | `VIDEO_TARGET_HEIGHT` |
| Codec              | `h264` (or `av1`)   | `VIDEO_CODEC`      |
| Quality / CRF      | 23                  | `VIDEO_CRF`        |
| Frame rate cap     | 30 FPS              | —                  |
| Audio              | AAC 96k             | —                  |
| Containers         | `yuv420p`, `+faststart`, VBV `maxrate 3M` | — |

`VIDEO_CODEC=av1` switches to SVT-AV1 (~30–50% smaller, slower encode).

### Manifest, index & logs (written to GCS)

The pipeline writes durable, query-able artifacts to
`gs://sm-optimizer-processed/manifests/` so progress and failures survive a
crash and are inspectable after the fact:

- **Incremental index shards** — `index_shard_<run>_<seq>.parquet`, flushed every
  `index_flush_every` (default 20) posts. One row per processed video with
  `platform, post_id, url, status, gcs_path, published_at, duration_s, title,
  sha256, size_bytes, source_codec, source_resolution, error, processed_at,
  transcode_args`. A crashed run still leaves a durable record.
- **Full-run manifest** — `video_manifest_<ts>.parquet`, written at the end of a
  successful complete pass.
- **Cumulative index** — `manifests/videos_index.parquet`, the consolidated
  registry of everything uploaded/failed (build with `--consolidate`).
- **Run logs** — pass `--log <file>` to also upload the run's stdout to
  `logs/<run_id>.log`.

```bash
# Consolidate all index shards into the cumulative videos_index.parquet
python -m src.video.main --consolidate

# Rebuild the cumulative index from existing GCS objects (for videos uploaded
# before incremental indexing existed) — see src/video/index.py::rebuild_index_from_gcs
```

### Idempotency & concurrency

- **Idempotent** — videos already in GCS (and already-transcoded locals) are
  skipped, so any machine can resume the shared batch; GCS is the source of truth.
- **Concurrency** — `--concurrency N` / `VIDEO_CONCURRENCY` processes posts in
  parallel (8 cores unused by serial runs). YouTube throttles under parallel
  anonymous downloads, so keep it serial (`1`) with a small `VIDEO_REQUEST_DELAY`
  for YouTube; TikTok benefits from higher concurrency (e.g. `3`).
- Requires **ffmpeg** on PATH for the transcode step (see `requirements.txt`).

Videos and all artifacts land in **GCS, not the local machine** — the local
disk is only a transient staging area (`data/videos/`, git-ignored).

## Tests

```bash
python -m pytest tests/ -q
```
