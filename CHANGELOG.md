# Changelog

All notable changes to this project are documented here, grouped by logical
feature areas. This file is maintained alongside the code so collaborators —
new and existing — can see what moved and why.

The project is currently pre-release: all changes appear under
**`[Unreleased]`** until the pipeline reaches a stable first release.

## [Unreleased]

### Added
- **`utils/config.py`** — centralised, typed environment/config loader
  (string/int/float/bool/path getters, grouped GCP/GCS/Gemini/video accessors,
  service-account key resolution that prefers the project key over a stale
  ambient variable).
- **`src/ingestion/`** (Phase 2 — data ingestion & cleaning)
  - `clean.py`: maps vendor `NZR - ...` columns to a stable schema, filters to
    short-form posts on target platforms, removes duplicate links (keeps the
    most-engaged representative), normalises types and drops blank rows.
  - `main.py`: CLI that always writes **both** `posts_clean.csv` and
    `posts_clean.parquet`.
  - `summary.py`: per-stage run summary (rows removed per step), JSON-exportable.
- **`src/video/`** (Phase 3 — video pipeline)
  - `download.py`: per-platform resolver + `yt-dlp` downloader; platform↔name
    mapping and `media_id_from_url` extraction; browser-cookie authentication.
  - `upload.py`: `resolve → download → 480p transcode → GCS upload`
    orchestration; idempotent skip; crash protection.
  - `index.py`: incremental index shards, cumulative `videos_index.parquet`,
    log shipping, and backfill from existing GCS objects.
  - `main.py`: CLI with platform/limit/dry-run/transcode/concurrency/run-id/log
    /consolidate/cookies options.
- **`utils/gcs.py`** — GCS upload via `google-cloud-storage` (with `gsutil`
  fallback), `object_exists`, `list_existing_objects`, `sha256_file`,
  skip-if-exists, retries.
- **`utils/ffmpeg.py`** — 480p transcode profile builder (H.264/AV1, 96k audio,
  ～30 FPS cap, `yuv420p`, CRF + VBV rate cap) and `ffprobe` duration probe.
- **Tests** — `tests/test_ingestion.py` and `tests/test_video.py`.

### Optimised
- **Fast GCS skip** — the video pipeline now does **one** `list_existing_objects()`
  call at startup to load all already-uploaded videos into memory, then skips any
  post whose object already exists (O(1) lookup) instead of a per-video network
  round-trip. This makes re-runs/resumes and cross-machine sharing dramatically
  cheaper and faster.
- **Parallel/concurrent scraping** — `--concurrency N` / `VIDEO_CONCURRENCY`
  processes posts in parallel (YouTube stays serial due to throttling).
- **Rate-limit control** — `VIDEO_REQUEST_DELAY` paces anonymous requests.

### Security / hardening
- Git-ignore raw data + generated outputs: `data/raw/`, `data/processed/`,
  `data/videos/`, plus global `*.csv`, `*.tsv`, `*.jsonl`, `*.parquet`, `*.log`,
  cookies files, `.env`, and `config/*.json` (service-account keys).
- GCS service-account credentials and `.env` are never committed.

### Fixed
- **Broken import** — `src/video/upload.py` referenced `utils.hashing`, which
  does not exist; `sha256_file` lives in `utils.gcs`. The import now points to
  `utils.gcs`, so the pipeline imports and runs correctly (all tests pass).

### Removed
- Data files purged from **all** git history (raw + processed CSV/Parquet) so
  they exist only locally (and in GCS), not in version control.

---

## History (by contributor)

Practical change history condensed from `git log`, roughly oldest → newest.

### Project bootstrap
- Initial project structure and notebooks.
- `requirements.txt`, `.gitignore`, `.env.example` with GCP bucket config.

### Contributor: **Alden Wahsono**
- Phase 2 ingestion: `src/ingestion/` (clean, CLI, summary) + tests.
- Phase 3 video pipeline: `utils/gcs.py`, `utils/ffmpeg.py`, `src/video/`
  (download, upload, index, main) + tests.
- Incremental index + cumulative `videos_index.parquet` + log shipping.
- Data purge from git history + hardened `.gitignore`.
- Fixed the `utils.hashing` broken import (see above).

### Contributor: **Leon Helfinger**
- PR #4 — fast skip-existing + GCS object listing utility
  (`list_existing_objects`) for cheap idempotent re-runs.
- PR #5 — browser cookie authentication and realistic desktop headers:
  `--cookies` / `--cookies-from-browser` CLI flags,
  `_DEFAULT_HTTP_HEADERS`, `_build_ydl_opts` dynamic cookie injection,
  `ytdlp_cookies_from_browser()` config getter, plus unit tests.

### Contributor: **Robert Safin**
- Notebook / dataset file maintenance.

---

## Test status

```bash
python -m pytest tests/ -q   # currently 27 passing
```
