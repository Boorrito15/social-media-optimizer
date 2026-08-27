# Changelog

Reverse-chronological timeline (newest first) of committed changes, extracted
from the GitHub repository history. Each day groups its commits with the exact
ISO-8601 time and author. For a quick read, start with the **summary below**;
for the raw commit-by-commit record, scroll to the timeline.

Contributors referenced here:
- **Alden** (aldenwahsono / Alden Wahsono)
- **Leon** (Leon Helfinger / leonhelfinger)
- **Robert** (Robert Safin / Robert-Safin)

---

## What changed & why it matters (plain-English summary)

### The project now actually works end-to-end
The repo went from a bare skeleton to a working pipeline that:
1. **Reads the raw social-media funnel CSV**, cleans it (standardises the messy
   vendor column names, keeps only short-form posts on the target platforms, and
   removes duplicate links, keeping the most-engaged copy of each), and writes
   the result as both a CSV and a faster Parquet file.
2. **Downloads the actual videos** behind those posts (via `yt-dlp`), re-encodes
   each one to a **480p MP4** for consistency and smaller size, and **uploads it
   to Google Cloud Storage**.

So instead of just a cleaned spreadsheet, we now have the real short-form videos
stored centrally in GCS — which is what we need for the downstream vision/ML work.

### Scraping is now realistic and won't balloon your GCS bill
Two things made a big practical difference:

- **Browser-cookie authentication (Leon's "cookie-layer"):** The scraper can now
  present a logged-in browser's cookies (via `--cookies` or `--cookies-from-browser`)
  plus realistic desktop browser headers. This makes the requests look like a real
  person instead of a bot, which **dramatically reduces the chance of YouTube/TikTok
  rate-limiting or blocking us** — the exact problem that kept stalling the runs.
- **Fast skip of already-downloaded videos ("skip_existing"):** Before processing,
  the pipeline asks GCS once for the full list of videos we already have, then skips
  anything already uploaded. Previously it checked per-video (hundreds/thousands of
  network calls); now it's one call. This makes **re-runs and resumes nearly free**,
  and lets several people (or machines) share the same GCS bucket without re-doing
  each other's work.

### The pipeline is resilient to interruptions
We now write a durable **index / manifest / logs** to GCS as we go, so if a run
crashes mid-way (it has), we don't lose track of what was already uploaded or what
failed — we can resume and inspect failures afterward instead of starting over.

### We tightened up git hygiene & fixed a latent bug
- **Security/data:** raw data, generated outputs, service-account keys, `.env`, and
  cookies are all **git-ignored** (never committed), and data files that had been
  committed were **purged from git history** so they exist only locally and in GCS.
- **A real bug was fixed:** the merged `cookie-layer` code imported a module
  (`utils.hashing`) that didn't exist — it would have crashed the moment you ran
  anything. We pointed that import at the actual file and confirmed all **27 tests
  pass**, so the pipeline runs instead of failing instantly.

### Where things stand
- **Phase 2 (ingest/clean):** done.
- **Phase 3 (scrape → 480p → GCS):** done for **YouTube** and **TikTok**.
- **Instagram / Facebook:** still stubbed/queued (they need a real login/browser
  strategy, which is a separate piece of work).

---

## 2026-08-27

- **15:48** (Alden) `b2c9cfe` — docs: overhaul README + add CHANGELOG
- **15:35** (Alden) `1bc3e52` — fix(video): import `sha256_file` from `utils.gcs` not `utils.hashing`
- **15:12** (Alden) `d67f22c` — chore: harden gitignore against data/credentials/artifacts
- **15:08** (Leon) `1a676d6` — Merge pull request #5 from `Boorrito15/cookie-layer`
- **15:07** (Leon) `56ff0f5` — merge: resolve conflicts with origin/main keeping cookie layer
- **14:53** (Leon) `4b1b10a` — feat(video): browser cookie authentication + desktop headers
  (`--cookies`, `--cookies-from-browser`, `_DEFAULT_HTTP_HEADERS`, `_build_ydl_opts`,
  `ytdlp_cookies_from_browser`, unit tests)
- **14:37** (Alden) `9701ce5` — chore: ignore generated/raw datasets (`data/*`, `*.parquet`)
- **13:51** (Alden) `2a078e5` — Merge pull request #4 from `Boorrito15/fix/autoskip-video-ingestion`
- **13:06** (Leon) `1370afc` — feat: skip-existing logic + `list_existing_objects` (fast idempotent re-runs)
- **11:34** (Alden) `bc8f7c1` — docs: teammate scraping setup (clone + manual GCS key/.env)
- **10:07** (Alden) `2a6d13c` — Merge pull request #3 from `Boorrito15/feature/video-scraper`
- **10:06** (Alden) `8d5cf92` — Merge pull request #2 from `Boorrito15/feature/ingestion-cleaning`
- **10:06** (Alden) `8659c0f` — Merge pull request #1 from `Boorrito15/chore/project-config`
- **09:52** (Alden) `80c1912` — test(video): resolver, download, pipeline + index
- **09:52** (Alden) `fc12f4a` — feat(video): CLI entrypoint
- **09:52** (Alden) `e6f3f76` — feat(video): `resolve → download → 480p → GCS` orchestration
- **09:52** (Alden) `fd0e148` — feat(video): index shards + cumulative registry + log shipping
- **09:52** (Alden) `cfb5dfd` — feat(video): per-platform resolver + `yt-dlp` downloader
- **09:52** (Alden) `4606b9b` — feat(utils): GCS upload + ffmpeg 480p transcode helpers
- **09:51** (Alden) `00f29dc` — test(ingestion): cleaning + dedup + summary
- **09:51** (Alden) `c1570b8` — feat(ingestion): CLI entrypoint (CSV + Parquet)
- **09:51** (Alden) `66b2d2f` — feat(ingestion): run summary with per-stage counts
- **09:51** (Alden) `f0c151b` — feat(ingestion): cleaning/normalisation for master dataset
- **09:51** (Alden) `81285be` — feat(core): config loader + package init
- **09:50** (Alden) `7bb431f` — docs: project setup, GCP buckets, module layout
- **09:50** (Alden) `e654512` — chore(deps): runner deps + tighten gitignore for credentials

## 2026-08-26

- **14:50** (Robert) `b556d6e` — update notebook
- **13:54** (Robert) `549f74d` — Rename `Untitled.ipynb` → `rob.ipynb`
- **13:54** (Robert) `45f09aa` — Delete `notebooks/rob.ipynb`
- **13:53** (Robert) `f005b46` — Add files via upload

## 2026-08-25

- **14:35** (Alden) `404de03` — chore: set up GCP bucket in `.env.example`
- **13:31** (Alden) `b8ecd44` — chore: initial `requirements.txt`
- **13:29** (Alden) `73b92ca` — chore: added `.gitignore`
- **13:27** (Alden) `c844be8` — chore: dedicated notebooks per team member
- **13:21** (Alden) `b36611e` — Initial project structure

---

## Summary of work by area

| Area | Status |
| ---- | ------ |
| Data ingestion & cleaning (Phase 2) | ✅ `src/ingestion/` |
| Video pipeline — scrape → 480p → GCS (Phase 3) | ✅ `src/video/` |
| Fast GCS skip / idempotency (Leon PR #4) | ✅ |
| Browser-cookie auth for scraping (Leon PR #5) | ✅ |
| Index / manifest / log shipping | ✅ |
| Git hygiene — data + credentials ignored | ✅ |
| Data files purged from history | ✅ |

**Tests:** `python -m pytest tests/ -q` — currently **27 passing**.
