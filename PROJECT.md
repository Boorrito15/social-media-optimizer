# Project: Cloud Run Meta Video Scraper Ingestion

## Architecture
- **Source Data**: `data/processed/posts_clean.parquet` (13,142 rows, 9,453 Meta targets: 4,578 Facebook + 4,875 Instagram).
- **Target Bucket**: `gs://sm-optimizer-processed` in `asia-southeast2` (Jakarta). Already has 5,443 Meta videos (57.6%), leaving 4,010 pending videos (2,225 FB + 1,785 IG).
- **Compute Platform**: Google Cloud Run Jobs in `asia-southeast2` (co-located with GCS bucket for $0.00 network egress cost).
- **Packaging**: Lightweight Docker container (~220 MB) based on `python:3.11-slim` + system `ffmpeg` + `requirements-scraper.txt`. Image built and verified: `asia-southeast2-docker.pkg.dev/le-wagon-2303/sm-optimizer-repo/meta-video-scraper:latest`.
- **Task Sharding**: 10 parallel tasks sharded deterministically via `index % CLOUD_RUN_TASK_COUNT == CLOUD_RUN_TASK_INDEX`, each running internal `DynamicSupervisor` with 2 threads (20 total concurrency).
- **Idempotency & Dashboard**: Pre-flight batch listing skips existing GCS videos at O(1) in-memory cost; new uploads to `videos/<platform>/<post_id>.mp4` and 15-column Snappy Parquet shards in `manifests/` stream live into MinionsScout dashboard on port 8505.
- **Budget Guardrail**: Expected compute spend ~$1.33 USD. Hard ceiling: 10 tasks × 1 vCPU × 2 GiB × 3600s = $1.46 USD (strictly under $2.00 USD).

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Git Branch Isolation | Branch `meta_layer_scr_cloud_scraper` strictly from `meta_layer_scr` with no remote pushes | M1 | Survey |
| 2 | Cloud Run Task Sharding | Interleaved modulo sharding for `CLOUD_RUN_TASK_INDEX` and `CLOUD_RUN_TASK_COUNT` | M1 | Survey |
| 3 | Cloud Run ADC Support | Allow GCP Application Default Credentials in `main.py` when service account key file is not present | M1 | Survey |
| 4 | Scraper Dependency Isolation | Dedicated `requirements-scraper.txt` omitting heavy ML libraries | M1 | Survey |
| 5 | Scraper Containerization | Lightweight multi-stage `Dockerfile.scraper` (~220 MB) with `python:3.11-slim` & `ffmpeg` | M2 | Survey |
| 6 | Artifact Registry & Cloud Build | Build and push image to `asia-southeast2-docker.pkg.dev/le-wagon-2303/sm-optimizer-repo/meta-video-scraper:latest` | M2 | Survey |
| 7 | Cloud Run Job Creation | Create job with 10 tasks, 10 parallelism, 1 vCPU, 2 GiB RAM, 3600s timeout in `asia-southeast2` | M3 | Survey |
| 8 | Cloud Run Job Execution | Execute job to ingest 4,010 pending Meta videos into `gs://sm-optimizer-processed` | M3 | Survey |
| 9 | GCS Idempotent Ingestion | Upload to `videos/facebook/` and `videos/instagram/`, skipping existing blobs | M3 | Survey |
| 10 | Real-Time Index Sharding | Emit 15-column Snappy Parquet shards to `manifests/` | M3 | Survey |
| 11 | MinionsScout Live Sync | Real-time tracking verification on MinionsScout web dashboard | M4 | Survey |
| 12 | Budget & Spend Verification | Compute cost tracking ensuring total spend strictly < $2.00 USD | M4 | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Branch & Pipeline Core Setup | Create `meta_layer_scr_cloud_scraper`, task sharding logic, ADC support, `requirements-scraper.txt` | none | DONE |
| M2 | Containerization & Cloud Build | `Dockerfile.scraper`, Artifact Registry setup in `asia-southeast2`, Cloud Build push | M1 | DONE |
| M3 | Cloud Run Job Deployment & Execution | Deploy and run Cloud Run Job with 10 tasks in `asia-southeast2` with budget caps | M2 | IN_PROGRESS |
| M4 | Ingestion Verification & Dashboard Tracking | Validate GCS video counts, Parquet shards, MinionsScout live status, compute spend audit | M3 | PLANNED |

## Interface Contracts
### Cloud Run Task Sharding
- Environment Variables: `CLOUD_RUN_TASK_INDEX` (int 0..N-1), `CLOUD_RUN_TASK_COUNT` (int N, default 1).
- Selection: Filter dataset to `df[df.index % CLOUD_RUN_TASK_COUNT == CLOUD_RUN_TASK_INDEX]`.
- Destination: `gs://sm-optimizer-processed/videos/<platform>/<post_id>.mp4`
- Shard Manifest: `gs://sm-optimizer-processed/manifests/index_shard_<run_id>_<seq:06d>.parquet` (15 columns, snappy compression).

## Code Layout
- `Dockerfile.scraper`: Lightweight Dockerfile for Cloud Run Jobs scraper.
- `requirements-scraper.txt`: Minimal scraper runtime dependencies.
- `src/video/main.py`: Scraper entrypoint with ADC fallback and task sharding.
- `src/video/dynamic_pool.py`: Multi-platform worker pool supervisor.
- `src/video/upload.py`: Resolution, download, transcode, GCS upload, and disk cleanup.
- `src/video/index.py`: Index shard emission to GCS.
- `src/video/web_dashboard.py`: MinionsScout live web dashboard server.
