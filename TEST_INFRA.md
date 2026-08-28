# E2E Test Infra: Cloud Run Meta Video Scraper

## Test Philosophy
- Opaque-box and unit/integration verification derived from requirements and GCP Cloud Run execution constraints.
- Methodology: Category-Partition + BVA + Cross-Feature Combinations + Real-World Workload Scenarios.

## Feature Inventory
| # | Feature | Source | Tier 1 | Tier 2 | Tier 3 |
|---|---------|--------|:------:|:------:|:------:|
| 1 | Git Branch Isolation | ORIGINAL_REQUEST §1 | 5 | 5 | ✓ |
| 2 | Cloud Run Task Sharding | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 3 | Cloud Run ADC Support | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 4 | Scraper Dependency Isolation | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 5 | Containerization & Dockerfile | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 6 | Cloud Build & Artifact Registry | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 7 | Cloud Run Job Config & Budget | ORIGINAL_REQUEST §3 | 5 | 5 | ✓ |
| 8 | Cloud Run Job Execution | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 9 | GCS Ingestion & Idempotency | ORIGINAL_REQUEST §4, §5 | 5 | 5 | ✓ |
| 10 | Index Shard Parquet Emission | ORIGINAL_REQUEST §4 | 5 | 5 | ✓ |
| 11 | MinionsScout Live Sync | ORIGINAL_REQUEST §4 | 5 | 5 | ✓ |
| 12 | Budget Cap Enforcement | ORIGINAL_REQUEST §3 | 5 | 5 | ✓ |

## Test Architecture
- Test Runner: `pytest`
- Test Files: `tests/test_cloud_scraper.py`, `tests/test_dynamic_pool.py`, `tests/test_video.py`, `tests/test_ingestion.py`
- Directory layout: standard repo tests directory.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Task Sharding & Backlog Partitioning | F2, F9, F10 | Medium |
| 2 | Docker Build & Minimal Dependency Check | F4, F5 | Medium |
| 3 | Idempotency Pre-flight GCS Skip | F9, F10 | Medium |
| 4 | Index Shard 15-Column Parquet Validation | F10, F11 | High |
| 5 | Budget Formula & Hard Limit Safety Audit | F7, F12 | High |

## Coverage Thresholds
- Tier 1: ≥ 5 per feature
- Tier 2: ≥ 5 per feature
- Tier 3: Pairwise coverage of feature interactions
- Tier 4: ≥ 5 realistic application scenarios
