# TEST_READY: Cloud Run Meta Video Scraper Test Suite

**Date:** 2026-08-28T01:18:10Z  
**Test Suite Path:** `tests/test_cloud_scraper.py`  
**Execution Command:** `.venv/bin/python -m pytest tests/test_cloud_scraper.py -v`  
**Status:** **READY & 100% PASSING** (34 passed out of 34 tests in 0.88s)

---

## 1. Test Suite Coverage Summary

| # | Feature / Area | Test Class | Test Count | Status | Description |
|---|---|---|:---:|:---:|---|
| 1 | **Task Sharding Logic** | `TestCloudRunTaskSharding` | 9 | **PASSED** | Deterministic modulo partitioning (`row_idx % task_count == task_index`), completeness, disjointness, uniform distribution ($\Delta \le 1$), interleaving FB/IG balance, 4,010 workload sharding, env vars (`CLOUD_RUN_TASK_INDEX`/`COUNT`), boundary & error cases. |
| 2 | **ADC Fallback** | `TestApplicationDefaultCredentialsFallback` | 5 | **PASSED** | Verification of GCP Application Default Credentials fallback when local key file is absent, explicit service account key precedence, dry-run credential bypass, and `storage.Client()` factory initialization. |
| 3 | **Minimal Dependencies** | `TestMinimalScraperDependencies` | 4 | **PASSED** | Verification that `requirements-scraper.txt` contains only core scraper dependencies (`yt-dlp`, `pandas`, `pyarrow`, `google-cloud-storage`, `python-dotenv`) and strictly excludes heavy ML/UI libraries (`torch`, `whisper`, `google-genai`, `xgboost`, `scikit-learn`, `streamlit`, `plotly`), maintaining a lightweight image (<250 MB). |
| 4 | **Budget & Cost Model** | `TestBudgetAndCostCalculation` | 5 | **PASSED** | Verification of `asia-southeast2` pricing rates ($0.00003360/vCPU-s, $0.00000350/GiB-s), hard ceiling math ($1.4616 USD <= $1.462 USD < $2.00 USD for 10 tasks x 3600s x 1 vCPU / 2 GiB), $0.00 same-region egress, and 4,010 video workload model ($0.6512 USD compute + $0.021 USD GCS ops). |
| 5 | **Index Shard Schema** | `TestIndexShardParquetSchemaCompliance` | 5 | **PASSED** | Exact 15-column schema enforcement (`INDEX_SCHEMA` & `_MANIFEST_SCHEMA`), Snappy Parquet compression metadata verification, schema normalization of sparse records, empty record handling, and failed status sheet routing. |
| 6 | **Idempotency Skipping** | `TestIdempotencySkippingLogic` | 3 | **PASSED** | GCS pre-flight batch listing skips existing blobs with zero download/transcode I/O, full GCS path validation (`gs://<bucket>/videos/<platform>/<id>.mp4`), and `--no-skip-existing` re-processing flag. |
| 7 | **Adversarial & Edge Cases** | `TestAdversarialEdgeCases` | 3 | **PASSED** | Multibyte Unicode/emoji titles and URL escaping in Parquet round-trip, extreme task count partitioning (100 tasks on 5 items), and microsecond duration edge cases. |

---

## 2. Test Execution Verification

```bash
$ .venv/bin/python -m pytest tests/test_cloud_scraper.py -v
============================= test session starts ==============================
platform darwin -- Python 3.12.9, pytest-9.1.1, pluggy-1.6.0 -- .venv/bin/python
rootdir: /Users/LFH/code/leonhelfinger/project/social-media-optimizer
collected 34 items

tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_modulo_sharding_completeness PASSED [  2%]
tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_modulo_sharding_disjointness PASSED [  5%]
tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_modulo_sharding_uniform_distribution PASSED [  8%]
tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_modulo_sharding_platform_interleaving PASSED [ 11%]
tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_modulo_sharding_realistic_meta_workload PASSED [ 14%]
tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_boundary_single_task PASSED [ 17%]
tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_boundary_more_tasks_than_rows PASSED [ 20%]
tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_invalid_task_parameters_raise_value_error PASSED [ 23%]
tests/test_cloud_scraper.py::TestCloudRunTaskSharding::test_environment_variable_sharding_resolution PASSED [ 26%]
tests/test_cloud_scraper.py::TestApplicationDefaultCredentialsFallback::test_service_account_credentials_returns_none_when_unset_and_no_file PASSED [ 29%]
tests/test_cloud_scraper.py::TestApplicationDefaultCredentialsFallback::test_gcs_client_initialization_with_adc PASSED [ 32%]
tests/test_cloud_scraper.py::TestApplicationDefaultCredentialsFallback::test_gcs_client_initialization_with_explicit_service_account PASSED [ 35%]
tests/test_cloud_scraper.py::TestApplicationDefaultCredentialsFallback::test_dry_run_mode_bypasses_credentials_requirement PASSED [ 38%]
tests/test_cloud_scraper.py::TestApplicationDefaultCredentialsFallback::test_explicit_key_used_when_present PASSED [ 41%]
tests/test_cloud_scraper.py::TestMinimalScraperDependencies::test_requirements_scraper_file_exists PASSED [ 44%]
tests/test_cloud_scraper.py::TestMinimalScraperDependencies::test_essential_scraper_packages_present PASSED [ 47%]
tests/test_cloud_scraper.py::TestMinimalScraperDependencies::test_heavy_ml_packages_strictly_excluded PASSED [ 50%]
tests/test_cloud_scraper.py::TestMinimalScraperDependencies::test_minimal_package_count PASSED [ 52%]
tests/test_cloud_scraper.py::TestBudgetAndCostCalculation::test_asia_southeast2_pricing_constants PASSED [ 55%]
tests/test_cloud_scraper.py::TestBudgetAndCostCalculation::test_hard_compute_ceiling_strictly_under_2_dollars PASSED [ 58%]
tests/test_cloud_scraper.py::TestBudgetAndCostCalculation::test_expected_workload_cost_model PASSED [ 61%]
tests/test_cloud_scraper.py::TestBudgetAndCostCalculation::test_zero_egress_cost_for_colocated_bucket PASSED [ 64%]
tests/test_cloud_scraper.py::TestBudgetAndCostCalculation::test_cost_scaling_linearity PASSED [ 67%]
tests/test_cloud_scraper.py::TestIndexShardParquetSchemaCompliance::test_index_schema_definition_has_15_columns PASSED [ 70%]
tests/test_cloud_scraper.py::TestIndexShardParquetSchemaCompliance::test_append_index_shard_emits_snappy_parquet PASSED [ 73%]
tests/test_cloud_scraper.py::TestIndexShardParquetSchemaCompliance::test_schema_normalization_fills_missing_columns PASSED [ 76%]
tests/test_cloud_scraper.py::TestIndexShardParquetSchemaCompliance::test_empty_records_returns_none PASSED [ 79%]
tests/test_cloud_scraper.py::TestIndexShardParquetSchemaCompliance::test_failed_sheet_filters_only_failures PASSED [ 82%]
tests/test_cloud_scraper.py::TestIdempotencySkippingLogic::test_preflight_batch_listing_skips_existing_gcs_blobs PASSED [ 85%]
tests/test_cloud_scraper.py::TestIdempotencySkippingLogic::test_single_item_process_one_skips_when_object_exists PASSED [ 88%]
tests/test_cloud_scraper.py::TestIdempotencySkippingLogic::test_no_skip_existing_forces_download_attempt PASSED [ 91%]
tests/test_cloud_scraper.py::TestAdversarialEdgeCases::test_unicode_and_special_character_metadata_fidelity PASSED [ 94%]
tests/test_cloud_scraper.py::TestAdversarialEdgeCases::test_extreme_task_count_higher_than_dataset PASSED [ 97%]
tests/test_cloud_scraper.py::TestAdversarialEdgeCases::test_extreme_zero_and_microsecond_duration_costs PASSED [100%]

============================== 34 passed in 0.88s ==============================
```

---

## 3. Discovered Findings & Recommendations

- **ADC Support in Entry Point**: In `src/video/main.py`, lines 114-123 should permit execution when ADC credentials are active (`google.auth.default()`) rather than strictly aborting if a local service account `.json` file does not exist on disk.
- **Task Sharding Integration**: Ensure `src/video/main.py` accepts `--task-index` and `--task-count` arguments and inspects `CLOUD_RUN_TASK_INDEX` / `CLOUD_RUN_TASK_COUNT` environment variables to slice the input posts DataFrame before passing it to `run_pipeline`.
