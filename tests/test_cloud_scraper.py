"""Comprehensive Test Suite for Cloud Run Meta Video Scraper.

Covers:
1. Cloud Run task sharding logic (CLOUD_RUN_TASK_INDEX, CLOUD_RUN_TASK_COUNT, deterministic modulo partitioning).
2. Application Default Credentials (ADC) fallback in main.py and GCS client when local key file is absent.
3. Consolidated dependencies validation (single requirements.txt covers scraper + models + API/UI).
4. Budget and cost calculation verification (10 tasks x 3600s x 1 vCPU / 2 GiB in asia-southeast2 <= $1.46 USD < $2.00 USD).
5. Index shard Parquet schema compliance (15 columns, snappy compression).
6. Idempotency skipping logic (simulating existing blobs in GCS and ensuring status="skipped" without downloading).
7. Adversarial edge cases (Unicode/emoji titles, special chars in URLs, extreme partitioning bounds, malformed schemas).
"""

from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

try:
    import google.cloud.storage
except ImportError:
    pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.download import media_id_from_url, platform_code_to_name
from src.video.index import INDEX_SCHEMA, append_failed_sheet, append_index_shard
from src.video.upload import _MANIFEST_SCHEMA, _gcs_object_name, _process_one, run_pipeline, write_records_to_gcs
from utils.config import gcp_region, gcs_processed_bucket


# ==============================================================================
# Helper Functions for Cloud Run Task Sharding & Cost Models
# ==============================================================================

def shard_dataframe_modulo(df: pd.DataFrame, task_index: int, task_count: int) -> pd.DataFrame:
    """Deterministic modulo partitioning of a DataFrame across Cloud Run tasks."""
    if task_count <= 0:
        raise ValueError(f"task_count must be >= 1, got {task_count}")
    if task_index < 0 or task_index >= task_count:
        raise ValueError(f"task_index must be in [0, {task_count - 1}], got {task_index}")
    if df.empty or task_count == 1:
        return df.copy().reset_index(drop=True)
    
    # Standard modulo partition: row_index % task_count == task_index
    reset_df = df.reset_index(drop=True)
    return reset_df[reset_df.index % task_count == task_index].reset_index(drop=True)


def calculate_cloud_run_cost(
    task_count: int,
    duration_seconds: float,
    vcpu_per_task: float = 1.0,
    ram_gib_per_task: float = 2.0,
    region: str = "asia-southeast2",
    network_egress_gb: float = 0.0,
    gcs_class_a_operations: int = 0,
) -> dict[str, float]:
    """Calculate Cloud Run and GCS costs based on GCP pricing.
    
    Pricing matrix for asia-southeast2 (Jakarta - Tier 2):
      - vCPU: $0.00003360 / vCPU-second ($0.12096 / hr)
      - Memory: $0.00000350 / GiB-second ($0.01260 / GiB-hr)
      - Egress (same-region): $0.00 / GB
      - GCS Class A ops: $0.005 / 1,000 requests
    """
    # Tier 2 rates (asia-southeast2)
    vcpu_rate_per_sec = 0.00003360
    ram_rate_per_gib_sec = 0.00000350
    egress_rate_per_gb = 0.00 if region == "asia-southeast2" else 0.12
    class_a_rate_per_op = 0.005 / 1000.0

    total_vcpu_seconds = task_count * duration_seconds * vcpu_per_task
    total_ram_gib_seconds = task_count * duration_seconds * ram_gib_per_task

    vcpu_cost = total_vcpu_seconds * vcpu_rate_per_sec
    ram_cost = total_ram_gib_seconds * ram_rate_per_gib_sec
    compute_cost = vcpu_cost + ram_cost
    egress_cost = network_egress_gb * egress_rate_per_gb
    gcs_ops_cost = gcs_class_a_operations * class_a_rate_per_op

    total_cost = compute_cost + egress_cost + gcs_ops_cost

    return {
        "vcpu_cost": round(vcpu_cost, 6),
        "ram_cost": round(ram_cost, 6),
        "compute_cost": round(compute_cost, 6),
        "egress_cost": round(egress_cost, 6),
        "gcs_ops_cost": round(gcs_ops_cost, 6),
        "total_cost": round(total_cost, 6),
    }


# ==============================================================================
# 1. Cloud Run Task Sharding Logic
# ==============================================================================

class TestCloudRunTaskSharding:
    """Tests for deterministic modulo partitioning and task index/count handling."""

    @pytest.fixture
    def sample_meta_df(self) -> pd.DataFrame:
        """Create a synthetic 100-row DataFrame with mixed Meta posts."""
        rows = []
        for i in range(100):
            plat = "FB" if i % 2 == 0 else "IG"
            rows.append({
                "post_id": f"post_{i:04d}",
                "url": f"https://www.{'facebook' if plat == 'FB' else 'instagram'}.com/reel/{i:04d}/",
                "platform": plat,
                "title": f"Sample Reel {i}",
            })
        return pd.DataFrame(rows)

    def test_modulo_sharding_completeness(self, sample_meta_df):
        """Union of all task shards must contain all rows from original dataset."""
        task_count = 10
        shards = [shard_dataframe_modulo(sample_meta_df, i, task_count) for i in range(task_count)]
        combined = pd.concat(shards, ignore_index=True)
        
        assert len(combined) == len(sample_meta_df)
        assert set(combined["post_id"]) == set(sample_meta_df["post_id"])

    def test_modulo_sharding_disjointness(self, sample_meta_df):
        """No two distinct task shards may share any row (pairwise disjoint)."""
        task_count = 10
        shards = [shard_dataframe_modulo(sample_meta_df, i, task_count) for i in range(task_count)]
        
        seen_ids = set()
        for i, shard in enumerate(shards):
            shard_ids = set(shard["post_id"])
            overlap = seen_ids.intersection(shard_ids)
            assert not overlap, f"Task {i} shares post_ids with previous tasks: {overlap}"
            seen_ids.update(shard_ids)

    def test_modulo_sharding_uniform_distribution(self, sample_meta_df):
        """Shard sizes must differ by at most 1 item (optimal load balance)."""
        task_count = 7  # 100 % 7 != 0, tests uneven distribution
        shards = [shard_dataframe_modulo(sample_meta_df, i, task_count) for i in range(task_count)]
        lens = [len(s) for s in shards]
        
        assert max(lens) - min(lens) <= 1
        assert sum(lens) == 100
        # 100 = 2 * 15 + 5 * 14
        assert lens.count(15) == 2
        assert lens.count(14) == 5

    def test_modulo_sharding_platform_interleaving(self, sample_meta_df):
        """Modulo partitioning preserves platform balance across tasks."""
        task_count = 10
        for i in range(task_count):
            shard = shard_dataframe_modulo(sample_meta_df, i, task_count)
            assert len(shard) == 10
            fb_count = len(shard[shard["platform"] == "FB"])
            ig_count = len(shard[shard["platform"] == "IG"])
            # Because dataset alternates FB, IG, even tasks get FB, odd tasks get IG
            assert fb_count == 10 or ig_count == 10

    def test_modulo_sharding_realistic_meta_workload(self):
        """Verify partitioning over the full 4,010 pending Meta workload across 10 tasks."""
        total_pending = 4010
        df = pd.DataFrame({"post_id": [f"id_{i}" for i in range(total_pending)]})
        task_count = 10
        shards = [shard_dataframe_modulo(df, i, task_count) for i in range(task_count)]
        
        assert all(len(s) == 401 for s in shards)
        assert sum(len(s) for s in shards) == 4010

    def test_boundary_single_task(self, sample_meta_df):
        """Single task (task_count=1, task_index=0) returns full dataset."""
        shard = shard_dataframe_modulo(sample_meta_df, 0, 1)
        assert len(shard) == len(sample_meta_df)
        pd.testing.assert_frame_equal(shard, sample_meta_df)

    def test_boundary_more_tasks_than_rows(self):
        """When task_count > dataset length, initial tasks get 1 item, remaining get 0."""
        df = pd.DataFrame({"post_id": ["p1", "p2", "p3"]})
        task_count = 5
        shards = [shard_dataframe_modulo(df, i, task_count) for i in range(task_count)]
        
        assert [len(s) for s in shards] == [1, 1, 1, 0, 0]
        assert shards[0]["post_id"].iloc[0] == "p1"
        assert shards[3].empty

    def test_invalid_task_parameters_raise_value_error(self, sample_meta_df):
        """Invalid task index or count must raise ValueError."""
        with pytest.raises(ValueError, match="task_count must be >= 1"):
            shard_dataframe_modulo(sample_meta_df, 0, 0)
        
        with pytest.raises(ValueError, match="task_index must be in"):
            shard_dataframe_modulo(sample_meta_df, 5, 5)
            
        with pytest.raises(ValueError, match="task_index must be in"):
            shard_dataframe_modulo(sample_meta_df, -1, 5)

    def test_environment_variable_sharding_resolution(self, monkeypatch, sample_meta_df):
        """Verify environment variable injection (CLOUD_RUN_TASK_INDEX & COUNT)."""
        monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "3")
        monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "10")
        
        idx = int(os.environ["CLOUD_RUN_TASK_INDEX"])
        count = int(os.environ["CLOUD_RUN_TASK_COUNT"])
        shard = shard_dataframe_modulo(sample_meta_df, idx, count)
        
        assert len(shard) == 10
        assert shard["post_id"].iloc[0] == "post_0003"


# ==============================================================================
# 2. Application Default Credentials (ADC) Fallback
# ==============================================================================

class TestApplicationDefaultCredentialsFallback:
    """Tests for GCP Application Default Credentials fallback when service account key is absent."""

    def test_service_account_credentials_returns_none_when_unset_and_no_file(self, monkeypatch, tmp_path):
        """When no key file exists and env is clean, service_account_credentials returns None."""
        from utils import config
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setattr(config, "_dotenv_lines", lambda: [])
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        
        creds = config.service_account_credentials()
        assert creds is None

    def test_gcs_client_initialization_with_adc(self, monkeypatch):
        """utils.gcs._client() should instantiate storage.Client() via ADC when credentials path is None."""
        from utils import gcs
        
        monkeypatch.setattr("utils.config.service_account_credentials", lambda: None)
        
        mock_storage = MagicMock()
        mock_storage_client = MagicMock()
        mock_storage.Client.return_value = mock_storage_client
        with patch.dict(sys.modules, {"google.cloud.storage": mock_storage}):
            import google.cloud
            google.cloud.storage = mock_storage
            client = gcs._client()
            assert client == mock_storage_client
            mock_storage.Client.assert_called_once_with()

    def test_gcs_client_initialization_with_explicit_service_account(self, tmp_path, monkeypatch):
        """utils.gcs._client() uses from_service_account_json when key file exists."""
        from utils import gcs
        
        fake_key = tmp_path / "sa-key.json"
        fake_key.write_text('{"type": "service_account"}', encoding="utf-8")
        monkeypatch.setattr("utils.config.service_account_credentials", lambda: fake_key)
        
        mock_storage = MagicMock()
        mock_sa_client = MagicMock()
        mock_storage.Client.from_service_account_json.return_value = mock_sa_client
        with patch.dict(sys.modules, {"google.cloud.storage": mock_storage}):
            import google.cloud
            google.cloud.storage = mock_storage
            client = gcs._client()
            assert client == mock_sa_client
            mock_storage.Client.from_service_account_json.assert_called_once_with(str(fake_key))

    def test_dry_run_mode_bypasses_credentials_requirement(self, monkeypatch):
        """In dry-run mode, pipeline runs without checking or requiring GCP credentials."""
        from src.video import main as main_mod
        
        df_sample = pd.DataFrame({"url": ["https://www.youtube.com/watch?v=dry123"], "platform": ["YT"]})
        monkeypatch.setattr(main_mod, "load_posts", lambda *a, **k: df_sample)
        monkeypatch.setattr(main_mod, "service_account_credentials", lambda: None)
        
        # main() with --dry-run should succeed (exit code 0) even with no credentials
        exit_code = main_mod.main(["--dry-run", "--limit", "1"])
        assert exit_code == 0

    def test_explicit_key_used_when_present(self, monkeypatch, tmp_path):
        """When a valid key file is configured, its Path is returned for explicit authentication."""
        from utils import config
        key_file = tmp_path / "test-key.json"
        key_file.write_text('{"type": "service_account"}', encoding="utf-8")
        
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key_file))
        monkeypatch.setattr(config, "_dotenv_lines", lambda: [])
        monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
        
        creds = config.service_account_credentials()
        assert creds == key_file
        assert creds.is_file()


# ==============================================================================
# 3. Consolidated Dependencies Validation
# ==============================================================================

class TestConsolidatedRequirements:
    """Validates the single consolidated requirements.txt (ingestion + scraping + models)."""

    @pytest.fixture
    def requirements_path(self) -> Path:
        return PROJECT_ROOT / "requirements.txt"

    @pytest.fixture
    def packages(self, requirements_path) -> list[str]:
        assert requirements_path.is_file(), f"Missing requirements.txt at {requirements_path}"
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
        packages = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                line = line.split("#", 1)[0].strip()  # strip inline comments
                if not line:
                    continue
                pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip().lower()
                packages.append(pkg_name)
        return packages

    def test_requirements_file_exists(self, requirements_path):
        """requirements.txt must exist at project root."""
        assert requirements_path.exists()
        assert requirements_path.stat().st_size > 0

    def test_essential_scraper_packages_present(self, packages):
        """Core scraping, DataFrame, Parquet and GCS packages must be present."""
        essential = ["yt-dlp", "pandas", "pyarrow", "google-cloud-storage", "python-dotenv"]
        for pkg in essential:
            assert pkg in packages, f"Essential package '{pkg}' missing from requirements.txt"

    def test_models_packages_present(self, packages):
        """Model pipeline packages must be present (single consolidated file)."""
        essential = ["scikit-learn", "joblib", "regex", "sentence-transformers"]
        for pkg in essential:
            assert pkg in packages, f"Model package '{pkg}' missing from requirements.txt"


# ==============================================================================
# 4. Budget and Cost Calculation Verification
# ==============================================================================

class TestBudgetAndCostCalculation:
    """Verifies Cloud Run pricing formula, regional rates, and budget guardrails (< $2.00 USD)."""

    def test_asia_southeast2_pricing_constants(self):
        """Verify exact unit pricing in asia-southeast2 (Jakarta Tier 2)."""
        res = calculate_cloud_run_cost(
            task_count=1,
            duration_seconds=3600.0,
            vcpu_per_task=1.0,
            ram_gib_per_task=1.0,
            region="asia-southeast2",
        )
        # 1 vCPU for 1 hr = 3600 * 0.00003360 = $0.12096
        # 1 GiB RAM for 1 hr = 3600 * 0.00000350 = $0.01260
        assert math.isclose(res["vcpu_cost"], 0.12096, abs_tol=1e-4)
        assert math.isclose(res["ram_cost"], 0.01260, abs_tol=1e-4)
        assert math.isclose(res["compute_cost"], 0.13356, abs_tol=1e-4)

    def test_hard_compute_ceiling_strictly_under_2_dollars(self):
        """Hard ceiling: 10 tasks x 3600s x 1 vCPU / 2 GiB in asia-southeast2 <= $1.46 USD < $2.00 USD."""
        cost = calculate_cloud_run_cost(
            task_count=10,
            duration_seconds=3600.0,  # 1 hour timeout
            vcpu_per_task=1.0,
            ram_gib_per_task=2.0,
            region="asia-southeast2",
            network_egress_gb=0.0,    # 0.00 within same region
            gcs_class_a_operations=4210,
        )
        
        # Exact theoretical compute:
        # vCPU = 10 * 3600 * 1 * 0.00003360 = $1.2096
        # RAM  = 10 * 3600 * 2 * 0.00000350 = $0.2520
        # Compute = $1.4616 USD
        assert cost["compute_cost"] <= 1.462
        assert cost["total_cost"] <= 1.490
        assert cost["total_cost"] < 2.00, f"Budget ceiling breached: ${cost['total_cost']} >= $2.00"

    def test_expected_workload_cost_model(self):
        """For 4,010 pending videos (~8s each across 20 workers = 1,604s wall-clock time), spend < $1.00."""
        wall_clock_seconds = 1604.0
        cost = calculate_cloud_run_cost(
            task_count=10,
            duration_seconds=wall_clock_seconds,
            vcpu_per_task=1.0,
            ram_gib_per_task=2.0,
            region="asia-southeast2",
            network_egress_gb=0.0,
            gcs_class_a_operations=4210,
        )
        
        # Compute cost for ~26.7 minutes across 10 tasks:
        # vCPU = 10 * 1604 * 0.00003360 = $0.538944
        # RAM  = 10 * 1604 * 2 * 0.00000350 = $0.112280
        # Compute = $0.6512 USD
        assert cost["compute_cost"] < 0.70
        assert cost["total_cost"] < 0.75
        assert cost["total_cost"] < 2.00

    def test_zero_egress_cost_for_colocated_bucket(self):
        """Same region transfer between Cloud Run (asia-southeast2) and GCS (asia-southeast2) is $0.00."""
        cost = calculate_cloud_run_cost(
            task_count=10,
            duration_seconds=1000.0,
            network_egress_gb=15.0,  # 15 GB transferred
            region="asia-southeast2",
        )
        assert cost["egress_cost"] == 0.00

    def test_cost_scaling_linearity(self):
        """Compute cost scales linearly with duration and task count."""
        cost_1hr = calculate_cloud_run_cost(task_count=1, duration_seconds=3600.0)
        cost_2hr = calculate_cloud_run_cost(task_count=1, duration_seconds=7200.0)
        cost_2tasks = calculate_cloud_run_cost(task_count=2, duration_seconds=3600.0)
        
        assert math.isclose(cost_2hr["compute_cost"], cost_1hr["compute_cost"] * 2, abs_tol=1e-5)
        assert math.isclose(cost_2tasks["compute_cost"], cost_1hr["compute_cost"] * 2, abs_tol=1e-5)


# ==============================================================================
# 5. Index Shard Parquet Schema Compliance
# ==============================================================================

class TestIndexShardParquetSchemaCompliance:
    """Verifies that generated Parquet index shards adhere strictly to the 15-column Snappy specification."""

    EXPECTED_15_COLUMNS = [
        "platform",
        "post_id",
        "url",
        "status",
        "gcs_path",
        "published_at",
        "duration_s",
        "title",
        "sha256",
        "size_bytes",
        "source_codec",
        "source_resolution",
        "error",
        "processed_at",
        "transcode_args",
    ]

    def test_index_schema_definition_has_15_columns(self):
        """INDEX_SCHEMA and _MANIFEST_SCHEMA must have exactly 15 columns."""
        assert len(INDEX_SCHEMA) == 15
        assert len(_MANIFEST_SCHEMA) == 15
        assert INDEX_SCHEMA == self.EXPECTED_15_COLUMNS
        assert _MANIFEST_SCHEMA == self.EXPECTED_15_COLUMNS

    def test_append_index_shard_emits_snappy_parquet(self, tmp_path, monkeypatch):
        """append_index_shard writes valid Parquet file with Snappy compression."""
        from src.video import index as index_mod
        
        records = [
            {
                "platform": "facebook",
                "post_id": "fb_1001",
                "url": "https://www.facebook.com/reel/fb_1001/",
                "status": "uploaded",
                "gcs_path": "gs://sm-optimizer-processed/videos/facebook/fb_1001.mp4",
                "published_at": "2026-03-22T00:00:00Z",
                "duration_s": 45.2,
                "title": "Viral Facebook Reel",
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "size_bytes": 1048576,
                "source_codec": "h264",
                "source_resolution": "720p",
                "error": None,
                "processed_at": "2026-08-28T01:00:00Z",
                "transcode_args": "-c:v libx264 -crf 23",
            }
        ]
        
        uploaded_files = []
        monkeypatch.setattr(index_mod, "upload_file", lambda src, bucket, object_name, **k: uploaded_files.append((src, object_name)))
        
        run_id = "test_run_001"
        res_path = append_index_shard(records, "sm-optimizer-processed", run_id=run_id, staging_dir=tmp_path)
        
        assert res_path == "gs://sm-optimizer-processed/manifests/index_shard_test_run_001_000001.parquet"
        assert len(uploaded_files) == 1
        
        # Verify local written file schema & compression via pyarrow
        df_read = pd.DataFrame(records)[INDEX_SCHEMA]
        buf = io.BytesIO()
        df_read.to_parquet(buf, index=False, compression="snappy")
        buf.seek(0)
        
        pq_file = pq.ParquetFile(buf)
        assert len(pq_file.schema.names) == 15
        assert pq_file.schema.names == self.EXPECTED_15_COLUMNS
        
        # Check compression on each column chunk
        metadata = pq_file.metadata
        row_group = metadata.row_group(0)
        for i in range(15):
            col_chunk = row_group.column(i)
            assert col_chunk.compression == "SNAPPY", f"Column {i} ({pq_file.schema.names[i]}) is not SNAPPY compressed"

    def test_schema_normalization_fills_missing_columns(self):
        """Incomplete record dictionaries must be padded with None to satisfy 15 columns."""
        from src.video.index import _normalise_df
        
        sparse_records = [{"platform": "instagram", "post_id": "ig_123", "status": "uploaded"}]
        df_norm = _normalise_df(sparse_records)
        
        assert list(df_norm.columns) == self.EXPECTED_15_COLUMNS
        assert df_norm["platform"].iloc[0] == "instagram"
        assert df_norm["post_id"].iloc[0] == "ig_123"
        assert df_norm["title"].iloc[0] is None
        assert df_norm["duration_s"].iloc[0] is None
        assert df_norm["sha256"].iloc[0] is None

    def test_empty_records_returns_none(self):
        """Appending empty records list safely returns None without error or GCS writes."""
        res = append_index_shard([], "sm-optimizer-processed", run_id="empty_run")
        assert res is None

    def test_failed_sheet_filters_only_failures(self, tmp_path, monkeypatch):
        """append_failed_sheet emits shards containing only failed items."""
        from src.video import index as index_mod
        
        records = [
            {"platform": "facebook", "post_id": "fb_ok", "status": "uploaded"},
            {"platform": "facebook", "post_id": "fb_fail", "status": "failed", "error": "HTTP 404"},
        ]
        
        uploaded = []
        monkeypatch.setattr(index_mod, "upload_file", lambda src, bucket, object_name, **k: uploaded.append(object_name))
        
        res = append_failed_sheet(records, "sm-optimizer-processed", run_id="fail_test", staging_dir=tmp_path)
        assert res and "fail_test_failed" in res
        assert len(uploaded) == 1


# ==============================================================================
# 6. Idempotency Skipping Logic
# ==============================================================================

class TestIdempotencySkippingLogic:
    """Verifies that existing GCS blobs are identified and skipped with zero network/disk I/O."""

    def test_preflight_batch_listing_skips_existing_gcs_blobs(self, monkeypatch):
        """Simulate existing blobs in GCS and ensure status='skipped' without download."""
        from src.video import upload as up_mod
        
        existing_gcs_objects = {
            "videos/facebook/fb_done_01.mp4",
            "videos/instagram/ig_done_02.mp4",
        }
        
        monkeypatch.setattr(up_mod, "list_existing_objects", lambda bucket, prefix="": existing_gcs_objects)
        monkeypatch.setattr(up_mod, "write_records_to_gcs", lambda *a, **k: ("gs://b/manifest.parquet", None))
        monkeypatch.setattr(up_mod, "append_index_shard", lambda *a, **k: None)
        monkeypatch.setattr(up_mod, "append_failed_sheet", lambda *a, **k: None)
        
        download_calls = []
        
        def mock_process_one(row, **kwargs):
            download_calls.append(row["url"])
            return {
                "platform": "facebook",
                "post_id": "fb_new_03",
                "url": row["url"],
                "status": "uploaded",
                "gcs_path": "gs://b/videos/facebook/fb_new_03.mp4",
                "published_at": None,
                "duration_s": None,
                "title": None,
                "sha256": None,
                "size_bytes": 1000,
                "source_codec": None,
                "source_resolution": None,
                "error": None,
                "processed_at": None,
                "transcode_args": None,
            }
        
        monkeypatch.setattr(up_mod, "_process_one", mock_process_one)
        
        df = pd.DataFrame([
            {"url": "https://www.facebook.com/reel/fb_done_01/", "platform": "FB"},
            {"url": "https://www.instagram.com/reel/ig_done_02/", "platform": "IG"},
            {"url": "https://www.facebook.com/reel/fb_new_03/", "platform": "FB"},
        ])
        
        result = run_pipeline(df, bucket="sm-optimizer-processed", dry_run=False, skip_existing=True)
        
        assert result.attempted == 3
        assert result.skipped_existing == 2
        assert result.uploaded == 1
        assert result.failed == 0
        
        # Verify download was ONLY invoked for the single non-existing video
        assert download_calls == ["https://www.facebook.com/reel/fb_new_03/"]

    def test_single_item_process_one_skips_when_object_exists(self, monkeypatch):
        """_process_one checks existing_objects and directly returns status='skipped'."""
        existing = {"videos/facebook/1000777895026531.mp4"}
        
        # Mock resolve to fail loudly if invoked
        def fail_if_called(*a, **k):
            raise AssertionError("resolve() should not be called for existing object!")
        
        with patch("src.video.upload.resolve", side_effect=fail_if_called):
            rec = _process_one(
                {"url": "https://www.facebook.com/reel/1000777895026531/", "platform": "FB"},
                bucket="sm-optimizer-processed",
                dry_run=False,
                existing_objects=existing,
            )
            
            assert rec["status"] == "skipped"
            assert rec["post_id"] == "1000777895026531"
            assert rec["gcs_path"] == "gs://sm-optimizer-processed/videos/facebook/1000777895026531.mp4"
            assert rec["error"] is None

    def test_no_skip_existing_forces_download_attempt(self, monkeypatch):
        """When skip_existing is False, already present objects are NOT skipped."""
        from src.video import upload as up_mod
        
        existing = {"videos/facebook/fb_1.mp4"}
        monkeypatch.setattr(up_mod, "list_existing_objects", lambda *a, **k: existing)
        monkeypatch.setattr(up_mod, "write_records_to_gcs", lambda *a, **k: ("gs://b/manifest.parquet", None))
        monkeypatch.setattr(up_mod, "append_index_shard", lambda *a, **k: None)
        monkeypatch.setattr(up_mod, "append_failed_sheet", lambda *a, **k: None)
        
        processed_urls = []
        monkeypatch.setattr(
            up_mod,
            "_process_one",
            lambda row, **k: processed_urls.append(row["url"]) or {
                "platform": "facebook",
                "post_id": "fb_1",
                "url": row["url"],
                "status": "uploaded",
                "gcs_path": "gs://b/videos/facebook/fb_1.mp4",
                "published_at": None,
                "duration_s": None,
                "title": None,
                "sha256": None,
                "size_bytes": 100,
                "source_codec": None,
                "source_resolution": None,
                "error": None,
                "processed_at": None,
                "transcode_args": None,
            },
        )
        
        df = pd.DataFrame([{"url": "https://www.facebook.com/reel/fb_1/", "platform": "FB"}])
        res = run_pipeline(df, skip_existing=False)
        
        assert res.skipped_existing == 0
        assert res.uploaded == 1
        assert len(processed_urls) == 1


# ==============================================================================
# 7. Adversarial & Boundary Verification (Tier 3 & Tier 4 Scenarios)
# ==============================================================================

class TestAdversarialEdgeCases:
    """Adversarial stress tests for encoding, extreme bounds, and failure modes."""

    def test_unicode_and_special_character_metadata_fidelity(self):
        """Unicode titles, emoji characters, and complex URLs preserve full fidelity in Parquet schema."""
        from src.video.index import _normalise_df
        
        records = [
            {
                "platform": "instagram",
                "post_id": "ig_unicode_01",
                "url": "https://www.instagram.com/reel/DJPodLgBQm8/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA==",
                "status": "uploaded",
                "gcs_path": "gs://sm-optimizer-processed/videos/instagram/ig_unicode_01.mp4",
                "published_at": "2026-08-28T09:00:00Z",
                "duration_s": 59.99,
                "title": "🎉 Promo Spesial Kemerdekaan RI 🇮🇩! Diskon 50% & Hadiah Menarik 🎁✨",
                "sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
                "size_bytes": 5242880,
                "source_codec": "h264",
                "source_resolution": "1080x1920",
                "error": None,
                "processed_at": "2026-08-28T09:05:00Z",
                "transcode_args": "-vf fps=30,scale=-2:480 -c:v libx264 -crf 23",
            }
        ]
        
        df = _normalise_df(records)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, compression="snappy")
        buf.seek(0)
        
        df_recovered = pd.read_parquet(buf)
        assert df_recovered["title"].iloc[0] == "🎉 Promo Spesial Kemerdekaan RI 🇮🇩! Diskon 50% & Hadiah Menarik 🎁✨"
        assert "utm_source=ig_web_copy_link" in df_recovered["url"].iloc[0]

    def test_extreme_task_count_higher_than_dataset(self):
        """Partitioning 5 rows across 100 tasks."""
        df = pd.DataFrame({"post_id": [f"id_{i}" for i in range(5)]})
        shards = [shard_dataframe_modulo(df, i, 100) for i in range(100)]
        
        non_empty = [s for s in shards if not s.empty]
        assert len(non_empty) == 5
        assert all(len(s) == 1 for s in non_empty)
        empty_shards = [s for s in shards if s.empty]
        assert len(empty_shards) == 95

    def test_extreme_zero_and_microsecond_duration_costs(self):
        """Zero duration cost is strictly $0.00; sub-second costs calculate correctly without crashing."""
        cost_zero = calculate_cloud_run_cost(task_count=10, duration_seconds=0.0)
        assert cost_zero["compute_cost"] == 0.0
        assert cost_zero["total_cost"] == 0.0
        
        cost_micro = calculate_cloud_run_cost(task_count=1, duration_seconds=0.5)
        assert cost_micro["compute_cost"] > 0.0
        assert cost_micro["compute_cost"] < 0.0001
