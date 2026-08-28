"""Milestone 1 Challenger 2 Empirical Stress Test & Adversarial Verification Suite.

Verification Scope:
1. Environment Variable Injection (CLOUD_RUN_TASK_INDEX & CLOUD_RUN_TASK_COUNT) simulating Cloud Run container execution.
2. Budget Formula Robustness against Parameter Variations (Task count scaling, timeout limits, vCPU/RAM sizing, region pricing).
3. Scraper Dependency Resolution & Clean Pip Installability (requirements-scraper.txt).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.video.main import parse_args, main
from src.video.upload import load_posts


# =====================================================================
# 1. ENVIRONMENT VARIABLE INJECTION & CONTAINER EXECUTION TESTS
# =====================================================================

class TestEnvVarInjectionAndSharding:
    """Empirical verification of CLOUD_RUN_TASK_INDEX and CLOUD_RUN_TASK_COUNT."""

    def test_default_env_vars_when_unset(self, monkeypatch):
        """When env vars are unset, defaults should be index=0, count=1."""
        monkeypatch.delenv("CLOUD_RUN_TASK_INDEX", raising=False)
        monkeypatch.delenv("CLOUD_RUN_TASK_COUNT", raising=False)

        args = parse_args([])
        assert args.task_index == 0
        assert args.task_count == 1

    def test_env_var_injection_parsing(self, monkeypatch):
        """When env vars are set by Cloud Run runtime, CLI parser picks them up."""
        monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "4")
        monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "10")

        args = parse_args([])
        assert args.task_index == 4
        assert args.task_count == 10

    def test_cli_explicit_override_precedence_over_env_vars(self, monkeypatch):
        """Explicit CLI arguments must override environment variables."""
        monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "2")
        monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "8")

        args = parse_args(["--task-index", "7", "--task-count", "12"])
        assert args.task_index == 7
        assert args.task_count == 12

    def test_partition_completeness_and_disjointness_on_real_data(self):
        """Verify that partitioning the real dataset across N tasks is 100% complete and disjoint."""
        clean_parquet_path = Path("data/processed/posts_clean.parquet")
        if not clean_parquet_path.exists():
            pytest.skip("data/processed/posts_clean.parquet not found")

        df = load_posts(str(clean_parquet_path))
        total_rows = len(df)
        assert total_rows > 0

        for task_count in [2, 5, 10, 13]:  # Test standard, even, and prime partition counts
            shards = []
            shard_indices = []

            for task_index in range(task_count):
                shard = df[df.reset_index().index % task_count == task_index]
                shards.append(shard)
                shard_indices.append(set(shard.index))

            # Completeness: Sum of shard sizes must equal total dataset size
            total_partitioned = sum(len(s) for s in shards)
            assert total_partitioned == total_rows, f"Failed for task_count={task_count}: {total_partitioned} != {total_rows}"

            # Disjointness: No two shards can share any row index
            for i in range(task_count):
                for j in range(i + 1, task_count):
                    intersection = shard_indices[i].intersection(shard_indices[j])
                    assert len(intersection) == 0, f"Overlap between shard {i} and {j}: {len(intersection)} rows"

    def test_partition_balance_across_platforms(self):
        """Verify that modulo sharding distributes platform workloads evenly."""
        clean_parquet_path = Path("data/processed/posts_clean.parquet")
        if not clean_parquet_path.exists():
            pytest.skip("data/processed/posts_clean.parquet not found")

        df = load_posts(str(clean_parquet_path))
        meta_df = df[df["platform"].str.upper().isin(["FB", "IG", "FACEBOOK", "INSTAGRAM"])]
        
        task_count = 10
        meta_counts_per_task = []
        for task_index in range(task_count):
            shard = df[df.reset_index().index % task_count == task_index]
            meta_in_shard = shard[shard["platform"].str.upper().isin(["FB", "IG", "FACEBOOK", "INSTAGRAM"])]
            meta_counts_per_task.append(len(meta_in_shard))

        avg_meta = np.mean(meta_counts_per_task)
        max_deviation = np.max(np.abs(meta_counts_per_task - avg_meta))
        
        # Max deviation should be very small (< 10% of mean) indicating great load balance
        assert max_deviation / avg_meta < 0.10, f"Uneven load distribution: {meta_counts_per_task}"

    def test_out_of_bounds_task_index_handled_gracefully(self, capsys):
        """Task index >= task count or negative index must be rejected with exit code 1."""
        ret_negative = main(["--task-index", "-1", "--task-count", "10", "--dry-run"])
        assert ret_negative == 1
        captured = capsys.readouterr()
        assert "[ERROR] Invalid --task-index -1" in captured.out

        ret_oob = main(["--task-index", "10", "--task-count", "10", "--dry-run"])
        assert ret_oob == 1
        captured = capsys.readouterr()
        assert "[ERROR] Invalid --task-index 10" in captured.out

    def test_dataframe_index_invariance(self):
        """Modulo sharding via reset_index().index must work regardless of underlying DataFrame index type."""
        # Case 1: Custom String Index
        df_str = pd.DataFrame({"val": range(100)}, index=[f"id_{i}" for i in range(100)])
        shard_str = df_str[df_str.reset_index().index % 5 == 2]
        assert len(shard_str) == 20
        assert list(shard_str["val"]) == list(range(2, 100, 5))

        # Case 2: Non-contiguous / filtered Index
        df_filtered = pd.DataFrame({"val": range(100)}).iloc[10:60] # length 50, index 10..59
        shard_filtered = df_filtered[df_filtered.reset_index().index % 5 == 1]
        assert len(shard_filtered) == 10
        assert shard_filtered.iloc[0]["val"] == 11

    def test_full_subprocess_container_dry_run_execution(self):
        """Simulate Cloud Run container execution by launching python -m src.video.main in subprocess."""
        env = os.environ.copy()
        env["CLOUD_RUN_TASK_INDEX"] = "3"
        env["CLOUD_RUN_TASK_COUNT"] = "10"

        res = subprocess.run(
            [sys.executable, "-m", "src.video.main", "--dry-run", "--limit", "5"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parent.parent)
        )
        assert res.returncode == 0, f"Subprocess failed with stderr:\n{res.stderr}"
        assert "Task shard 3/10" in res.stdout


# =====================================================================
# 2. BUDGET FORMULA ROBUSTNESS & PARAMETER VARIATION TESTS
# =====================================================================

class TestBudgetRobustness:
    """Empirical calculations and parameter stress-testing of Cloud Run Job compute cost."""

    # GCP Cloud Run v2 Job Pricing (asia-southeast2 / Jakarta - Tier 2)
    VCPU_PRICE_TIER2 = 0.00003360  # per vCPU-second
    RAM_PRICE_TIER2 = 0.00000350   # per GiB-second

    # GCP Cloud Run v2 Job Pricing (us-central1 - Tier 1)
    VCPU_PRICE_TIER1 = 0.00002400  # per vCPU-second
    RAM_PRICE_TIER1 = 0.00000250   # per GiB-second

    @classmethod
    def calculate_cost(
        cls,
        task_count: int,
        duration_seconds: float,
        vcpu: float = 1.0,
        ram_gib: float = 2.0,
        tier: int = 2,
    ) -> float:
        """Calculate total gross Cloud Run job compute spend."""
        vcpu_rate = cls.VCPU_PRICE_TIER2 if tier == 2 else cls.VCPU_PRICE_TIER1
        ram_rate = cls.RAM_PRICE_TIER2 if tier == 2 else cls.RAM_PRICE_TIER1

        vcpu_cost = task_count * vcpu * duration_seconds * vcpu_rate
        ram_cost = task_count * ram_gib * duration_seconds * ram_rate
        return vcpu_cost + ram_cost

    def test_m1_baseline_budget_hard_ceiling(self):
        """Hard ceiling: 10 tasks x 1 vCPU x 2 GiB x 3600s must strictly remain under $2.00 USD."""
        max_duration = 3600  # 1 hour timeout
        task_count = 10
        vcpu = 1.0
        ram_gib = 2.0

        max_spend_tier2 = self.calculate_cost(task_count, max_duration, vcpu, ram_gib, tier=2)
        max_spend_tier1 = self.calculate_cost(task_count, max_duration, vcpu, ram_gib, tier=1)

        # In asia-southeast2 (Tier 2):
        # 10 * 1 * 3600 * 0.00003360 = $1.2096
        # 10 * 2 * 3600 * 0.00000350 = $0.2520
        # Total = $1.4616 USD
        assert max_spend_tier2 < 2.00, f"Tier 2 max spend exceeded: ${max_spend_tier2:.4f}"
        assert max_spend_tier1 < 2.00, f"Tier 1 max spend exceeded: ${max_spend_tier1:.4f}"
        assert abs(max_spend_tier2 - 1.4616) < 0.001

    def test_realistic_ingestion_spend_estimate(self):
        """Realistic scenario: 4,010 pending videos with 20 total concurrency."""
        pending_videos = 4010
        total_workers = 20  # 10 tasks * 2 workers/task
        avg_seconds_per_video = 3.5  # metadata + download + 480p transcode + upload

        total_task_runtime_s = (pending_videos / total_workers) * avg_seconds_per_video
        expected_spend = self.calculate_cost(
            task_count=10,
            duration_seconds=total_task_runtime_s,
            vcpu=1.0,
            ram_gib=2.0,
            tier=2,
        )
        # Expected duration is ~700 seconds (~11.7 minutes) -> ~$0.285 USD
        assert expected_spend < 0.60, f"Realistic spend higher than expected: ${expected_spend:.4f}"
        assert expected_spend < 2.00

    def test_parameter_variation_safety_matrix(self):
        """Evaluate grid of task count, timeouts, and hardware sizing to map budget boundaries."""
        results = []
        for tasks in [1, 5, 10, 12, 15, 20]:
            for timeout in [600, 1800, 3600, 5400, 7200]:
                for ram in [1.0, 2.0, 4.0]:
                    cost = self.calculate_cost(tasks, timeout, vcpu=1.0, ram_gib=ram, tier=2)
                    results.append({
                        "tasks": tasks,
                        "timeout": timeout,
                        "ram": ram,
                        "cost": cost,
                        "within_budget": cost < 2.00,
                    })

        df_results = pd.DataFrame(results)

        # 10 tasks at 3600s with 2GB is strictly True
        baseline_row = df_results[
            (df_results["tasks"] == 10) &
            (df_results["timeout"] == 3600) &
            (df_results["ram"] == 2.0)
        ].iloc[0]
        assert baseline_row["within_budget"] == True

        # Document exact tipping points:
        # At 10 tasks and 2GB RAM, max timeout allowed before hitting $2.00:
        # 2.00 / (10 * (1 * 0.00003360 + 2 * 0.00000350)) = 2.00 / 0.000406 = 4,926.1 seconds (~82 mins)
        max_safe_timeout = 2.00 / (10 * (1 * self.VCPU_PRICE_TIER2 + 2 * self.RAM_PRICE_TIER2))
        assert max_safe_timeout > 3600, f"Baseline 3600s timeout is unsafe: max={max_safe_timeout:.1f}s"
        assert max_safe_timeout < 5000, f"Unexpected safety boundary: max={max_safe_timeout:.1f}s"

    def test_max_parallelism_scaling_within_1_hour(self):
        """At 3600s timeout, what is the maximum number of parallel 1 vCPU / 2 GiB tasks permitted under $2.00?"""
        cost_per_task_per_hour = (1 * self.VCPU_PRICE_TIER2 + 2 * self.RAM_PRICE_TIER2) * 3600
        max_tasks = int(2.00 // cost_per_task_per_hour)
        
        # cost_per_task_per_hour = $0.14616 -> max tasks = 13 tasks (13 * 0.14616 = $1.900)
        assert max_tasks == 13, f"Expected 13 max tasks, got {max_tasks}"
        assert self.calculate_cost(10, 3600) < 2.00
        assert self.calculate_cost(13, 3600) < 2.00
        assert self.calculate_cost(14, 3600) > 2.00  # 14 tasks exceeds $2.00 at full 1h


# =====================================================================
# 3. REQUIREMENTS & DEPENDENCY ISOLATION TESTS
# =====================================================================

class TestDependencyResolution:
    """Empirical verification of requirements-scraper.txt."""

    def test_requirements_file_exists_and_is_non_empty(self):
        req_file = Path("requirements-scraper.txt")
        assert req_file.exists()
        assert req_file.stat().st_size > 0

    def test_heavy_ml_packages_omitted(self):
        """Verify heavy ML libraries are excluded from scraper requirements to maintain lightweight image."""
        req_content = Path("requirements-scraper.txt").read_text().lower()
        heavy_packages = [
            "torch", "torchvision", "torchaudio", "transformers",
            "scikit-learn", "sklearn", "tensorflow", "keras",
            "matplotlib", "seaborn", "scipy", "plotly"
        ]
        for pkg in heavy_packages:
            assert pkg not in req_content, f"Heavy dependency '{pkg}' found in requirements-scraper.txt"

    def test_essential_scraper_packages_present(self):
        """Verify all core dependencies for GCS, yt-dlp, and Parquet are present."""
        req_lines = [
            line.strip().split("==")[0].split(">=")[0].strip().lower()
            for line in Path("requirements-scraper.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        required_cores = ["yt-dlp", "pandas", "pyarrow", "google-cloud-storage", "python-dotenv"]
        for core in required_cores:
            assert core in req_lines, f"Essential dependency '{core}' missing from requirements-scraper.txt"
