"""Empirical Stress Testing & Verification Suite for Milestone 1 Task Sharding & CLI.

Verifies:
1. Deterministic task sharding across extreme task counts (1, 2, 5, 10, 100, 1000) on data/processed/posts_clean.parquet:
   - 0 duplicate rows across all shards (pairwise disjointness).
   - 0 dropped rows (union of all shards == full dataset).
   - Balanced distribution of Facebook and Instagram posts across all shards (max diff <= 1).
2. CLI execution of src/video/main.py with various --task-index, --task-count, --dry-run, and env vars.
3. Robust error handling on invalid task configurations.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "posts_clean.parquet"


@pytest.fixture(scope="module")
def full_dataset() -> pd.DataFrame:
    assert DATA_PATH.is_file(), f"Dataset not found at {DATA_PATH}"
    df = pd.read_parquet(DATA_PATH)
    assert len(df) > 0
    return df


@pytest.mark.parametrize("task_count", [1, 2, 5, 10, 100, 1000])
def test_empirical_sharding_completeness_and_disjointness(full_dataset: pd.DataFrame, task_count: int):
    """Verify 0 duplicate rows and 0 dropped rows across all shards."""
    total_len = len(full_dataset)
    shards: list[pd.DataFrame] = []

    for task_index in range(task_count):
        shard = full_dataset[full_dataset.reset_index().index % task_count == task_index]
        shards.append(shard)

    # 1. Total row count across all shards equals full dataset
    total_shard_rows = sum(len(s) for s in shards)
    assert total_shard_rows == total_len, f"Total rows ({total_shard_rows}) != dataset size ({total_len})"

    # 2. Pairwise disjointness (0 duplicate indices)
    all_indices = []
    for s in shards:
        all_indices.extend(s.index.tolist())
    assert len(all_indices) == len(set(all_indices)), f"Found duplicate row indices in task_count={task_count}"

    # 3. Exact reconstruction equals full dataset
    reconstructed = pd.concat(shards).sort_index()
    pd.testing.assert_frame_equal(reconstructed, full_dataset)

    # 4. Uniform partition size distribution: max diff <= 1
    shard_lens = [len(s) for s in shards]
    assert max(shard_lens) - min(shard_lens) <= 1, (
        f"Imbalanced shard sizes for task_count={task_count}: min={min(shard_lens)}, max={max(shard_lens)}"
    )


@pytest.mark.parametrize("task_count", [1, 2, 5, 10, 100, 1000])
def test_empirical_sharding_meta_platform_balance(full_dataset: pd.DataFrame, task_count: int):
    """Verify balanced distribution of Facebook and Instagram posts across all shards."""
    shards: list[pd.DataFrame] = []
    for task_index in range(task_count):
        shard = full_dataset[full_dataset.reset_index().index % task_count == task_index]
        shards.append(shard)

    fb_counts = [len(s[s["platform"] == "FB"]) for s in shards]
    ig_counts = [len(s[s["platform"] == "IG"]) for s in shards]
    meta_counts = [fb + ig for fb, ig in zip(fb_counts, ig_counts)]

    # FB counts across shards must differ by at most 1
    assert max(fb_counts) - min(fb_counts) <= 1, (
        f"Imbalanced FB distribution for task_count={task_count}: min={min(fb_counts)}, max={max(fb_counts)}"
    )

    # IG counts across shards must differ by at most 1
    assert max(ig_counts) - min(ig_counts) <= 1, (
        f"Imbalanced IG distribution for task_count={task_count}: min={min(ig_counts)}, max={max(ig_counts)}"
    )

    # Total Meta posts preserved
    assert sum(fb_counts) == len(full_dataset[full_dataset["platform"] == "FB"])
    assert sum(ig_counts) == len(full_dataset[full_dataset["platform"] == "IG"])
    assert sum(meta_counts) == len(full_dataset[full_dataset["platform"].isin(["FB", "IG"])])


def test_cli_execution_sharding_dry_run():
    """Verify CLI execution of main.py with --task-index and --task-count in --dry-run mode."""
    cmd = [
        sys.executable,
        "-m",
        "src.video.main",
        "--task-index",
        "0",
        "--task-count",
        "10",
        "--dry-run",
        "--limit",
        "5",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Task shard 0/10: assigned 1,315 of 13,142 posts" in proc.stdout
    assert "Video pipeline summary:" in proc.stdout
    assert "Attempted:        5" in proc.stdout
    assert "Uploaded:         5" in proc.stdout
    assert "Failed:           0" in proc.stdout


def test_cli_execution_last_task_shard():
    """Verify CLI execution for last shard index (9 of 10)."""
    cmd = [
        sys.executable,
        "-m",
        "src.video.main",
        "--task-index",
        "9",
        "--task-count",
        "10",
        "--dry-run",
        "--limit",
        "5",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Task shard 9/10: assigned 1,314 of 13,142 posts" in proc.stdout
    assert "Attempted:        5" in proc.stdout


def test_cli_execution_env_var_fallback():
    """Verify CLI execution reads CLOUD_RUN_TASK_INDEX and CLOUD_RUN_TASK_COUNT."""
    env = os.environ.copy()
    env["CLOUD_RUN_TASK_INDEX"] = "3"
    env["CLOUD_RUN_TASK_COUNT"] = "5"
    cmd = [
        sys.executable,
        "-m",
        "src.video.main",
        "--dry-run",
        "--limit",
        "3",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    assert "Task shard 3/5: assigned 2,628 of 13,142 posts" in proc.stdout
    assert "Attempted:        3" in proc.stdout


def test_cli_execution_invalid_task_index_rejected():
    """Verify CLI exits with code 1 when --task-index is out of range."""
    cmd_neg = [
        sys.executable,
        "-m",
        "src.video.main",
        "--task-index",
        "-1",
        "--task-count",
        "10",
        "--dry-run",
    ]
    proc_neg = subprocess.run(cmd_neg, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert proc_neg.returncode == 1
    assert "[ERROR] Invalid --task-index -1 for --task-count 10" in proc_neg.stdout

    cmd_eq = [
        sys.executable,
        "-m",
        "src.video.main",
        "--task-index",
        "10",
        "--task-count",
        "10",
        "--dry-run",
    ]
    proc_eq = subprocess.run(cmd_eq, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert proc_eq.returncode == 1
    assert "[ERROR] Invalid --task-index 10 for --task-count 10" in proc_eq.stdout
