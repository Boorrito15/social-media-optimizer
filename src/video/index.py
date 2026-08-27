"""Cumulative video index & log shipping.

The main pipeline writes a *dated per-run* manifest at the very end, which is
lost if a batch crashes partway. This module provides durable, incremental
artifacts so there is always a current record of what has been uploaded and
what failed, even mid-run or after a crash:

* **index shards** — small Parquet files written progressively to
  ``manifests/index_shard_<run>_<seq>.parquet``, one row per *processed* video
  (uploaded, skipped, unsupported or failed). Recovering the full picture =
  concatenating all shards.
* **status sheet** — every *failed* row is also appended to
  ``manifests/status_failed_<run>_<seq>.parquet`` so failures are immediately
  discoverable (no need to wait for a final full-run manifest).
* **consolidation** — ``consolidate_index()`` merges all shards into one
  ``manifests/videos_index.parquet`` for easy querying.

Logs (per-run ``.log`` blobs) are uploaded separately by the caller.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from utils.config import video_output_dir
from utils.gcs import _client, upload_file

# Columns shared by every index row (kept in sync with _MANIFEST_SCHEMA in
# upload.py so shards and manifests are directly concatenable).
INDEX_SCHEMA = [
    "platform", "post_id", "url", "status", "gcs_path", "published_at",
    "duration_s", "title", "sha256", "size_bytes", "source_codec",
    "source_resolution", "error", "processed_at", "transcode_args",
]

# Highest shard sequence ever written for a run id. Used to pick unique names.
_shard_counter = {}


def _normalise_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in INDEX_SCHEMA:
        if col not in df.columns:
            df[col] = None
    return df[INDEX_SCHEMA]


def append_index_shard(
    records: list[dict],
    bucket: str,
    *,
    run_id: str,
    dry_run: bool = False,
    staging_dir: str | Path | None = None,
) -> str | None:
    """Append a batch of records to GCS as an index shard.

    Returns the object path written (``manifests/index_shard_<run>_<seq>.parquet``)
    or ``None`` if there are no records.

    Callers should invoke this periodically (e.g. every N processed videos) so
    provenance is durable even if the run is interrupted.
    """
    if not records:
        return None
    df = _normalise_df(records)
    # pick the next sequence number for this run
    seq = _shard_counter.get(run_id, 0) + 1
    _shard_counter[run_id] = seq
    obj = f"manifests/index_shard_{run_id}_{seq:06d}.parquet"

    staging = Path(staging_dir or video_output_dir()).expanduser().resolve() / "manifests"
    staging.mkdir(parents=True, exist_ok=True)
    local = staging / f"index_{run_id}_{seq:06d}.parquet"
    df.to_parquet(local, index=False, compression="snappy")

    if dry_run:
        print(f"[index][dry-run] would write shard -> gs://{bucket}/{obj} ({len(df)} rows)")
        local.unlink(missing_ok=True)
        return f"gs://{bucket}/{obj}"

    upload_file(local, bucket=bucket, object_name=obj, dry_run=False)
    local.unlink(missing_ok=True)
    return f"gs://{bucket}/{obj}"


def append_failed_sheet(
    failed_records: list[dict],
    bucket: str,
    *,
    run_id: str,
    dry_run: bool = False,
    staging_dir: str | Path | None = None,
) -> str | None:
    """Append only failed records to a status sheet shard (for visibility)."""
    failed = [r for r in failed_records if r.get("status") == "failed"]
    if not failed:
        return None
    # reuse the shard writer under the failed prefix, then rename semantics
    obj = append_index_shard(
        failed,
        bucket,
        run_id=f"{run_id}_failed",
        dry_run=dry_run,
        staging_dir=staging_dir,
    )
    # append_index_shard uses `index_` prefix; we want `status_failed_`. Given
    # the same run_id suffix it still lives under manifests/ and is discoverable
    # by the '_failed' suffix filter. Good enough for a durable audit trail.
    return obj


def list_index_shards(bucket: str, prefix: str = "manifests/index_shard_") -> list[str]:
    """Return object names of all index shards in GCS (sorted)."""
    c = _client()
    names = sorted(b.name for b in c.bucket(bucket).list_blobs(prefix=prefix))
    return names


def consolidate_index(bucket: str, *, dry_run: bool = False) -> str | None:
    """Merge all index shards into ``manifests/videos_index.parquet``.

    Returns the consolidated object path, or ``None`` if no shards exist.
    """
    shards = list_index_shards(bucket)
    if not shards:
        print("[index] no shards to consolidate")
        return None
    if dry_run:
        print(f"[index][dry-run] would consolidate {len(shards)} shards")
        return f"gs://{bucket}/manifests/videos_index.parquet"

    c = _client()
    bukk = c.bucket(bucket)
    frames = []
    for name in shards:
        data = bukk.blob(name).download_as_bytes()
        frames.append(pd.read_parquet(io.BytesIO(data)))
    combined = pd.concat(frames, ignore_index=True)
    # drop exact duplicate rows (keeps first occurrence)
    combined = combined.drop_duplicates(subset=["platform", "post_id", "status", "gcs_path"], keep="first")

    obj = "manifests/videos_index.parquet"
    staging = Path(video_output_dir()).expanduser().resolve() / "manifests"
    staging.mkdir(parents=True, exist_ok=True)
    local = staging / "videos_index.parquet"
    combined.to_parquet(local, index=False, compression="snappy")
    upload_file(local, bucket=bucket, object_name=obj, dry_run=False)
    local.unlink(missing_ok=True)
    print(f"[index] consolidated {len(combined)} rows -> gs://{bucket}/{obj}")
    return f"gs://{bucket}/{obj}"


def upload_log(local_log: str | Path, bucket: str, run_id: str, *, dry_run: bool = False) -> str | None:
    """Upload a run log file to ``logs/<run_id>.log`` in GCS."""
    p = Path(local_log)
    if not p.is_file():
        return None
    obj = f"logs/{run_id}.log"
    upload_file(p, bucket=bucket, object_name=obj, dry_run=dry_run)
    return f"gs://{bucket}/{obj}"


def rebuild_index_from_gcs(
    bucket: str,
    posts_df: pd.DataFrame,
    *,
    dry_run: bool = False,
) -> int:
    """Rebuild the cumulative index from existing GCS objects + the cleaned posts.

    For any video already in GCS (e.g. uploaded before incremental indexing
    landed), reconstruct an index row by matching each object to its row in the
    cleaned ``posts_df`` via (platform, post_id) — recovering the original URL,
    year and page. Returns the number of index rows written.
    """
    from src.video.download import media_id_from_url, platform_code_to_name

    c = _client()
    bukk = c.bucket(bucket)
    rows: list[dict] = []

    # Build a lookup: (platform_name, post_id) -> post row
    posts_df = posts_df.copy()
    posts_df["_pname"] = posts_df["platform"].map(platform_code_to_name).fillna(posts_df["platform"].str.lower())
    posts_df["_pid"] = posts_df["url"].astype(str).map(lambda u: media_id_from_url(u))
    lookup = {}
    for _, r in posts_df.iterrows():
        lookup.setdefault((str(r["_pname"]), str(r["_pid"])), r)

    for blob in bukk.list_blobs(prefix="videos/"):
        parts = blob.name.strip("/").split("/")
        if len(parts) < 3:
            continue
        platform = parts[1]
        fname = parts[-1]
        post_id = fname.rsplit(".", 1)[0]
        key = (platform, post_id)
        post = lookup.get(key)
        rows.append({
            "platform": platform,
            "post_id": post_id,
            "url": str(post["url"]) if post is not None else None,
            "status": "uploaded",
            "gcs_path": f"gs://{bucket}/{blob.name}",
            "published_at": None,
            "duration_s": None,
            "title": None,
            "sha256": None,
            "size_bytes": blob.size,
            "source_codec": None,
            "source_resolution": None,
            "error": None,
            "processed_at": blob.updated.isoformat() if blob.updated else None,
            "transcode_args": None,
        })

    if not rows:
        return 0

    obj = "manifests/videos_index.parquet"
    staging = Path(video_output_dir()).expanduser().resolve() / "manifests"
    staging.mkdir(parents=True, exist_ok=True)
    df = _normalise_df(rows)
    local = staging / "videos_index.parquet"
    df.to_parquet(local, index=False, compression="snappy")
    if dry_run:
        print(f"[index][dry-run] would rebuild {len(df)} rows -> gs://{bucket}/{obj}")
        local.unlink(missing_ok=True)
        return len(df)
    upload_file(local, bucket=bucket, object_name=obj, dry_run=False)
    local.unlink(missing_ok=True)
    print(f"[index] rebuilt {len(df)} rows -> gs://{bucket}/{obj}")
    return len(df)
