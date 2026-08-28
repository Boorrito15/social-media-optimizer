"""Orchestrates video scraping, transcoding, GCS upload and manifest writing.

Flow:
    1. Read ``data/processed/posts_clean.parquet`` (or CSV).
    2. Filter to supported video platforms (or user-specified ``--platforms``).
    3. Concurrently or serially:
        a. Resolve media metadata via yt-dlp.
        b. Download source to ``data/videos/<platform>/<id>.<ext>``.
        c. Transcode to standard 480p H.264 / AAC (unless ``--no-transcode``).
        d. Upload to ``gs://<bucket>/videos/<platform>/<id>.mp4``.
        e. Periodically append to an index shard in GCS so long runs leave a
           durable trail if interrupted.
    4. Write a consolidated manifest Parquet file to ``gs://<bucket>/manifests/``.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.config import (
    gcs_processed_bucket,
    resolve_path,
    video_concurrency,
    video_gcs_prefix,
    video_output_dir,
)
from utils.gcs import list_existing_objects, object_exists, sha256_file, upload_file
from src.video.download import (
    SUPPORTED_EXTRACTORS,
    MediaResolutionError,
    ResolvedMedia,
    download,
    media_id_from_url,
    platform_code_to_name,
    published_at_from_info,
    resolve,
)
from src.video.index import append_failed_sheet, append_index_shard


# Target schema for both the consolidated manifest and index shards.
_MANIFEST_SCHEMA = [
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


@dataclass
class VideoJobResult:
    """Summary of a video scraping / transcode / upload run."""

    attempted: int = 0
    uploaded: int = 0
    skipped_existing: int = 0
    unsupported: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    status_sheet_path: str | None = None
    index_shards: list[str] = field(default_factory=list)
    run_id: str | None = None

    def __str__(self) -> str:
        lines = [
            "Video pipeline summary:",
            f"  Attempted:        {self.attempted:,}",
            f"  Uploaded:         {self.uploaded:,}",
            f"  Skipped existing: {self.skipped_existing:,}",
            f"  Unsupported:      {self.unsupported:,}",
            f"  Failed:           {self.failed:,}",
        ]
        if self.manifest_path:
            lines.append(f"  Manifest:         {self.manifest_path}")
        if self.status_sheet_path:
            lines.append(f"  Failed status:    {self.status_sheet_path}")
        if self.index_shards:
            lines.append(f"  Index shards:     {len(self.index_shards)} written to GCS")
        return "\n".join(lines)


def check_disk_space(
    path: str | Path | None = None,
    min_free_gb: float = 50.0,
    strict_min_gb: float = 30.0,
) -> float:
    """Check free disk space in GB and enforce safety thresholds."""
    if "VIDEO_MIN_FREE_DISK_GB" in os.environ and min_free_gb == 50.0:
        min_free_gb = float(os.environ["VIDEO_MIN_FREE_DISK_GB"])
    elif min_free_gb == 50.0 and (os.getenv("CLOUD_RUN_JOB") or os.getenv("CLOUD_RUN_TASK_INDEX") or os.getenv("K_SERVICE")):
        min_free_gb = 0.5

    if "VIDEO_STRICT_MIN_DISK_GB" in os.environ and strict_min_gb == 30.0:
        strict_min_gb = float(os.environ["VIDEO_STRICT_MIN_DISK_GB"])
    elif strict_min_gb == 30.0 and (os.getenv("CLOUD_RUN_JOB") or os.getenv("CLOUD_RUN_TASK_INDEX") or os.getenv("K_SERVICE")):
        strict_min_gb = 0.05

    target_path = Path(path or video_output_dir()).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(target_path)
    free_gb = free / (1024 ** 3)
    if free_gb < strict_min_gb:
        raise RuntimeError(
            f"Strict disk space safety threshold breached: {free_gb:.2f} GB free "
            f"(< {strict_min_gb} GB strict minimum). Halting ingestion immediately."
        )
    if free_gb < min_free_gb:
        print(f"[video][warning] Free disk space low: {free_gb:.2f} GB free (target > {min_free_gb} GB)")
    return free_gb


def _gcs_object_name(post_id: str, platform: str) -> str:
    prefix = video_gcs_prefix().strip("/")
    return f"{prefix}/{platform}/{post_id}.mp4"


def _manifest_object_name() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"manifests/video_manifest_{ts}.parquet"


def _status_object_name() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"manifests/video_status_failed_{ts}.parquet"


def _process_one(
    row: dict,
    *,
    bucket: str,
    dry_run: bool = False,
    transcode: bool = True,
    out_dir: str | Path | None = None,
    ffmpeg_threads: int = 0,
    existing_objects: set[str] | None = None,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
    min_free_disk_gb: float = 50.0,
    strict_min_disk_gb: float = 30.0,
) -> dict:
    """Process a single post row, returning a manifest record dict."""
    url = row.get("url")
    platform_code = str(row.get("platform") or "").strip().upper()
    platform_name = platform_code_to_name(platform_code) or platform_code.lower()
    post_id = media_id_from_url(str(url or ""), platform=platform_name)
    out_dir = Path(out_dir or video_output_dir()).expanduser().resolve()
    obj = _gcs_object_name(post_id, platform_name)  # object key inside the bucket

    rec: dict = {
        "platform": platform_name,
        "post_id": post_id,
        "url": str(url) if url else None,
        "status": "skipped",
        "gcs_path": f"gs://{bucket}/{obj}",
        "published_at": None,
        "duration_s": None,
        "title": None,
        "sha256": None,
        "size_bytes": None,
        "source_codec": None,
        "source_resolution": None,
        "error": None,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "transcode_args": None,
    }

    if not url:
        rec["error"] = "missing url"
        return rec

    if platform_name not in SUPPORTED_EXTRACTORS:
        rec["status"] = "unsupported"
        rec["error"] = f"no automated extractor for {platform_name}"
        return rec

    if dry_run:
        rec["status"] = "uploaded"  # dry-run treats a would-upload as ok
        print(f"[video][dry-run] {platform_code}/{post_id} -> {rec['gcs_path']}")
        return rec

    already_in_gcs = (obj in existing_objects) if existing_objects is not None else object_exists(bucket, obj)
    if already_in_gcs:
        rec["status"] = "skipped"
        print(f"[video] already in GCS, skipping: {rec['gcs_path']}")
        return rec

    check_disk_space(out_dir, min_free_gb=min_free_disk_gb, strict_min_gb=strict_min_disk_gb)

    # --- resolve ---
    try:
        media: ResolvedMedia = resolve(
            url,
            platform=platform_name,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
        )
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"resolve: {exc}"
        print(f"[video] resolve failed [{platform_code}] {post_id}: {exc}")
        return rec

    rec["published_at"] = published_at_from_info(media.info)
    rec["title"] = media.display_name or media.info.get("title")
    rec["duration_s"] = media.info.get("duration")
    rec["source_codec"] = media.info.get("vcodec")
    res = media.info.get("resolution") or media.info.get("height")
    rec["source_resolution"] = str(res) if res is not None else None

    source_local: Path | None = None
    transcoded_local: Path | None = None

    try:
        # --- download ---
        try:
            source_local = download(
                media,
                out_dir=out_dir,
                cookies=cookies,
                cookies_from_browser=cookies_from_browser,
            )
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = f"download: {exc}"
            print(f"[video] download failed [{platform_code}] {post_id}: {exc}")
            return rec

        if source_local is None or not source_local.exists():
            rec["status"] = "failed"
            rec["error"] = "download produced no file"
            return rec

        local = source_local

        # --- transcode to 480p ---
        if transcode:
            from utils.ffmpeg import build_480p_profile, transcode_to_480p

            profile = build_480p_profile(threads=ffmpeg_threads)
            rec["transcode_args"] = " ".join(profile)
            target = out_dir / platform_name / f"{media.post_id}_480p.mp4"
            try:
                tr = transcode_to_480p(source_local, target, profile=profile, overwrite=True)
                transcoded_local = tr.output
                local = transcoded_local
                if tr.duration_s:
                    rec["duration_s"] = tr.duration_s
                if source_local != transcoded_local:
                    source_local.unlink(missing_ok=True)
                    source_local = None
            except (RuntimeError, FileNotFoundError) as exc:
                rec["status"] = "failed"
                rec["error"] = f"transcode: {exc}"
                print(f"[video] transcode failed [{platform_code}] {post_id}: {exc}")
                return rec

        rec["size_bytes"] = local.stat().st_size if local.exists() else None
        try:
            rec["sha256"] = sha256_file(local)
        except OSError as exc:
            rec["sha256"] = None
            rec["error"] = f"sha256: {exc}"

        # --- upload to GCS ---
        try:
            upload_file(local, bucket=bucket, object_name=obj, dry_run=False)
            rec["status"] = "uploaded"
            rec["gcs_path"] = f"gs://{bucket}/{obj}"
            print(f"[video] uploaded {rec['gcs_path']}")
            local.unlink(missing_ok=True)
            if local == transcoded_local:
                transcoded_local = None
            if local == source_local:
                source_local = None
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = f"upload: {exc}"
            print(f"[video] upload failed [{platform_code}] {post_id}: {exc}")
        return rec
    finally:
        if source_local is not None and source_local.exists():
            source_local.unlink(missing_ok=True)
        if transcoded_local is not None and transcoded_local.exists():
            transcoded_local.unlink(missing_ok=True)


def write_records_to_gcs(
    records: list[dict],
    bucket: str,
    *,
    dry_run: bool = False,
    staging_dir: str | Path | None = None,
) -> tuple[str | None, str | None]:
    """Persist manifest + failed-status sheets as snappy Parquet in GCS.

    Returns ``(manifest_gcs_path, status_gcs_path)``.
    """
    if not records:
        return None, None
    df = pd.DataFrame(records)
    for col in _MANIFEST_SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[_MANIFEST_SCHEMA]

    staging = Path(staging_dir or video_output_dir()).expanduser().resolve() / "manifests"
    staging.mkdir(parents=True, exist_ok=True)

    def _write(df_sub: pd.DataFrame, obj: str) -> str:
        local = staging / Path(obj).name
        df_sub.to_parquet(local, index=False, compression="snappy")
        if dry_run:
            print(f"[video][dry-run] would write manifest -> gs://{bucket}/{obj}")
            local.unlink(missing_ok=True)
            return f"gs://{bucket}/{obj}"
        try:
            upload_file(local, bucket=bucket, object_name=obj, dry_run=False)
            return f"gs://{bucket}/{obj}"
        except Exception as exc:
            print(f"[video] manifest upload failed: {exc}")
            return f"gs://{bucket}/{obj}"
        finally:
            local.unlink(missing_ok=True)

    manifest_obj = _manifest_object_name()
    manifest_path = _write(df, manifest_obj)

    failed = df[df["status"] == "failed"]
    status_path = None
    if not failed.empty:
        status_obj = _status_object_name()
        status_path = _write(failed, status_obj)
    return manifest_path, status_path


def run_pipeline(
    posts_df: pd.DataFrame,
    *,
    bucket: str | None = None,
    dry_run: bool = False,
    transcode_to_480p_enabled: bool = True,
    limit: int | None = None,
    platforms: list[str] | None = None,
    concurrency: int | None = None,
    run_id: str | None = None,
    index_flush_every: int = 20,
    skip_existing: bool = True,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
) -> VideoJobResult:
    """Run the pipeline over a cleaned posts dataframe and aggregate results.

    ``concurrency`` workers process posts in parallel (defaults to
    ``VIDEO_CONCURRENCY``). Progress is flushed to GCS index shards every
    ``index_flush_every`` processed posts (and at the end) so a crashed run
    still leaves a durable record of what was uploaded/failed.
    """
    bucket = bucket or gcs_processed_bucket()
    concurrency = max(1, concurrency or video_concurrency())
    ncpu = os.cpu_count() or 4
    ffmpeg_threads = max(1, ncpu // concurrency) if concurrency > 1 else 0
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    index_paths: list[str] = []

    df = posts_df.copy()
    if platforms:
        wanted_lower = {p.lower() for p in platforms}
        df["_resolver"] = (
            df["platform"].astype(str).map(platform_code_to_name)
            .fillna(df["platform"].astype(str).str.lower())
        )
        df = df[df["_resolver"].isin(wanted_lower)].drop(columns=["_resolver"])

    result = VideoJobResult()
    all_records: list[dict] = []   # everything, for the final full manifest
    pending: list[dict] = []       # incrementally flushed to index shards

    # Single batch call to GCS to discover all existing videos instantly
    existing_objects: set[str] = set()
    if not dry_run and skip_existing:
        prefix = f"{video_gcs_prefix().strip('/')}/"
        try:
            existing_objects = list_existing_objects(bucket, prefix=prefix)
        except Exception as exc:
            print(f"[video] Note: could not pre-list GCS objects ({exc}); falling back to per-item check")

    rows_to_process: list[dict] = []

    for _, r in df.iterrows():
        row_dict = dict(r)
        url = row_dict.get("url")
        platform_code = str(row_dict.get("platform") or "").strip().upper()
        platform_name = platform_code_to_name(platform_code) or platform_code.lower()
        post_id = media_id_from_url(str(url or ""), platform=platform_name)
        obj = _gcs_object_name(post_id, platform_name)

        if not dry_run and skip_existing and obj in existing_objects:
            rec = {
                "platform": platform_name,
                "post_id": post_id,
                "url": str(url) if url else None,
                "status": "skipped",
                "gcs_path": f"gs://{bucket}/{obj}",
                "published_at": None,
                "duration_s": None,
                "title": None,
                "sha256": None,
                "size_bytes": None,
                "source_codec": None,
                "source_resolution": None,
                "error": None,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "transcode_args": None,
            }
            all_records.append(rec)
            result.skipped_existing += 1
            result.attempted += 1
        else:
            rows_to_process.append(row_dict)

    if result.skipped_existing > 0:
        print(
            f"[video] Direct skip: {result.skipped_existing} video(s) already exist in GCS. "
            f"Resuming with {len(rows_to_process)} remaining post(s)."
        )

    if limit is not None:
        rows_to_process = rows_to_process[:limit]

    rows = rows_to_process

    def _flush_index() -> None:
        if not pending:
            return
        idx = append_index_shard(pending, bucket, run_id=run_id, dry_run=dry_run)
        if idx:
            index_paths.append(idx)
        failed_sheet = append_failed_sheet(pending, bucket, run_id=run_id, dry_run=dry_run)
        if failed_sheet:
            index_paths.append(failed_sheet)
        pending.clear()

    def _handle(rec: dict) -> dict:
        all_records.append(rec)
        pending.append(rec)
        status = rec["status"]
        if status == "uploaded":
            result.uploaded += 1
        elif status == "skipped":
            result.skipped_existing += 1
        elif status == "unsupported":
            result.unsupported += 1
        else:
            result.failed += 1
            result.failures.append(str(rec.get("url")))
        result.attempted += 1
        if len(pending) >= index_flush_every:
            _flush_index()
        return rec

    if concurrency == 1 or len(rows) <= 1:
        for row in rows:
            _handle(_process_one(
                row, bucket=bucket, dry_run=dry_run,
                transcode=transcode_to_480p_enabled,
                ffmpeg_threads=ffmpeg_threads,
                existing_objects=existing_objects,
                cookies=cookies,
                cookies_from_browser=cookies_from_browser,
            ))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(_process_one, row, bucket=bucket, dry_run=dry_run,
                          transcode=transcode_to_480p_enabled,
                          ffmpeg_threads=ffmpeg_threads,
                          existing_objects=existing_objects,
                          cookies=cookies,
                          cookies_from_browser=cookies_from_browser)
                for row in rows
            ]
            # Collect results linearly so record ordering + summary stay simple.
            for fut in futures:
                _handle(fut.result())

    # Flush any remaining incremental shards, then the full-run manifest.
    _flush_index()
    result.manifest_path, result.status_sheet_path = write_records_to_gcs(
        all_records, bucket, dry_run=dry_run
    )
    result.run_id = run_id
    result.index_shards = index_paths
    return result


def load_posts(path: str | None = None) -> pd.DataFrame:
    """Load the cleaned posts file (Parquet or CSV)."""
    path = Path(path or resolve_path("CLEAN_DATA_PATH", "data/processed/posts_clean.parquet"))
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)
