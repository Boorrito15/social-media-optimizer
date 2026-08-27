"""Video pipeline orchestration: resolve -> download -> 480p transcode -> GCS.

Reads the cleaned posts (``posts_clean.parquet``), and for each short-form
post that has a resolvable source video, produces a 480p MP4 and uploads it
to GCS under ``<VIDEO_GCS_PREFIX>/<platform>/<post_id>.mp4``.

Every run emits two artifacts to GCS (snappy-compressed Parquet):

* a **manifest** — one row per post attempted, with the GCS path, publish
  timestamp, duration, source metadata and the SHA-256 of the *uploaded* file;
* a **status sheet** — specifically the *failed* uploads, with the reason, so
  you always have a durable record of what did not make it and why.

Design goals:
* **Idempotent** — finished videos (and already-downloaded/transcoded files)
  are skipped, so the job can be re-run / run overnight safely.
* **Tolerant** — a failing post is recorded, never aborts the run.
* **GCP-first** — videos, manifest and status sheets all land in GCS, not the
  local machine (local disk is only a transient staging area).
"""

from __future__ import annotations

import os
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


@dataclass
class VideoJobResult:
    """Aggregated outcome of a pipeline run."""

    attempted: int = 0
    uploaded: int = 0
    skipped_existing: int = 0
    unsupported: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    status_sheet_path: str | None = None
    run_id: str | None = None
    index_shards: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "VIDEO UPLOAD SUMMARY",
            "=" * 50,
            f"Run id                        : {self.run_id or 'n/a'}",
            f"Posts attempted               : {self.attempted}",
            f"Uploaded to GCS               : {self.uploaded}",
            f"Skipped (already in GCS)      : {self.skipped_existing}",
            f"Unsupported platform (IG/FB)  : {self.unsupported}",
            f"Failed                        : {self.failed}",
        ]
        if self.manifest_path:
            lines.append("-" * 50)
            lines.append(f"Manifest : {self.manifest_path}")
        if self.status_sheet_path and self.failed:
            lines.append(f"Failures : {self.status_sheet_path}")
        if self.index_shards:
            lines.append(f"Index shards: {len(self.index_shards)} written to GCS")
        if self.failures:
            lines.append("-" * 50)
            lines.append("Failed URLs:")
            lines.extend(f"  - {f}" for f in self.failures[:20])
        lines.append("=" * 50)
        return "\n".join(lines)


# Columns that every manifest/status record carries.
_MANIFEST_SCHEMA = [
    "platform", "post_id", "url", "status", "gcs_path", "published_at",
    "duration_s", "title", "sha256", "size_bytes", "source_codec",
    "source_resolution", "error", "processed_at", "transcode_args",
]


def _gcs_object_name(post_id: str, platform: str) -> str:
    return f"{video_gcs_prefix().strip('/')}/{platform.lower()}/{post_id}.mp4"


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

    # --- resolve ---
    try:
        media: ResolvedMedia = resolve(url, platform=platform_name)
    except MediaResolutionError as exc:
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

    # --- download ---
    local = download(media, out_dir=out_dir)
    if local is None:
        rec["status"] = "failed"
        rec["error"] = "download produced no file"
        return rec

    # --- transcode to 480p ---
    if transcode:
        from utils.ffmpeg import build_480p_profile, transcode_to_480p

        profile = build_480p_profile(threads=ffmpeg_threads)
        rec["transcode_args"] = " ".join(profile)
        target = out_dir / platform_name / f"{media.post_id}_480p.mp4"
        try:
            tr = transcode_to_480p(local, target, profile=profile, overwrite=True)
            local = tr.output
            if tr.duration_s:
                rec["duration_s"] = tr.duration_s
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
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = f"upload: {exc}"
        print(f"[video] upload failed [{platform_code}] {post_id}: {exc}")
    return rec


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
        upload_file(local, bucket=bucket, object_name=obj, dry_run=False)
        local.unlink(missing_ok=True)
        return f"gs://{bucket}/{obj}"

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
            ))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(_process_one, row, bucket=bucket, dry_run=dry_run,
                          transcode=transcode_to_480p_enabled,
                          ffmpeg_threads=ffmpeg_threads,
                          existing_objects=existing_objects)
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
