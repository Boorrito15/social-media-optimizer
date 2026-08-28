"""Dynamic Multi-Platform Worker Pool & Auto-Capacity Balancing Supervisor.

Features:
- Continuously monitors active backlog for Facebook, Instagram, and other platforms.
- Manages 20 concurrent worker threads (10 FB + 10 IG in balanced mode by default).
- Dynamic Power Handoff: When one queue finishes, workers automatically shift to the remaining active platform, scaling it to 20 threads without thread re-creation overhead.
- Ingestion velocity monitoring with EMA smoothing (ThroughputTracker).
- Stalled worker watchdog & recycling (>60s inactivity).
- Snappy Parquet index sharding: buffer completed items and flush 15-column Snappy Parquet shards every 20 items to gs://sm-optimizer-processed/manifests/ (plus failed item sheets).
- Pre-flight batch GCS skip check (list_existing_objects) for fast idempotency.
- Operates 100% locally with zero unauthorized git commits/pushes.
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Add root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    gcs_processed_bucket,
    video_concurrency,
    video_gcs_prefix,
    video_output_dir,
)
from utils.gcs import _client, list_existing_objects, object_exists
from src.video.download import (
    SUPPORTED_EXTRACTORS,
    media_id_from_url,
    platform_code_to_name,
)
from src.video.index import (
    INDEX_SCHEMA,
    append_failed_sheet,
    append_index_shard,
    consolidate_index,
)
from src.video.upload import (
    _MANIFEST_SCHEMA,
    _gcs_object_name,
    _process_one,
    check_disk_space,
    write_records_to_gcs,
)

TOTAL_TARGETS = {
    "facebook": 4578,
    "instagram": 4875,
}

MAX_TOTAL_CONCURRENCY = 20
RUNNING_PROCESSES = {}


@dataclass(frozen=True)
class PlatformStats:
    """Immutable snapshot of platform-specific ingestion metrics."""

    attempted: int
    uploaded: int
    skipped: int
    failed: int
    target: int
    total_bytes: int
    velocity_vpm: float
    progress_pct: float

    @property
    def processed(self) -> int:
        return self.uploaded + self.skipped

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.processed)


@dataclass(frozen=True)
class TrackerSnapshot:
    """Atomic, immutable snapshot of overall ingestion metrics."""

    total_attempted: int
    total_uploaded: int
    total_skipped: int
    total_failed: int
    total_processed: int
    total_target: int
    total_bytes: int
    remaining: int
    progress_pct: float
    velocity_vpm: float
    velocity_vph: int
    session_velocity_vpm: float
    eta_seconds: float | None
    eta_str: str
    duration_seconds: float
    platform_stats: dict[str, PlatformStats]


@dataclass
class SupervisorResult:
    """Result summary emitted upon supervisor run completion."""

    total_processed: int
    total_failed: int
    total_skipped: int
    platform_counts: dict[str, int]
    power_handoff_triggered: bool
    average_speed_vpm: float
    duration_seconds: float
    total_uploaded: int = 0
    index_shards: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    status_sheet_path: str | None = None
    run_id: str | None = None


class ThroughputTracker:
    """Thread-safe velocity and throughput tracker for multi-platform scraping pools."""

    def __init__(
        self,
        targets: dict[str, int] | None = None,
        window_seconds: float = 60.0,
        ema_alpha: float = 0.15,
        eta_beta: float = 0.10,
        start_time: float | None = None,
    ) -> None:
        self._targets = dict(targets if targets is not None else {"facebook": 4578, "instagram": 4875})
        self._window_seconds = float(window_seconds)
        self._ema_alpha = float(ema_alpha)
        self._eta_beta = float(eta_beta)
        self._start_time = float(start_time if start_time is not None else time.time())
        self._lock = threading.RLock()
        self._events: deque[tuple[float, str, str, int]] = deque()
        self._platform_counts: dict[str, dict[str, int]] = {}
        for p in self._targets:
            self._platform_counts[p] = {"attempted": 0, "uploaded": 0, "skipped": 0, "failed": 0, "bytes": 0}
        self._ema_vpm: float | None = None
        self._platform_ema_vpm: dict[str, float | None] = {}
        self._last_event_time: float | None = None
        self._last_platform_event_time: dict[str, float | None] = {}
        self._platform_first_time: dict[str, float] = {}
        self._ema_eta_seconds: float | None = None

    def _ensure_platform(self, platform: str, now: float) -> None:
        if platform not in self._platform_counts:
            self._platform_counts[platform] = {"attempted": 0, "uploaded": 0, "skipped": 0, "failed": 0, "bytes": 0}
        if platform not in self._targets:
            self._targets[platform] = 0
        if platform not in self._platform_first_time:
            self._platform_first_time[platform] = now

    def record_item(
        self,
        platform: str,
        status: str,
        bytes_count: int = 0,
        duration_s: float | None = None,
        now: float | None = None,
    ) -> None:
        t = float(now if now is not None else time.time())
        with self._lock:
            self._ensure_platform(platform, t)
            counts = self._platform_counts[platform]
            counts["attempted"] += 1
            st = status.lower()
            if st == "uploaded":
                counts["uploaded"] += 1
                counts["bytes"] += max(0, int(bytes_count))
            elif st == "skipped":
                counts["skipped"] += 1
            elif st == "failed":
                counts["failed"] += 1

            if st in ("uploaded", "skipped"):
                self._events.append((t, platform, st, bytes_count))
                self._prune_events(t)
                win_vpm = self._calc_window_velocity(now=t)
                if self._ema_vpm is None:
                    self._ema_vpm = win_vpm
                else:
                    self._ema_vpm = self._ema_alpha * win_vpm + (1.0 - self._ema_alpha) * self._ema_vpm
                self._last_event_time = t

                p_win_vpm = self._calc_window_velocity(platform=platform, now=t)
                if platform not in self._platform_ema_vpm or self._platform_ema_vpm[platform] is None:
                    self._platform_ema_vpm[platform] = p_win_vpm
                else:
                    self._platform_ema_vpm[platform] = (
                        self._ema_alpha * p_win_vpm + (1.0 - self._ema_alpha) * (self._platform_ema_vpm[platform] or 0.0)
                    )
                self._last_platform_event_time[platform] = t

    def record_success(
        self,
        platform: str,
        bytes_count: int = 0,
        duration_s: float | None = None,
        now: float | None = None,
    ) -> None:
        self.record_item(platform, "uploaded", bytes_count=bytes_count, duration_s=duration_s, now=now)

    def record_skip(self, platform: str, now: float | None = None) -> None:
        self.record_item(platform, "skipped", now=now)

    def record_failure(self, platform: str, error: str | None = None, now: float | None = None) -> None:
        self.record_item(platform, "failed", now=now)

    def _prune_events(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _calc_window_velocity(self, platform: str | None = None, now: float | None = None) -> float:
        t = float(now if now is not None else time.time())
        cutoff = t - self._window_seconds
        count = 0
        for ts, p, st, _ in reversed(self._events):
            if ts > cutoff:
                if platform is None or p == platform:
                    count += 1
            elif ts < cutoff - 120.0:
                break
        t_start = self._platform_first_time.get(platform, self._start_time) if platform else self._start_time
        dt_eff = min(self._window_seconds, max(1.0, t - t_start))
        return (count / dt_eff) * 60.0

    def get_velocity(self, platform: str | None = None, now: float | None = None, ema: bool = True) -> float:
        t = float(now if now is not None else time.time())
        with self._lock:
            win_vpm = self._calc_window_velocity(platform=platform, now=t)
            if not ema:
                return win_vpm
            base_ema = self._platform_ema_vpm.get(platform) if platform else self._ema_vpm
            if base_ema is None:
                return win_vpm
            last_t = self._last_platform_event_time.get(platform) if platform else self._last_event_time
            if last_t is None:
                return win_vpm
            idle = max(0.0, t - last_t)
            if idle > 5.0:
                decay = math.exp(-idle / 30.0)
                decayed_ema = base_ema * decay
            else:
                decayed_ema = base_ema
            effective = max(win_vpm, decayed_ema)
            return 0.0 if effective < 0.01 else effective

    def get_session_velocity(self, now: float | None = None) -> float:
        t = float(now if now is not None else time.time())
        with self._lock:
            total_proc = sum(c["uploaded"] + c["skipped"] for c in self._platform_counts.values())
            elapsed = max(1.0, t - self._start_time)
            return (total_proc / elapsed) * 60.0

    def get_eta_seconds(self, now: float | None = None, use_ema: bool = True) -> float | None:
        t = float(now if now is not None else time.time())
        with self._lock:
            total_target = sum(self._targets.values())
            total_proc = sum(c["uploaded"] + c["skipped"] for c in self._platform_counts.values())
            remaining = max(0, total_target - total_proc)
            if remaining == 0:
                return 0.0
            vel = self.get_velocity(now=t, ema=use_ema)
            if vel <= 0.01:
                vel = self.get_session_velocity(now=t)
            if vel <= 0.01:
                return None
            raw_eta = remaining / (vel / 60.0)
            if self._ema_eta_seconds is None:
                self._ema_eta_seconds = raw_eta
            else:
                self._ema_eta_seconds = self._eta_beta * raw_eta + (1.0 - self._eta_beta) * self._ema_eta_seconds
            return self._ema_eta_seconds if use_ema else raw_eta

    def format_speed(self, ema: bool = True, now: float | None = None) -> str:
        vpm = self.get_velocity(now=now, ema=ema)
        return f"Speed {vpm:.1f} /min"

    def format_eta(self, now: float | None = None) -> str:
        eta_s = self.get_eta_seconds(now=now, use_ema=True)
        if eta_s is None:
            return "Calculating..."
        if eta_s <= 0.0:
            return "Complete"
        if eta_s < 60:
            return f"in ~{int(round(eta_s))}s"
        elif eta_s < 3600:
            mins = int(round(eta_s / 60.0))
            return f"in ~{mins}m"
        else:
            hours = int(eta_s // 3600)
            mins = int(round((eta_s % 3600) / 60.0))
            if mins == 60:
                hours += 1
                mins = 0
            return f"in ~{hours}h {mins:02d}m" if mins > 0 else f"in ~{hours}h"

    def format_status_line(self, now: float | None = None) -> str:
        snap = self.snapshot(now=now)
        fb = snap.platform_stats.get("facebook")
        ig = snap.platform_stats.get("instagram")
        fb_str = f"FB {fb.processed}/{fb.target}" if fb else ""
        ig_str = f"IG {ig.processed}/{ig.target}" if ig else ""
        parts = [p for p in [fb_str, ig_str] if p]
        plat_summary = " | ".join(parts)
        return (
            f"📊 Status: {plat_summary} | Total {snap.total_processed}/{snap.total_target} "
            f"({snap.progress_pct:.1f}%) | {self.format_speed(now=now)} (~{snap.velocity_vph:,} /h) | ETA {snap.eta_str}"
        )

    def snapshot(self, now: float | None = None) -> TrackerSnapshot:
        t = float(now if now is not None else time.time())
        with self._lock:
            p_stats = {}
            for p, c in self._platform_counts.items():
                target = self._targets.get(p, 0)
                proc = c["uploaded"] + c["skipped"]
                pct = (proc / target * 100.0) if target > 0 else 0.0
                vel = self.get_velocity(platform=p, now=t, ema=True)
                p_stats[p] = PlatformStats(
                    attempted=c["attempted"],
                    uploaded=c["uploaded"],
                    skipped=c["skipped"],
                    failed=c["failed"],
                    target=target,
                    total_bytes=c["bytes"],
                    velocity_vpm=vel,
                    progress_pct=pct,
                )
            tot_att = sum(c["attempted"] for c in self._platform_counts.values())
            tot_up = sum(c["uploaded"] for c in self._platform_counts.values())
            tot_sk = sum(c["skipped"] for c in self._platform_counts.values())
            tot_fa = sum(c["failed"] for c in self._platform_counts.values())
            tot_proc = tot_up + tot_sk
            tot_tgt = sum(self._targets.values())
            tot_bytes = sum(c["bytes"] for c in self._platform_counts.values())
            rem = max(0, tot_tgt - tot_proc)
            pct = (tot_proc / tot_tgt * 100.0) if tot_tgt > 0 else 0.0
            vel_vpm = self.get_velocity(now=t, ema=True)
            vel_vph = int(round(vel_vpm * 60.0))
            sess_vpm = self.get_session_velocity(now=t)
            eta_s = self.get_eta_seconds(now=t, use_ema=True)
            eta_str = self.format_eta(now=t)
            dur = max(0.0, t - self._start_time)
            return TrackerSnapshot(
                total_attempted=tot_att,
                total_uploaded=tot_up,
                total_skipped=tot_sk,
                total_failed=tot_fa,
                total_processed=tot_proc,
                total_target=tot_tgt,
                total_bytes=tot_bytes,
                remaining=rem,
                progress_pct=pct,
                velocity_vpm=vel_vpm,
                velocity_vph=vel_vph,
                session_velocity_vpm=sess_vpm,
                eta_seconds=eta_s,
                eta_str=eta_str,
                duration_seconds=dur,
                platform_stats=p_stats,
            )

    def get_platform_counts(self) -> dict[str, int]:
        with self._lock:
            return {p: c["uploaded"] + c["skipped"] for p, c in self._platform_counts.items()}

    def reset(self, start_time: float | None = None) -> None:
        with self._lock:
            self._start_time = float(start_time if start_time is not None else time.time())
            self._events.clear()
            self._ema_vpm = None
            self._platform_ema_vpm.clear()
            self._last_event_time = None
            self._last_platform_event_time.clear()
            self._platform_first_time.clear()
            self._ema_eta_seconds = None
            for p in self._platform_counts:
                self._platform_counts[p] = {"attempted": 0, "uploaded": 0, "skipped": 0, "failed": 0, "bytes": 0}


class DynamicSupervisor:
    """Intelligent Dynamic Scraper Supervisor managing concurrent multi-platform workers."""

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        max_total_concurrency: int = 20,
        initial_allocations: dict[str, int] | None = None,
        bucket_name: str | None = None,
        dry_run: bool = False,
        stall_timeout_s: float = 60.0,
        index_flush_every: int = 20,
        staging_dir: str | Path | None = None,
        transcode_to_480p_enabled: bool = True,
        min_free_disk_gb: float = 50.0,
        strict_min_disk_gb: float = 30.0,
        skip_existing: bool = True,
        cookies: str | None = None,
        cookies_from_browser: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.max_total_concurrency = int(max_total_concurrency)
        self.bucket_name = bucket_name or gcs_processed_bucket()
        self.dry_run = bool(dry_run)
        self.stall_timeout_s = float(stall_timeout_s)
        self.index_flush_every = int(index_flush_every)
        self.staging_dir = Path(staging_dir or video_output_dir()).expanduser().resolve()
        self.transcode_to_480p_enabled = bool(transcode_to_480p_enabled)
        if min_free_disk_gb == 50.0 and "VIDEO_MIN_FREE_DISK_GB" in os.environ:
            self.min_free_disk_gb = float(os.environ["VIDEO_MIN_FREE_DISK_GB"])
        elif min_free_disk_gb == 50.0 and (os.getenv("CLOUD_RUN_JOB") or os.getenv("CLOUD_RUN_TASK_INDEX") or os.getenv("K_SERVICE")):
            self.min_free_disk_gb = 0.5
        else:
            self.min_free_disk_gb = float(min_free_disk_gb)

        if strict_min_disk_gb == 30.0 and "VIDEO_STRICT_MIN_DISK_GB" in os.environ:
            self.strict_min_disk_gb = float(os.environ["VIDEO_STRICT_MIN_DISK_GB"])
        elif strict_min_disk_gb == 30.0 and (os.getenv("CLOUD_RUN_JOB") or os.getenv("CLOUD_RUN_TASK_INDEX") or os.getenv("K_SERVICE")):
            self.strict_min_disk_gb = 0.05
        else:
            self.strict_min_disk_gb = float(strict_min_disk_gb)
        self.skip_existing = bool(skip_existing)
        self.cookies = cookies
        self.cookies_from_browser = cookies_from_browser
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # Configure worker thread allocations
        if initial_allocations is not None:
            self.initial_allocations = dict(initial_allocations)
        else:
            fb_count = self.max_total_concurrency // 2
            ig_count = self.max_total_concurrency - fb_count
            self.initial_allocations = {"facebook": fb_count, "instagram": ig_count}

        # Multi-queue state
        self._queues: dict[str, queue.Queue] = {}
        if not self.df.empty and "platform" in self.df.columns:
            for raw_plat in self.df["platform"].dropna().unique():
                p_name = platform_code_to_name(str(raw_plat)) or str(raw_plat).lower()
                self._queues[p_name] = queue.Queue()
            for p in self.initial_allocations:
                if p not in self._queues and self.initial_allocations[p] > 0:
                    self._queues[p] = queue.Queue()
        else:
            for p in self.initial_allocations:
                self._queues[p] = queue.Queue()

        self._active_worker_counts: dict[str, int] = {p: 0 for p in self.initial_allocations}
        self.power_handoff_triggered = False

        # Synchronization & tracking
        self._rebalance_lock = threading.RLock()
        self._index_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_activity: dict[int, dict[str, Any]] = {}
        self._worker_threads: list[threading.Thread] = []
        self._next_worker_id = 0

        # Manifest & shard persistence
        self._pending_records: list[dict] = []
        self._all_records: list[dict] = []
        self._index_shard_paths: list[str] = []

        # Target accounting and ThroughputTracker
        platform_targets: dict[str, int] = {}
        for p in self.initial_allocations:
            platform_targets[p] = 0
        if not self.df.empty and "platform" in self.df.columns:
            for raw_plat, count in self.df["platform"].value_counts().items():
                p_name = platform_code_to_name(str(raw_plat)) or str(raw_plat).lower()
                platform_targets[p_name] = platform_targets.get(p_name, 0) + int(count)

        self.tracker = ThroughputTracker(targets=platform_targets, start_time=time.time())

    def calculate_velocity(self) -> float:
        """Return instantaneous / EMA ingestion speed in items per minute."""
        return self.tracker.get_velocity()

    def check_and_rebalance(self) -> bool:
        """Evaluate platform queues and trigger Power Handoff if an imbalance exists."""
        with self._rebalance_lock:
            empty_platforms = [p for p, q in self._queues.items() if q.empty()]
            active_platforms = [p for p, q in self._queues.items() if not q.empty()]
            if empty_platforms and active_platforms:
                self.power_handoff_triggered = True
                return True
            return False

    def _init_queues_and_preflight(self) -> None:
        """Pre-populate platform queues and perform pre-flight batch GCS skip check."""
        if self.df.empty:
            return

        existing_objects: set[str] | None = None
        if self.skip_existing and not self.dry_run:
            prefix = f"{video_gcs_prefix().strip('/')}/"
            try:
                existing_objects = list_existing_objects(self.bucket_name, prefix=prefix)
            except Exception as exc:
                print(f"[supervisor] Note: batch pre-listing GCS objects failed ({exc}), fallback active.")
                existing_objects = None

        for _, row in self.df.iterrows():
            row_dict = dict(row)
            url = row_dict.get("url")
            platform_code = str(row_dict.get("platform") or "").strip().upper()
            platform_name = platform_code_to_name(platform_code) or platform_code.lower() or "unknown"
            post_id = media_id_from_url(str(url or ""), platform=platform_name)
            obj = _gcs_object_name(post_id, platform_name)

            if platform_name not in self._queues:
                self._queues[platform_name] = queue.Queue()
                self._active_worker_counts[platform_name] = 0

            is_in_gcs = False
            if self.skip_existing and not self.dry_run:
                if existing_objects is not None:
                    is_in_gcs = obj in existing_objects
                else:
                    is_in_gcs = object_exists(self.bucket_name, obj)

            if is_in_gcs:
                rec = {
                    "platform": platform_name,
                    "post_id": post_id,
                    "url": str(url) if url else None,
                    "status": "skipped",
                    "gcs_path": f"gs://{self.bucket_name}/{obj}",
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
                with self._index_lock:
                    self._all_records.append(rec)
                    self._pending_records.append(rec)
                self.tracker.record_skip(platform_name)
            else:
                self._queues[platform_name].put(row_dict)

    def _flush_shards(self, force: bool = False) -> None:
        """Flush pending records to GCS as snappy Parquet index shards and failed sheets."""
        with self._index_lock:
            if not self._pending_records and not force:
                return
            if not self._pending_records:
                return
            records_to_flush = list(self._pending_records)
            self._pending_records.clear()

            shard_path = append_index_shard(
                records_to_flush,
                self.bucket_name,
                run_id=self.run_id,
                dry_run=self.dry_run,
                staging_dir=self.staging_dir,
            )
            if shard_path:
                self._index_shard_paths.append(shard_path)

            failed_sheet = append_failed_sheet(
                records_to_flush,
                self.bucket_name,
                run_id=self.run_id,
                dry_run=self.dry_run,
                staging_dir=self.staging_dir,
            )
            if failed_sheet:
                self._index_shard_paths.append(failed_sheet)

    def _process_post(self, row: dict, platform: str, worker_id: int | None = None) -> None:
        """Execute single post extraction, transcode, upload, and update tracker."""
        ncpu = os.cpu_count() or 4
        ffmpeg_threads = max(1, ncpu // self.max_total_concurrency) if self.max_total_concurrency > 1 else 0

        rec = _process_one(
            row,
            bucket=self.bucket_name,
            dry_run=self.dry_run,
            transcode=self.transcode_to_480p_enabled,
            out_dir=self.staging_dir,
            ffmpeg_threads=ffmpeg_threads,
            existing_objects=set(),
            cookies=self.cookies,
            cookies_from_browser=self.cookies_from_browser,
            min_free_disk_gb=self.min_free_disk_gb,
            strict_min_disk_gb=self.strict_min_disk_gb,
        )

        if worker_id is not None:
            with self._rebalance_lock:
                act = self._worker_activity.get(worker_id)
                if act is None or act.get("cancelled", False):
                    return

        st = rec.get("status", "failed")
        bytes_count = rec.get("size_bytes") or 0
        duration_s = rec.get("duration_s")

        if st == "uploaded":
            self.tracker.record_success(platform, bytes_count=bytes_count, duration_s=duration_s)
        elif st == "skipped":
            self.tracker.record_skip(platform)
        else:
            self.tracker.record_failure(platform, error=rec.get("error"))

        with self._index_lock:
            self._all_records.append(rec)
            self._pending_records.append(rec)
            if len(self._pending_records) >= self.index_flush_every:
                self._flush_shards()

    def _worker_loop(self, worker_id: int, initial_platform: str) -> None:
        """Worker event loop with zero-thread-allocation Dynamic Power Handoff."""
        active_platform = initial_platform
        with self._rebalance_lock:
            self._active_worker_counts[active_platform] = self._active_worker_counts.get(active_platform, 0) + 1

        try:
            while not self._stop_event.is_set():
                # Enforce disk space safety pre-check
                if not self.dry_run:
                    try:
                        check_disk_space(
                            self.staging_dir,
                            min_free_gb=self.min_free_disk_gb,
                            strict_min_gb=self.strict_min_disk_gb,
                        )
                    except RuntimeError as exc:
                        print(f"🛑 [supervisor] Disk space guard breached: {exc}", file=sys.stderr)
                        self._stop_event.set()
                        break

                item = None
                q = self._queues.get(active_platform)
                if q is not None:
                    try:
                        item = q.get(timeout=0.05)
                    except queue.Empty:
                        item = None

                if item is None:
                    # Current platform queue exhausted -> execute Dynamic Power Handoff
                    with self._rebalance_lock:
                        other_active = [
                            p for p, p_q in self._queues.items()
                            if p != active_platform and (not p_q.empty() or any(act.get("platform") == p for act in self._worker_activity.values()))
                        ]
                        if other_active:
                            self.power_handoff_triggered = True

                        candidate_platforms = [p for p, p_q in self._queues.items() if not p_q.empty()]
                        if not candidate_platforms:
                            # All platform queues are empty -> terminate worker
                            break

                        next_platform = candidate_platforms[0]
                        if active_platform != next_platform:
                            old_platform = active_platform
                            self._active_worker_counts[old_platform] = max(0, self._active_worker_counts.get(old_platform, 1) - 1)
                            self._active_worker_counts[next_platform] = self._active_worker_counts.get(next_platform, 0) + 1
                            active_platform = next_platform
                            self.power_handoff_triggered = True
                            active_count = self._active_worker_counts[next_platform]
                            print(
                                f"🚀 [supervisor] Power Handoff: Worker {worker_id} shifted from "
                                f"{old_platform} to {next_platform} (Scaling {next_platform} to {active_count} active threads)!"
                            )

                        try:
                            item = self._queues[active_platform].get(timeout=0.05)
                        except queue.Empty:
                            continue

                # Process the fetched post
                try:
                    with self._rebalance_lock:
                        self._worker_activity[worker_id] = {
                            "start_time": time.time(),
                            "row": item,
                            "platform": active_platform,
                            "cancelled": False,
                        }
                    self._process_post(item, active_platform, worker_id=worker_id)
                finally:
                    with self._rebalance_lock:
                        was_active = self._worker_activity.pop(worker_id, None)
                    if active_platform in self._queues and was_active is not None and not was_active.get("cancelled", False):
                        try:
                            self._queues[active_platform].task_done()
                        except ValueError:
                            pass
        finally:
            with self._rebalance_lock:
                self._active_worker_counts[active_platform] = max(0, self._active_worker_counts.get(active_platform, 1) - 1)

    def _check_watchdog(self, now: float) -> None:
        """Watchdog to detect stalled workers, record timeouts, and recycle connections."""
        with self._rebalance_lock:
            for w_id, act in list(self._worker_activity.items()):
                elapsed = now - act["start_time"]
                if elapsed > self.stall_timeout_s:
                    act["cancelled"] = True
                    plat = act["platform"]
                    row = act["row"]
                    post_id = media_id_from_url(row.get("url") or "", platform=plat)
                    err_msg = f"stalled socket watchdog timeout ({self.stall_timeout_s}s)"

                    rec = {
                        "platform": plat,
                        "post_id": post_id,
                        "url": str(row.get("url")) if row.get("url") else None,
                        "status": "failed",
                        "gcs_path": f"gs://{self.bucket_name}/{_gcs_object_name(post_id, plat)}",
                        "published_at": None,
                        "duration_s": None,
                        "title": None,
                        "sha256": None,
                        "size_bytes": None,
                        "source_codec": None,
                        "source_resolution": None,
                        "error": err_msg,
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "transcode_args": None,
                    }

                    self.tracker.record_failure(plat, error=err_msg, now=now)
                    with self._index_lock:
                        self._all_records.append(rec)
                        self._pending_records.append(rec)
                        if len(self._pending_records) >= self.index_flush_every:
                            self._flush_shards()

                    self._worker_activity.pop(w_id, None)
                    if plat in self._queues:
                        try:
                            self._queues[plat].task_done()
                        except ValueError:
                            pass

                    print(
                        f"⚠️ [supervisor] Watchdog: Worker {w_id} stalled > {self.stall_timeout_s:.1f}s on {plat}! "
                        f"Recycling worker connection..."
                    )

                    # Spawn a replacement worker thread to maintain pool concurrency
                    self._next_worker_id += 1
                    new_id = self._next_worker_id
                    plat_tag = "FB" if "face" in plat else ("IG" if "insta" in plat else plat.upper()[:2])
                    new_t = threading.Thread(
                        target=self._worker_loop,
                        args=(new_id, plat),
                        name=f"DynamicWorker-{plat_tag}-{new_id}",
                        daemon=True,
                    )
                    new_t.start()
                    self._worker_threads.append(new_t)

    def run(self) -> SupervisorResult:
        """Run the dynamic multi-platform worker pool until all Meta videos are processed."""
        start_time = time.time()
        self.tracker.reset(start_time=start_time)
        self._init_queues_and_preflight()

        # Check if all platform queues are empty at startup (e.g. empty df or 100% pre-flight skipped)
        all_empty = all(q.empty() for q in self._queues.values())
        if all_empty:
            self._flush_shards(force=True)
            manifest_path, status_path = write_records_to_gcs(
                self._all_records,
                self.bucket_name,
                dry_run=self.dry_run,
                staging_dir=self.staging_dir,
            )
            snap = self.tracker.snapshot()
            dur = max(0.001, time.time() - start_time)
            avg_speed = (snap.total_processed / dur) * 60.0
            return SupervisorResult(
                total_processed=snap.total_processed,
                total_failed=snap.total_failed,
                total_skipped=snap.total_skipped,
                platform_counts=self.tracker.get_platform_counts(),
                power_handoff_triggered=self.power_handoff_triggered,
                average_speed_vpm=avg_speed,
                duration_seconds=dur,
                total_uploaded=snap.total_uploaded,
                index_shards=list(self._index_shard_paths),
                manifest_path=manifest_path,
                status_sheet_path=status_path,
                run_id=self.run_id,
            )

        # Spawn worker threads across platforms according to allocations
        worker_id = 0
        for platform, count in self.initial_allocations.items():
            plat_tag = "FB" if "face" in platform else ("IG" if "insta" in platform else platform.upper()[:2])
            for i in range(count):
                worker_id += 1
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id, platform),
                    name=f"DynamicWorker-{plat_tag}-{worker_id}",
                    daemon=True,
                )
                self._worker_threads.append(t)
                t.start()
        self._next_worker_id = worker_id

        # Supervisory monitoring and watchdog loop
        try:
            last_status_print = time.time()
            while not self._stop_event.is_set():
                any_alive = any(t.is_alive() for t in self._worker_threads)
                if not any_alive:
                    break

                now = time.time()
                self._check_watchdog(now)
                self.check_and_rebalance()

                if now - last_status_print >= 5.0:
                    print(self.tracker.format_status_line(now=now))
                    last_status_print = now

                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n🛑 [supervisor] Received interrupt signal. Gracefully shutting down workers...")
            self._stop_event.set()

        # Join all worker threads
        for t in self._worker_threads:
            t.join(timeout=1.0)

        # Final shard flush and consolidated manifest persistence
        self._flush_shards(force=True)
        manifest_path, status_path = write_records_to_gcs(
            self._all_records,
            self.bucket_name,
            dry_run=self.dry_run,
            staging_dir=self.staging_dir,
        )

        snap = self.tracker.snapshot()
        dur = max(0.001, time.time() - start_time)
        avg_speed = (snap.total_processed / dur) * 60.0

        return SupervisorResult(
            total_processed=snap.total_processed,
            total_failed=snap.total_failed,
            total_skipped=snap.total_skipped,
            platform_counts=self.tracker.get_platform_counts(),
            power_handoff_triggered=self.power_handoff_triggered,
            average_speed_vpm=avg_speed,
            duration_seconds=dur,
            total_uploaded=snap.total_uploaded,
            index_shards=list(self._index_shard_paths),
            manifest_path=manifest_path,
            status_sheet_path=status_path,
            run_id=self.run_id,
        )


def get_current_gcs_counts() -> tuple[int, int]:
    """Retrieve existing blob counts for Facebook and Instagram from GCS."""
    try:
        client = _client()
        bucket = client.bucket(gcs_processed_bucket())
        fb_count = len(list(bucket.list_blobs(prefix="videos/facebook/", fields="items(name),nextPageToken")))
        ig_count = len(list(bucket.list_blobs(prefix="videos/instagram/", fields="items(name),nextPageToken")))
        return fb_count, ig_count
    except Exception as exc:
        print(f"[supervisor] Error querying GCS counts: {exc}", file=sys.stderr)
        return 0, 0


def supervisor_loop():
    """Supervisor loop CLI monitor."""
    print("⚡️ [supervisor] Dynamic Multi-Platform Worker Pool initialized (Max Concurrency: 20).")
    
    while True:
        fb_done, ig_done = get_current_gcs_counts()
        fb_remaining = max(0, TOTAL_TARGETS["facebook"] - fb_done)
        ig_remaining = max(0, TOTAL_TARGETS["instagram"] - ig_done)
        
        print(f"📊 [supervisor] Status: FB {fb_done}/{TOTAL_TARGETS['facebook']} (rem: {fb_remaining}) | IG {ig_done}/{TOTAL_TARGETS['instagram']} (rem: {ig_remaining})")
        
        if fb_remaining == 0 and ig_remaining == 0:
            print("🎉 [supervisor] All Meta videos successfully ingested into GCS!")
            break
            
        if ig_remaining == 0 and fb_remaining > 0:
            print(f"🚀 [supervisor] Instagram Completed! Power Handoff: Allocating ALL {MAX_TOTAL_CONCURRENCY} threads to Facebook!")
        elif fb_remaining == 0 and ig_remaining > 0:
            print(f"🚀 [supervisor] Facebook Completed! Power Handoff: Allocating ALL {MAX_TOTAL_CONCURRENCY} threads to Instagram!")
            
        time.sleep(30.0)


if __name__ == "__main__":
    supervisor_loop()
