"""Command-line entrypoint for the video pipeline.

Scrapes short-form videos, transcodes them to 480p and uploads them to GCS.

Examples:
    # Dry-run over 5 posts (no network / GCS writes)
    python -m src.video.main --limit 5 --dry-run

    # Process only YouTube posts
    python -m src.video.main --platforms youtube

    # Run for real (downloads + transcodes + uploads)
    python -m src.video.main

Options are documented with ``--help``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.config import gcs_processed_bucket, service_account_credentials, video_concurrency  # noqa: E402
from src.video.upload import load_posts, run_pipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape short-form videos, transcode to 480p and upload to GCS."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Cleaned posts file (Parquet or CSV). Defaults to CLEAN_DATA_PATH.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help=f"GCS bucket to upload to (default: {gcs_processed_bucket()}).",
    )
    parser.add_argument(
        "--platforms",
        type=str,
        default=None,
        help="Comma-separated platforms to process (e.g. youtube,tiktok). Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of posts processed (useful for tests/dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Trace the pipeline without downloading, transcoding or uploading.",
    )
    parser.add_argument(
        "--no-transcode",
        action="store_true",
        help="Upload the source (original resolution) instead of the 480p transcode.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: VIDEO_CONCURRENCY={video_concurrency()}).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Stable id for this run (used to name index shards + logs).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to a log file to upload to GCS as logs/<run_id>.log.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip videos that already exist in GCS.",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="After the run, merge all index shards into videos_index.parquet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.dry_run:
        creds_path = service_account_credentials()
        if creds_path is None or not creds_path.is_file():
            print(
                f"\n[ERROR] Service account key file not found: {creds_path}\n"
                f"Please place your GCP service account JSON key in '{creds_path or 'config/le-wagon-2303-service-account.json'}'\n"
                f"or update GOOGLE_APPLICATION_CREDENTIALS in your .env file.\n"
            )
            return 1

    df = load_posts(args.data)
    platforms = [p.strip() for p in args.platforms.split(",")] if args.platforms else None

    print(
        f"[video] pipeline over {len(df):,} posts "
        f"(platforms={platforms or 'all'}, limit={args.limit or 'none'}, "
        f"dry_run={args.dry_run}, transcode={not args.no_transcode}, "
        f"concurrency={args.concurrency or video_concurrency()})"
    )

    result = run_pipeline(
        df,
        bucket=args.bucket,
        dry_run=args.dry_run,
        transcode_to_480p_enabled=not args.no_transcode,
        limit=args.limit,
        platforms=platforms,
        concurrency=args.concurrency,
        run_id=args.run_id,
        skip_existing=not args.no_skip_existing,
    )
    print("\n" + str(result))

    # Upload the run log to GCS so failures are inspectable after the fact.
    if args.log and not args.dry_run:
        from src.video.index import upload_log

        log_path = upload_log(args.log, args.bucket or gcs_processed_bucket(), result.run_id)
        if log_path:
            print(f"[video] uploaded log -> {log_path}")

    # Optionally consolidate all index shards into one cumulative registry.
    if args.consolidate and not args.dry_run:
        from src.video.index import consolidate_index

        consolidate_index(args.bucket or gcs_processed_bucket())

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
