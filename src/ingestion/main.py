"""Phase 2 — Data Ingestion & Cleaning.

Loads the raw All Blacks master CSV, standardises column names, filters to
short-form posts across the target platforms, drops duplicate links and writes
a clean output for downstream target labelling.

Run:
    python -m src.ingestion.main --input <csv> --output <parquet|csv>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure `utils` is importable when run as `python -m ...` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd  # noqa: E402

from src.ingestion.clean import clean_dataframe  # noqa: E402
from utils.config import resolve_path  # noqa: E402

# Platforms whose short-form content (Reels/Shorts/TikToks/tweets) we keep.
DEFAULT_PLATFORMS = ["FB", "IG", "TT", "YT"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest & clean All Blacks short-form posts.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the raw master CSV (defaults to RAW_DATA_PATH).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the clean dataset (defaults to CLEAN_DATA_PATH).",
    )
    parser.add_argument(
        "--platforms",
        type=str,
        default=",".join(DEFAULT_PLATFORMS),
        help="Comma-separated platform codes to keep (default: FB,IG,TT,YT).",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Skip the duplicate-link removal step (mostly for inspection).",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Also write the run summary to this JSON file (default: alongside output as <output>.summary.json).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_path = resolve_path("RAW_DATA_PATH", "data/raw/Post level data 2023 - 2026 - Funnel data.csv")
    output_path = resolve_path("CLEAN_DATA_PATH", "data/processed/posts_clean.parquet")
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
    if args.output:
        output_path = Path(args.output).expanduser().resolve()

    platforms = [p.strip().upper() for p in args.platforms.split(",") if p.strip()]

    print(f"[ingest] Loading raw data from: {input_path}")
    raw = pd.read_csv(input_path)
    print(f"[ingest] Raw rows: {raw.shape[0]:,}  columns: {raw.shape[1]}")

    clean = clean_dataframe(raw, platforms=platforms, drop_duplicates=not args.no_dedupe)

    # Always write BOTH a Parquet and a CSV for the processed data. The base
    # output path (from --output / CLEAN_DATA_PATH) determines the stem and
    # directory; both files are derived from it.
    output_base = Path(str(output_path))
    if output_base.suffix.lower() not in (".parquet", ".csv"):
        # Pointed at a bare stem/dir -> build a parquet name from it.
        output_base = output_base / "posts_clean" if output_base.suffix == "" else output_base.with_suffix(".parquet")
    parquet_path = output_base if output_base.suffix.lower() == ".parquet" else output_base.with_suffix(".parquet")
    csv_path = output_base if output_base.suffix.lower() == ".csv" else output_base.with_suffix(".csv")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(parquet_path, index=False)
    clean.to_csv(csv_path, index=False)

    print(f"[ingest] Wrote {clean.shape[0]:,} rows -> {parquet_path}")
    print(f"[ingest] Wrote {clean.shape[0]:,} rows -> {csv_path}")
    print(f"[ingest] Platform counts:\n{clean.groupby('platform').size()}")

    # Print the run summary (how many removed at each stage, incl. duplicates).
    # attrs stores a plain JSON-serialisable dict; rebuild for formatted display.
    from src.ingestion.summary import CleanSummary  # noqa: E402

    summary_data = clean.attrs.get("summary")
    if summary_data is not None:
        summary = CleanSummary.from_dict(summary_data)
        print("\n" + summary.format())
        # Write the summary to JSON (next to the output by default, or --report).
        report_path = Path(args.report or (str(parquet_path) + ".summary.json"))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary.to_json(), indent=2) + "\n")
        print(f"[ingest] Wrote run summary -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
