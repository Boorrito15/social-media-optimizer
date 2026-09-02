"""CLI: ``python -m models.train [--task {lin,clas,all}] [--retrain] [...]``.

Reads defaults from the environment and overrides from CLI flags. Honouring
``RETRAIN`` and ``RETRAIN_STATE`` env vars lets you toggle the cache from
notebook / shell without changing the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import clas, linreg
from models.config import PIPELINE_STATE_VERSION, STATE_PATH, PipelineConfig, truthy
from models.shared.build_state import build_state


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train (or load) the SVM models.")
    ap.add_argument("--task", choices=["lin", "clas", "all"], default="all")
    ap.add_argument("--retrain", action="store_true", help="Overwrite saved model artefacts.")
    ap.add_argument(
        "--retrain-state",
        action="store_true",
        help="Force a rebuild of pipeline_state.joblib even when present.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--n-iqr-bins", type=int, default=2)
    ap.add_argument("--bottom-iqr-pct", type=float, default=0.1)
    ap.add_argument("--top-iqr-pct", type=float, default=0.9)
    ap.add_argument("--min-vocab-freq", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = PipelineConfig(
        train_test_random_seed=args.seed,
        test_size=args.test_size,
        n_iqr_bins=args.n_iqr_bins,
        bottom_iqr_percentile=args.bottom_iqr_pct,
        top_iqr_percentile=args.top_iqr_pct,
        min_vocab_frequency=args.min_vocab_freq,
    )
    retrain = args.retrain or truthy("RETRAIN")
    retrain_state = args.retrain_state or truthy("RETRAIN_STATE")

    if retrain_state or not STATE_PATH.exists():
        print(f"[state] building pipeline_state.joblib (version {PIPELINE_STATE_VERSION})")
        build_state(cfg, force=retrain_state)
    else:
        print(f"[state] {STATE_PATH} already exists; use --retrain-state to rebuild")

    if args.task in ("lin", "all"):
        b = linreg.train_or_load(config=cfg, retrain=retrain)
        print(f"[lin]  bundle keys: {sorted(b)}")

    if args.task in ("clas", "all"):
        b = clas.train_or_load(config=cfg, retrain=retrain)
        print(f"[clas] bundle keys: {sorted(b)}")


if __name__ == "__main__":
    main()
