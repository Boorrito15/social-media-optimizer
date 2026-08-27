"""Centralised environment / configuration loading.

Loads variables from a `.env` file (via python-dotenv) and exposes typed
getters so the rest of the codebase reads configuration from one place instead
of scattering `os.getenv` calls. All credentials referenced here must already
be git-ignored (see .gitignore).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Project root = two levels up from this file (utils/ -> project root).
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


def _dotenv_lines() -> list[str]:
    """Return the raw lines of .env (non-empty, comments stripped), if present."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return []
    try:
        return [
            line.strip()
            for line in env_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    except OSError:
        return []


def _missing(name: str) -> None:
    raise RuntimeError(
        f"Missing required environment variable '{name}'. "
        f"Copy .env.example to .env and set it."
    )


def get(key: str, default: str | None = None) -> str:
    """Return an environment variable as a trimmed string."""
    value = os.getenv(key)
    if value is None or value.strip() == "":
        if default is not None:
            return default
        _missing(key)
    return value.strip()


def get_int(key: str, default: int | None = None) -> int:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        if default is not None:
            return default
        _missing(key)
    return int(value)


def get_float(key: str, default: float | None = None) -> float:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        if default is not None:
            return default
        _missing(key)
    return float(value)


def get_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_optional(key: str) -> str | None:
    value = os.getenv(key)
    return value.strip() if value and value.strip() else None


def resolve_path(key: str, default: str | None = None) -> Path:
    """Return a path resolved relative to the project root."""
    raw = get(key, default)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# --- Grouped convenience accessors -----------------------------------------

def gcp_project_id() -> str:
    return get("GCP_PROJECT_ID")


def gcp_region() -> str:
    return get("GCP_REGION", "asia-southeast2")


def gcp_zone() -> str:
    return get("GCP_ZONE", f"{gcp_region()}-a")


def gcs_raw_bucket() -> str:
    return get("GCS_RAW_BUCKET", "sm-optimizer-raw")


def gcs_processed_bucket() -> str:
    return get("GCS_PROCESSED_BUCKET", "sm-optimizer-processed")


def service_account_credentials() -> Path | None:
    """Return the service-account key path, preferring the project's own key.

    The value normally comes from ``GOOGLE_APPLICATION_CREDENTIALS``
    (``os.getenv``), which is also how the ``google-cloud-storage`` and the
    GenAI SDK discover ADC. However, an *ambient* shell-exported value (e.g. a
    leftover path in ``~/Downloads``) can shadow the project's intended key from
    ``.env``. To guarantee the pipeline always authenticates with the project's
    key, we first look for a ``.env``-declared value (read from the file, not
    the process env) and fall back to the environment variable.
    """
    # 1) Prefer the key declared in this project's .env file (survives any
    #    stale shell-exported GOOGLE_APPLICATION_CREDENTIALS).
    env_value = next(
        (line.split("=", 1)[1].strip().strip('"\'')
         for line in _dotenv_lines()
         if line.startswith("GOOGLE_APPLICATION_CREDENTIALS=") and "=" in line),
        None,
    )
    if env_value:
        p = Path(env_value).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    # 2) Fall back to the ambient environment variable.
    raw = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def gemini_api_key() -> str | None:
    return get_optional("GEMINI_API_KEY")


def gemini_model() -> str:
    return get("GEMINI_MODEL", "models/gemini-2.0-flash")


def video_output_dir() -> str:
    return get("VIDEO_OUTPUT_DIR", "data/videos")


def video_gcs_prefix() -> str:
    return get("VIDEO_GCS_PREFIX", "videos")


def video_target_height() -> int:
    return get_int("VIDEO_TARGET_HEIGHT", 480)


def video_codec() -> str:
    return get("VIDEO_CODEC", "h264").lower()


def video_crf() -> int:
    return get_int("VIDEO_CRF", 23)


def ytdlp_cookies() -> str | None:
    return get_optional("YTDLP_COOKIES")


def video_concurrency() -> int:
    return get_int("VIDEO_CONCURRENCY", 1)


def video_retries() -> int:
    return get_int("VIDEO_RETRIES", 2)


def video_request_delay() -> float:
    """Seconds to sleep before each download request (rate limiting).

    Helps stay under a platform's throttling threshold when scraping
    anonymously. Default 0 (no delay); set a positive value to pace requests.
    """
    return get_float("VIDEO_REQUEST_DELAY", 0.0)
