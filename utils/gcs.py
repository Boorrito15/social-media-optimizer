"""Google Cloud Storage helpers.

Uploads files to GCS using the `google-cloud-storage` library when available
(preferred — works with just the service-account key / ADC, no CLI needed, so
it runs on lean processing machines like the Mac mini), falling back to the
`gsutil` CLI otherwise.

Features:
* skip-if-exists (re-runs don't re-upload finished work)
* retry with back-off for transient failures
* content-type inference for common media types
* ``dry_run`` mode that only prints what would be uploaded
"""

from __future__ import annotations

import hashlib
import mimetypes
import subprocess
import time
from pathlib import Path

from utils.config import gcp_project_id, gcs_processed_bucket


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 of a file (streamed, low memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".parquet": "application/vnd.apache.parquet",
    ".json": "application/json",
    ".csv": "text/csv",
}


def _content_type(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _CONTENT_TYPES:
        return _CONTENT_TYPES[ext]
    guess, _ = mimetypes.guess_type(str(path))
    return guess


def library_available() -> bool:
    """Return True if the ``google-cloud-storage`` library is installed."""
    try:
        import google.cloud.storage  # noqa: F401

        return True
    except ImportError:
        return False


def _client():
    """Return a ``google.cloud.storage.Client`` authenticated from the service key."""
    from google.cloud import storage  # type: ignore[import-not-found]

    from utils.config import service_account_credentials

    creds_path = service_account_credentials()
    if creds_path and creds_path.is_file():
        return storage.Client.from_service_account_json(str(creds_path))
    return storage.Client()


def object_exists(bucket: str, object_name: str, *, project_id: str | None = None) -> bool:
    """Return True if ``gs://bucket/object_name`` already exists."""
    if library_available():
        try:
            return _client().bucket(bucket).blob(object_name).exists()
        except Exception:
            pass  # fall back to gsutil
    uri = f"gs://{bucket}/{object_name}"
    try:
        res = subprocess.run(
            ["gsutil", "-q", "stat", uri],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0
    except FileNotFoundError:
        return False


def _upload_via_library(
    local: Path,
    bucket: str,
    object_name: str,
    *,
    skip_if_exists: bool,
    content_type: str | None,
    max_retries: int,
    dry_run: bool,
) -> bool:
    """Upload a single file using ``google-cloud-storage``."""
    uri = f"gs://{bucket}/{object_name}"
    if dry_run:
        print(f"[gcs][dry-run] would upload {local} -> {uri}")
        return True
    if skip_if_exists and object_exists(bucket, object_name):
        print(f"[gcs] already exists, skipping: {uri}")
        return True

    ctype = content_type or _content_type(local)
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            client = _client()
            blob = client.bucket(bucket).blob(object_name)
            if ctype:
                blob.content_type = ctype
            blob.upload_from_filename(str(local))
            print(f"[gcs] uploaded {local} -> {uri}")
            return True
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"[gcs] attempt {attempt} failed; retrying in {wait}s: {exc}")
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _upload_via_gsutil(
    local: Path,
    bucket: str,
    object_name: str,
    *,
    skip_if_exists: bool,
    max_retries: int,
    dry_run: bool,
) -> bool:
    """Upload a single file using the ``gsutil`` CLI (fallback)."""
    project_id = gcp_project_id()
    uri = f"gs://{bucket}/{object_name}"
    if dry_run:
        print(f"[gcs][dry-run] would upload {local} -> {uri}")
        return True
    if skip_if_exists and object_exists(bucket, object_name, project_id=project_id):
        print(f"[gcs] already exists, skipping: {uri}")
        return True

    cmd = ["gsutil", "cp", "-n", str(local), uri]
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                "Neither google-cloud-storage nor gsutil is available. "
                "Install one of them to upload to GCS."
            ) from None
        if res.returncode == 0:
            print(f"[gcs] uploaded {local} -> {uri}")
            return True
        last_err = RuntimeError(res.stderr.strip() or res.stdout.strip() or "gsutil failed")
        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"[gcs] attempt {attempt} failed; retrying in {wait}s: {last_err}")
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def upload_file(
    local_path: str | Path,
    bucket: str | None = None,
    object_name: str | None = None,
    *,
    skip_if_exists: bool = True,
    dry_run: bool = False,
    max_retries: int = 3,
    content_type: str | None = None,
) -> bool:
    """Upload ``local_path`` to ``gs://<bucket>/<object_name>``.

    Uses ``google-cloud-storage`` when installed, else ``gsutil``. Returns
    ``True`` if the object is present in GCS afterwards (uploaded or already
    existed and skipped). Raises on a hard failure after retries.
    """
    bucket = bucket or gcs_processed_bucket()
    local = Path(local_path).expanduser().resolve()
    if not local.is_file():
        raise FileNotFoundError(f"Local file not found: {local}")
    object_name = object_name or local.name

    if library_available():
        return _upload_via_library(
            local, bucket, object_name,
            skip_if_exists=skip_if_exists,
            content_type=content_type,
            max_retries=max_retries,
            dry_run=dry_run,
        )
    return _upload_via_gsutil(
        local, bucket, object_name,
        skip_if_exists=skip_if_exists,
        max_retries=max_retries,
        dry_run=dry_run,
    )
