"""Small file utilities. No project code or network calls run at import time."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def safe_cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    value = str(value if value is not None else "")
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def write_csv(path, rows, fields=None):
    """Atomic spreadsheet-safe UTF-8 CSV; always write a header, even with no rows."""
    import io
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["record_id"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: safe_cell(row.get(field, "")) for field in fields})
    atomic_text(path, "\ufeff" + stream.getvalue())


def read_csv(path):
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        # Reverse only the spreadsheet escaping that write_csv introduces.
        return [{key: value[1:] if value and value.startswith("'") and
                 value[1:].lstrip().startswith(("=", "+", "-", "@")) else (value or "")
                 for key, value in row.items() if key is not None}
                for row in csv.DictReader(handle)]


def project_path(config, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_project_dir"]) / path).resolve()


@contextmanager
def run_lock(directory):
    """OS-released lock: an interrupted process leaves no stale lock to delete."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / ".lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ValueError("This run is already in use by another process.") from None
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("This run is already in use by another process.") from None
        yield
    finally:
        handle.close()
