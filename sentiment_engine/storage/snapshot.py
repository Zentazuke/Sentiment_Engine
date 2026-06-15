"""Periodic, consistent snapshot of the live journal into the (synced) project
folder.

The live DB lives on local disk (see config._default_db_dir) precisely so that
cloud-sync never tears its pages. But we still want the data reviewable from the
project folder, so this writes a *consistent* copy using SQLite's online-backup
API (which is safe to run against a database being written), to a temp file that
is then atomically renamed into place — so a reader never sees a half-written
snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def snapshot_db(src_path: str | Path, dst_path: str | Path) -> bool:
    """Write a consistent copy of src_path to dst_path. Returns True on success."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True, timeout=30)
    try:
        dst = sqlite3.connect(tmp_path)
        try:
            with dst:
                src.backup(dst)  # consistent page copy; retries around writers
        finally:
            dst.close()
        os.replace(tmp_path, dst_path)  # atomic on same filesystem
        return True
    except Exception as exc:  # noqa: BLE001 - snapshot must never crash the feed
        logger.warning("snapshot failed (%s)", type(exc).__name__)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False
    finally:
        src.close()


async def snapshot_loop(src_path: str | Path, dst_path: str | Path, interval_seconds: float) -> None:
    """Snapshot every interval_seconds, forever. Non-fatal on errors."""
    logger.info("db snapshot: %s -> %s every %.0fs", src_path, dst_path, interval_seconds)
    while True:
        await asyncio.sleep(interval_seconds)
        ok = await asyncio.to_thread(snapshot_db, src_path, dst_path)
        if ok:
            logger.info("db snapshot written: %s", dst_path)
