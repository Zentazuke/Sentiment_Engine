"""DB snapshot: consistency + atomic write."""

import sqlite3
from pathlib import Path

from sentiment_engine.storage.snapshot import snapshot_db


def _make_db(path: Path, rows: int) -> None:
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, v REAL)")
    c.executemany("INSERT INTO t(v) VALUES (?)", [(float(i),) for i in range(rows)])
    c.commit()
    c.close()


def test_snapshot_copies_all_rows(tmp_path):
    src = tmp_path / "live.db"
    dst = tmp_path / "synced" / "snap.db"
    _make_db(src, 500)
    assert snapshot_db(src, dst) is True
    c = sqlite3.connect(dst)
    assert c.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 500
    assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    c.close()


def test_snapshot_missing_source_returns_false(tmp_path):
    assert snapshot_db(tmp_path / "nope.db", tmp_path / "out.db") is False


def test_snapshot_leaves_no_tmp_file(tmp_path):
    src = tmp_path / "live.db"
    dst = tmp_path / "snap.db"
    _make_db(src, 10)
    snapshot_db(src, dst)
    assert not (tmp_path / "snap.db.tmp").exists()
    assert dst.exists()


def test_snapshot_overwrites_existing(tmp_path):
    src = tmp_path / "live.db"
    dst = tmp_path / "snap.db"
    _make_db(src, 3)
    snapshot_db(src, dst)
    # grow source, snapshot again -> dst reflects new state
    c = sqlite3.connect(src)
    c.executemany("INSERT INTO t(v) VALUES (?)", [(1.0,)] * 7)
    c.commit()
    c.close()
    snapshot_db(src, dst)
    c = sqlite3.connect(dst)
    assert c.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 10
    c.close()
