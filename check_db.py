"""Quick local health check of the live journal. Run: python check_db.py

Reads the live DB at config.JOURNAL_DB_PATH (now on local disk, not the synced
folder). Pass a path to inspect a specific file, e.g. the project-folder
snapshot:  python check_db.py sentiment_journal_snapshot.db
"""
import sqlite3
import sys
import time

from sentiment_engine.config import JOURNAL_DB_PATH

db_path = sys.argv[1] if len(sys.argv) > 1 else JOURNAL_DB_PATH
print(f"db: {db_path}")
conn = sqlite3.connect(db_path)
now = time.time()
print("integrity:", conn.execute("PRAGMA integrity_check").fetchone()[0])
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("tables:", ", ".join(tables))
for table, ts in [("social_history", "timestamp"), ("micro_history", "received_at"),
                  ("outlook_history", "computed_at"), ("evaluations", "timestamp")]:
    if table not in tables:
        print(f"{table}: missing")
        continue
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    newest = conn.execute(f"SELECT MAX({ts}) FROM {table}").fetchone()[0]
    age = f"{(now - newest) / 60:.1f} min ago" if newest else "never"
    print(f"{table}: {total} rows, newest {age}")
if "social_history" in tables:
    print("\nsocial events by source (last 24h):")
    rows = conn.execute(
        "SELECT source, COUNT(*) FROM social_history WHERE timestamp > ?"
        " GROUP BY source ORDER BY 2 DESC LIMIT 12", (now - 86400,)).fetchall()
    for source, count in rows:
        print(f"  {source}: {count}")
conn.close()
