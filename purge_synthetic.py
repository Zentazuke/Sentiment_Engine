"""One-time cleanup: remove dashboard test injections from the history.
Run with the ENGINE STOPPED: python purge_synthetic.py"""
import sqlite3

conn = sqlite3.connect("sentiment_journal.db")
cursor = conn.execute(
    "DELETE FROM social_history WHERE source IN ('dashboard', 'dashboard-scenario', 'dashboard-baseline', 'manual')"
)
conn.commit()
print(f"removed {cursor.rowcount} synthetic rows")
print("remaining:", conn.execute("SELECT COUNT(*) FROM social_history").fetchone()[0])
conn.close()
