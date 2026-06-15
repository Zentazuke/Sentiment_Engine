"""Test isolation.

The engine's SQLite journal defaults to a fixed local path. Without isolation,
every test run appends to the SAME database, so social/outlook rows accumulate
across runs and leak between tests (e.g. piled-up events dilute the shock test).

Point the journal (and its snapshot) at a throwaway temp directory before any
engine module is imported, so each pytest session starts from a clean database.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="sentiment_test_journal_")
os.environ["SENTIMENT_JOURNAL_DB"] = os.path.join(_tmp, "journal.db")
os.environ["SENTIMENT_SNAPSHOT_DB"] = os.path.join(_tmp, "snapshot.db")
