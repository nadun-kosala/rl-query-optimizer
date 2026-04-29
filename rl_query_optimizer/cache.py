"""
cache.py
--------
Query feature cache for the optimizer.

At runtime (not training time) we only need two caches:
  - explain_cache : {query -> {rows, cost, width}} from EXPLAIN
  - feature_cache : {query -> np.ndarray(10,)}   pre-built observation vectors

Both are populated lazily on the first call to execute() or suggest() for
a given query, then reused on all subsequent calls at zero cost.

There is no baseline_cache or warm_exec_cache at inference time — those
are training-only concepts.  The optimizer does not need to measure the
default plan time; it just applies the best-known hint and returns results.
"""

from __future__ import annotations

import re
import time
from typing import Dict, Optional, Tuple

import numpy as np
import psycopg2

from .hints import RESET_HINTS


class QueryCache:
    """
    Lightweight cache for EXPLAIN-derived features.

    Thread safety: not thread-safe by default.  If you use the optimizer
    in a multi-threaded web server, wrap calls in a threading.Lock or
    create one QueryOptimizer instance per thread.
    """

    def __init__(self):
        self._explain_cache: Dict[str, Dict]        = {}
        self._feature_cache: Dict[str, np.ndarray]  = {}

    # ── EXPLAIN ──────────────────────────────────────────────────────────

    def get_explain_info(self, query: str, conn) -> Dict:
        """
        Return EXPLAIN plan info for a query.
        Runs EXPLAIN (FORMAT JSON) on first call; cached thereafter.
        """
        if query in self._explain_cache:
            return self._explain_cache[query]

        result = {"rows": 1, "cost": 0.0, "width": 0}
        try:
            cur = conn.cursor()
            cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
            top = cur.fetchone()[0][0]["Plan"]
            cur.close()
            result = {
                "rows":  top.get("Plan Rows",  1),
                "cost":  top.get("Total Cost", 0.0),
                "width": top.get("Plan Width", 0),
            }
        except Exception:
            pass  # fall back to defaults if EXPLAIN fails

        self._explain_cache[query] = result
        return result

    # ── Features ─────────────────────────────────────────────────────────

    def get_features(self, query: str, conn) -> np.ndarray:
        """
        Return the 10-dimensional feature vector for a query.
        Computes once (using EXPLAIN) then caches.

        Feature dimensions:
          0  FROM/JOIN clause count
          1  has JOIN
          2  has WHERE
          3  has GROUP BY
          4  has ORDER BY
          5  has subquery / IN / EXISTS
          6  AND/OR condition count
          7  log10(plan rows)
          8  log10(total cost)
          9  log10(row width + 1)
        """
        if query in self._feature_cache:
            return self._feature_cache[query]

        q          = query.upper()
        f          = np.zeros(10, dtype=np.float32)
        tables     = len(re.findall(r'\bFROM\b|\bJOIN\b', q))
        conditions = len(re.findall(r'\bAND\b|\bOR\b', q)) + (1 if 'WHERE' in q else 0)

        f[0] = float(min(tables, 10))
        f[1] = 1.0 if 'JOIN'     in q else 0.0
        f[2] = 1.0 if 'WHERE'    in q else 0.0
        f[3] = 1.0 if 'GROUP BY' in q else 0.0
        f[4] = 1.0 if 'ORDER BY' in q else 0.0
        f[5] = 1.0 if any(k in q for k in (' IN (', 'EXISTS')) else 0.0
        f[6] = float(min(conditions, 10))

        info = self.get_explain_info(query, conn)
        f[7] = float(np.log10(max(info["rows"],        1)))
        f[8] = float(np.log10(max(info["cost"],        1)))
        f[9] = float(np.log10(max(info["width"] + 1,   1)))

        self._feature_cache[query] = f
        return f

    # ── Utilities ─────────────────────────────────────────────────────────

    def invalidate(self, query: Optional[str] = None) -> None:
        """
        Remove a query from all caches (or clear everything if query=None).
        Useful when the schema or data distribution changes significantly.
        """
        if query is None:
            self._explain_cache.clear()
            self._feature_cache.clear()
        else:
            self._explain_cache.pop(query, None)
            self._feature_cache.pop(query, None)

    @property
    def size(self) -> int:
        """Number of queries currently cached."""
        return len(self._feature_cache)

    def stats(self) -> Dict:
        """Return cache statistics for debugging."""
        return {
            "cached_queries":  self.size,
            "explain_entries": len(self._explain_cache),
            "feature_entries": len(self._feature_cache),
        }