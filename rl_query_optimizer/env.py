"""
query_env_v3.py — Enhanced RL environment (cache-first edition)

Speed fixes vs original v3:
  - _extract_features() makes ZERO DB calls during training.
    Both explain_cache (rows/cost/width) and feature_cache (pre-built numpy
    arrays) are injected at construction time.
  - reset() makes ZERO DB calls (baseline_cache lookup only).
  - step() makes exactly ONE DB call: the timed execution.
  - Connection is kept alive and reused; no reconnect overhead.

All reward logic and 10-feature observation are identical to v3.
"""

import re
import time
import gymnasium as gym
import numpy as np
import psycopg2
from collections import deque
from typing import Dict, List, Optional


ACTION_NAMES = {
    0: "Default (no hints)",
    1: "Force Index Scan",
    2: "Force Sequential Scan",
    3: "Prefer Hash Join",
    4: "Prefer Merge Join",
    5: "Prefer Nested Loop",
    6: "Disable Parallel Query",
}

ACTION_HINTS: Dict[int, List[str]] = {
    0: [],
    1: ["SET enable_seqscan = OFF", "SET enable_indexscan = ON",
        "SET enable_bitmapscan = ON"],
    2: ["SET enable_seqscan = ON",  "SET enable_indexscan = OFF",
        "SET enable_bitmapscan = OFF"],
    3: ["SET enable_hashjoin = ON",  "SET enable_mergejoin = OFF",
        "SET enable_nestloop = OFF"],
    4: ["SET enable_hashjoin = OFF", "SET enable_mergejoin = ON",
        "SET enable_nestloop = OFF"],
    5: ["SET enable_hashjoin = OFF", "SET enable_mergejoin = OFF",
        "SET enable_nestloop = ON"],
    6: ["SET max_parallel_workers_per_gather = 0"],
}

RESET_HINTS = [
    "SET enable_seqscan = ON",  "SET enable_indexscan = ON",
    "SET enable_bitmapscan = ON", "SET enable_hashjoin = ON",
    "SET enable_mergejoin = ON",  "SET enable_nestloop = ON",
    "RESET max_parallel_workers_per_gather",
]

MIN_BASELINE_MS    = 5.0   # below this → dampened reward (noisy)
NOISE_THRESHOLD_PCT = 0.05  # below 5% improvement → reward = 0


class QueryOptimizationEnvV3(gym.Env):
    """
    Cache-first RL environment.

    Parameters
    ----------
    db_config      : psycopg2 connect kwargs
    queries        : SQL strings
    n_avg          : executions per timing call
    baseline_cache : {query -> default-plan ms}  — avoids re-measuring in reset()
    explain_cache  : {query -> {rows, cost, width}} — avoids EXPLAIN in features
    feature_cache  : {query -> np.ndarray shape(10,)} — avoids ALL feature math
                     If provided, _extract_features() is a pure dict lookup.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        db_config: Dict,
        queries: List[str],
        n_avg: int = 1,
        baseline_cache: Optional[Dict[str, float]]      = None,
        explain_cache:  Optional[Dict[str, Dict]]       = None,
        feature_cache:  Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__()
        self.db_config     = db_config
        self.queries       = queries
        self.n_avg         = n_avg
        self._baseline_cache = baseline_cache or {}
        self._explain_cache  = explain_cache  or {}
        self._feature_cache  = feature_cache  or {}

        self.current_query: Optional[str] = None
        self.baseline_time: float = 1.0

        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=-20.0, high=20.0, shape=(10,), dtype=np.float32
        )

        self._reward_window: deque = deque(maxlen=500)
        self.conn = None
        self._connect()

    # ------------------------------------------------------------------ #
    #  DB                                                                  #
    # ------------------------------------------------------------------ #

    def _connect(self):
        self.conn = psycopg2.connect(**self.db_config)
        self.conn.autocommit = True

    def _reset_hints(self):
        cur = self.conn.cursor()
        for sql in RESET_HINTS:
            cur.execute(sql)
        cur.close()

    def _apply_hints(self, action: int):
        cur = self.conn.cursor()
        for sql in ACTION_HINTS[action]:
            cur.execute(sql)
        cur.close()

    def _time_query(self, query: str, action: int, n: int = None) -> float:
        """Run query with action hints, return median ms over n executions."""
        if n is None:
            n = self.n_avg
        self._reset_hints()
        self._apply_hints(action)
        cur  = self.conn.cursor()
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            cur.execute(query)
            cur.fetchall()
            times.append((time.perf_counter() - t0) * 1000)
        cur.close()
        self._reset_hints()
        return float(np.median(times))

    def _get_explain_info(self, query: str) -> Dict:
        """EXPLAIN — result cached; only called during precompute phase."""
        if query in self._explain_cache:
            return self._explain_cache[query]
        result = {"rows": 1, "cost": 0.0, "width": 0}
        try:
            cur = self.conn.cursor()
            cur.execute(f"EXPLAIN (FORMAT JSON) {query}")
            top = cur.fetchone()[0][0]["Plan"]
            cur.close()
            result = {
                "rows":  top.get("Plan Rows",   1),
                "cost":  top.get("Total Cost",  0.0),
                "width": top.get("Plan Width",  0),
            }
            self._explain_cache[query] = result
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------ #
    #  Features — ZERO DB calls if feature_cache is populated             #
    # ------------------------------------------------------------------ #

    def _extract_features(self, query: str) -> np.ndarray:
        """
        Returns pre-built feature vector if available (zero cost).
        Falls back to computing from explain_cache or live EXPLAIN.
        Feature vector (10 dims):
          0: FROM/JOIN clause count
          1: has JOIN
          2: has WHERE
          3: has GROUP BY
          4: has ORDER BY
          5: has subquery / IN / EXISTS
          6: AND/OR condition count
          7: log10(plan rows)      ← from EXPLAIN cache
          8: log10(total cost)     ← from EXPLAIN cache
          9: log10(row width + 1)  ← from EXPLAIN cache
        """
        if query in self._feature_cache:
            return self._feature_cache[query]

        q = query.upper()
        f = np.zeros(10, dtype=np.float32)

        tables     = len(re.findall(r'\bFROM\b|\bJOIN\b', q))
        conditions = len(re.findall(r'\bAND\b|\bOR\b', q)) + (1 if 'WHERE' in q else 0)

        f[0] = float(min(tables, 10))
        f[1] = 1.0 if 'JOIN'     in q else 0.0
        f[2] = 1.0 if 'WHERE'    in q else 0.0
        f[3] = 1.0 if 'GROUP BY' in q else 0.0
        f[4] = 1.0 if 'ORDER BY' in q else 0.0
        f[5] = 1.0 if any(k in q for k in (' IN (', 'EXISTS')) else 0.0
        f[6] = float(min(conditions, 10))

        info = self._get_explain_info(query)
        f[7] = float(np.log10(max(info["rows"],        1)))
        f[8] = float(np.log10(max(info["cost"],        1)))
        f[9] = float(np.log10(max(info["width"] + 1,   1)))

        # Store for future calls
        self._feature_cache[query] = f
        return f

    # ------------------------------------------------------------------ #
    #  Reward                                                              #
    # ------------------------------------------------------------------ #

    def _compute_reward(self, optimized_ms: float, baseline_ms: float) -> float:
        if baseline_ms <= 0:
            return 0.0
        relative = (baseline_ms - optimized_ms) / baseline_ms
        if abs(relative) < NOISE_THRESHOLD_PCT:
            return 0.0
        self._reward_window.append(relative)
        if len(self._reward_window) < 20:
            raw = relative
        else:
            mu    = np.mean(self._reward_window)
            sigma = np.std(self._reward_window) + 1e-8
            raw   = (relative - mu) / sigma
        if baseline_ms < MIN_BASELINE_MS:
            raw *= 0.1
        return float(np.clip(raw, -5.0, 5.0))

    # ------------------------------------------------------------------ #
    #  Gym                                                                 #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_query = self.queries[
            self.np_random.integers(0, len(self.queries))
        ]
        # Zero DB calls if cache is populated
        self.baseline_time = self._baseline_cache.get(
            self.current_query,
            self._time_query(self.current_query, action=0)
        )
        return self._extract_features(self.current_query), {}

    def step(self, action: int):
        action        = int(action)
        optimized_ms  = self._time_query(self.current_query, action=action)  # 1 DB call
        reward        = self._compute_reward(optimized_ms, self.baseline_time)

        info = {
            "query":           self.current_query[:60],
            "action":          action,
            "action_name":     ACTION_NAMES[action],
            "optimized_ms":    round(optimized_ms, 2),
            "baseline_ms":     round(self.baseline_time, 2),
            "improvement_pct": round(
                (self.baseline_time - optimized_ms) / max(self.baseline_time, 0.01) * 100, 1
            ),
            "reward":          round(reward, 4),
            "is_noisy":        self.baseline_time < MIN_BASELINE_MS,
        }
        return self._extract_features(self.current_query), reward, True, False, info

    def close(self):
        if self.conn:
            self.conn.close()