"""
optimizer.py
------------
The main public class: QueryOptimizer.

This is the only class most developers will ever interact with.

Quick start
-----------
    from rl_query_optimizer import QueryOptimizer

    optimizer = QueryOptimizer(db_config={
        "host": "localhost", "database": "mydb",
        "user": "admin",     "password": "secret",
    })

    # Run a query with automatic plan optimization
    results = optimizer.execute("SELECT * FROM orders JOIN customers ON ...")

    # Just get the recommendation without running
    suggestion = optimizer.suggest("SELECT * FROM orders JOIN customers ON ...")

    # Swap to a better model when available — no restart needed
    optimizer.replace_model("path/to/improved_model.zip")
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import psycopg2

from .cache import QueryCache
from .execptions import (
    DatabaseConnectionError,
    ModelNotLoadedError,
    QueryExecutionError,
)
from .hints import ACTION_HINTS, ACTION_NAMES, RESET_HINTS
from .model_loader import get_model_info, load_model


class QueryOptimizer:
    """
    RL-based PostgreSQL query optimizer.

    Wraps a trained Stable Baselines3 model (PPO or DQN) and applies
    it to real queries at inference time.  The model is frozen — it does
    not update itself during use.  Call replace_model() to upgrade to a
    better model without restarting.

    Parameters
    ----------
    db_config : dict
        psycopg2 connection keyword arguments.
        Required keys: host, database, user, password.
        Optional keys: port (default 5432), sslmode, connect_timeout.
    model_path : str | Path | None
        Path to a .zip Stable Baselines3 model file.
        If None (default), the bundled default model is used.
    verbose : bool
        If True, prints the chosen action and timing for each query.
        Default False.
    auto_cache : bool
        If True (default), EXPLAIN info and feature vectors are cached
        after the first call for each unique query.  Subsequent calls
        to the same query are significantly faster.
    """

    def __init__(
        self,
        db_config:  Dict[str, Any],
        model_path: Optional[Union[str, Path]] = None,
        verbose:    bool = False,
        auto_cache: bool = True,
    ):
        self._db_config  = db_config
        self._verbose    = verbose
        self._auto_cache = auto_cache
        self._lock       = threading.Lock()  # protects model swap

        # Connect to database
        self._conn = self._connect(db_config)

        # Load model (bundled default if no path given)
        self._model = load_model(model_path)
        self._model_path = Path(model_path) if model_path else None

        # Query feature cache
        self._cache = QueryCache()

        if verbose:
            info = get_model_info(self._model)
            print(f"[RLOptimizer] Loaded {info['algorithm']} model  "
                  f"obs={info['observation_shape']}  "
                  f"actions={info['num_actions']}")

    # ── Connection ────────────────────────────────────────────────────────

    @staticmethod
    def _connect(db_config: Dict) -> "psycopg2.connection":
        """Connect to PostgreSQL. Raises DatabaseConnectionError on failure."""
        try:
            conn = psycopg2.connect(**db_config)
            conn.autocommit = True
            return conn
        except Exception as e:
            host = db_config.get("host", "unknown")
            port = db_config.get("port", 5432)
            raise DatabaseConnectionError(
                f"Cannot connect to PostgreSQL at {host}:{port}.\n"
                f"Check your db_config values.\nUnderlying error: {e}"
            ) from e

    def _ensure_connection(self) -> None:
        """Reconnect if the connection has been dropped."""
        try:
            self._conn.cursor().execute("SELECT 1")
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = self._connect(self._db_config)

    # ── Hint management ───────────────────────────────────────────────────

    def _reset_hints(self) -> None:
        cur = self._conn.cursor()
        for sql in RESET_HINTS:
            cur.execute(sql)
        cur.close()

    def _apply_hints(self, action: int) -> None:
        cur = self._conn.cursor()
        for sql in ACTION_HINTS[action]:
            cur.execute(sql)
        cur.close()

    # ── Model prediction ──────────────────────────────────────────────────

    def _predict_action(self, query: str) -> int:
        """Extract features and run the frozen policy to pick an action."""
        if self._model is None:
            raise ModelNotLoadedError(
                "No model is loaded. Call replace_model() with a valid path."
            )

        features = self._cache.get_features(query, self._conn)

        with self._lock:
            action, _ = self._model.predict(features, deterministic=True)

        return int(action)

    # ── Public API ────────────────────────────────────────────────────────

    def suggest(self, sql: str) -> Dict[str, Any]:
        """
        Return the optimizer's recommendation for a query without executing it.

        Parameters
        ----------
        sql : str
            A SQL SELECT statement.

        Returns
        -------
        dict with keys:
            action      (int)   action ID (0–6)
            name        (str)   human-readable action name
            hints       (list)  the PostgreSQL SET commands that would be applied
            confidence  (float) max logit - min logit (higher = more decisive)

        Example
        -------
        >>> optimizer.suggest("SELECT * FROM orders JOIN customers ON ...")
        {
            "action": 3,
            "name": "Prefer Hash Join",
            "hints": ["SET enable_hashjoin = ON", ...],
            "confidence": 8.42
        }
        """
        self._ensure_connection()
        action = self._predict_action(sql)

        # Compute confidence (logit spread)
        confidence = 0.0
        try:
            import torch
            features = self._cache.get_features(sql, self._conn)
            obs_t = torch.tensor(features[None], dtype=torch.float32)
            with torch.no_grad():
                # Works for both PPO (distribution logits) and DQN (Q-values)
                try:
                    dist = self._model.policy.get_distribution(obs_t)
                    logits = dist.distribution.logits.numpy()[0]
                except AttributeError:
                    logits = self._model.q_net(obs_t).numpy()[0]
            confidence = round(float(logits.max() - logits.min()), 4)
        except Exception:
            pass

        return {
            "action":     action,
            "name":       ACTION_NAMES[action],
            "hints":      ACTION_HINTS[action],
            "confidence": confidence,
        }

    def execute(
        self,
        sql:     str,
        params:  Optional[tuple] = None,
        compare: bool            = False,
    ) -> Dict[str, Any]:
        """
        Execute a SQL query using the optimizer's recommended plan.

        Parameters
        ----------
        sql : str
            A SQL SELECT statement.
        params : tuple | None
            Optional parameterized query values, passed to cursor.execute().
        compare : bool
            If True, also runs the query with the default plan and includes
            the comparison in the result.  Adds one extra DB round-trip.
            Default False.

        Returns
        -------
        dict with keys:
            rows            (list)   query result rows
            action          (int)    action applied
            action_name     (str)    human-readable name
            optimized_ms    (float)  execution time with optimizer hints
            default_ms      (float)  execution time without hints (only if compare=True)
            improvement_pct (float)  % improvement vs default   (only if compare=True)
            cached          (bool)   whether features were served from cache

        Raises
        ------
        QueryExecutionError
            If the query fails to execute.

        Example
        -------
        >>> result = optimizer.execute("SELECT * FROM orders JOIN customers ON ...")
        >>> print(result["rows"])
        >>> print(f"Used: {result['action_name']}  ({result['optimized_ms']:.1f}ms)")
        """
        self._ensure_connection()

        cached_before = self._cache.size
        action        = self._predict_action(sql)
        was_cached    = self._cache.size == cached_before

        # Execute with optimizer hints
        try:
            self._reset_hints()
            self._apply_hints(action)
            cur = self._conn.cursor()
            t0  = time.perf_counter()
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            rows          = cur.fetchall()
            optimized_ms  = (time.perf_counter() - t0) * 1000
            cur.close()
            self._reset_hints()
        except Exception as e:
            self._reset_hints()
            raise QueryExecutionError(
                f"Query execution failed.\nSQL: {sql[:120]}\nError: {e}"
            ) from e

        result = {
            "rows":         rows,
            "action":       action,
            "action_name":  ACTION_NAMES[action],
            "optimized_ms": round(optimized_ms, 3),
            "cached":       was_cached,
        }

        if self._verbose:
            print(f"[RLOptimizer] action=[{action}] {ACTION_NAMES[action]}  "
                  f"{optimized_ms:.2f}ms  "
                  f"{'(cached)' if was_cached else '(new query)'}")

        # Optional comparison with default plan
        if compare:
            try:
                cur = self._conn.cursor()
                t0  = time.perf_counter()
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
                cur.fetchall()
                default_ms = (time.perf_counter() - t0) * 1000
                cur.close()

                improvement = ((default_ms - optimized_ms) / max(default_ms, 0.01)) * 100
                result["default_ms"]      = round(default_ms, 3)
                result["improvement_pct"] = round(improvement, 2)
            except Exception:
                pass  # comparison failure should not break the main result

        return result

    def replace_model(self, model_path: Union[str, Path]) -> None:
        """
        Hot-swap the optimizer's model without restarting.

        The new model is validated before the old one is replaced.
        If validation fails, the original model keeps running and a
        ModelValidationError is raised.

        Parameters
        ----------
        model_path : str | Path
            Path to a .zip Stable Baselines3 PPO or DQN model file.

        Raises
        ------
        ModelValidationError
            If the new model has incompatible dimensions.

        Example
        -------
        >>> optimizer.replace_model("models/improved_model_v2.zip")
        [RLOptimizer] Model replaced: PPO  obs=(10,)  actions=7
        """
        new_model = load_model(model_path)  # validates before touching _model

        with self._lock:
            self._model      = new_model
            self._model_path = Path(model_path)

        # Clear the feature cache — new model may use features differently
        self._cache.invalidate()

        if self._verbose:
            info = get_model_info(new_model)
            print(f"[RLOptimizer] Model replaced: {info['algorithm']}  "
                  f"obs={info['observation_shape']}  "
                  f"actions={info['num_actions']}")

    def invalidate_cache(self, sql: Optional[str] = None) -> None:
        """
        Clear cached EXPLAIN info and feature vectors.

        Call this after major schema changes (new indexes, ANALYZE, etc.)
        so the optimizer re-computes features with fresh plan estimates.

        Parameters
        ----------
        sql : str | None
            If given, invalidates only that query's cache entry.
            If None (default), clears the entire cache.
        """
        self._cache.invalidate(sql)

    @property
    def model_info(self) -> Dict:
        """Return information about the currently loaded model."""
        if self._model is None:
            return {"status": "no model loaded"}
        info = get_model_info(self._model)
        info["path"] = str(self._model_path) if self._model_path else "bundled default"
        return info

    @property
    def cache_stats(self) -> Dict:
        """Return cache statistics."""
        return self._cache.stats()

    def close(self) -> None:
        """Close the database connection. Call when done with the optimizer."""
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        algo = type(self._model).__name__ if self._model else "None"
        return (f"QueryOptimizer(algorithm={algo}, "
                f"cached_queries={self._cache.size}, "
                f"verbose={self._verbose})")