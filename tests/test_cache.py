"""
tests/test_cache.py
-------------------
Tests for the QueryCache — no database required.
We mock the psycopg2 connection.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rl_query_optimizer.cache import QueryCache
from rl_query_optimizer.hints import OBS_DIMENSIONS


def make_mock_conn(plan_rows=1000, total_cost=500.0, plan_width=64):
    """Build a mock psycopg2 connection that returns a canned EXPLAIN result."""
    plan = [{
        "Plan": {
            "Plan Rows":  plan_rows,
            "Total Cost": total_cost,
            "Plan Width": plan_width,
        }
    }]
    cursor = MagicMock()
    cursor.fetchone.return_value = [plan]
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


class TestQueryCache:

    def test_feature_vector_shape(self):
        """Feature vector must always be shape (10,)."""
        cache = QueryCache()
        conn  = make_mock_conn()
        feat  = cache.get_features("SELECT * FROM students WHERE age > 20;", conn)
        assert feat.shape == (OBS_DIMENSIONS,)
        assert feat.dtype == np.float32

    def test_feature_vector_cached_on_second_call(self):
        """Second call for the same query should NOT call EXPLAIN again."""
        cache = QueryCache()
        conn  = make_mock_conn()
        sql   = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id;"

        feat1 = cache.get_features(sql, conn)
        calls_after_first = conn.cursor.call_count

        feat2 = cache.get_features(sql, conn)
        calls_after_second = conn.cursor.call_count

        # conn.cursor should not have been called again
        assert calls_after_second == calls_after_first
        np.testing.assert_array_equal(feat1, feat2)

    def test_different_queries_get_different_features(self):
        """Two structurally different queries should produce different vectors."""
        cache = QueryCache()
        conn  = make_mock_conn()

        feat_simple = cache.get_features(
            "SELECT * FROM students WHERE age > 20;", conn
        )
        feat_join = cache.get_features(
            "SELECT s.name FROM students s JOIN courses c ON s.course_id = c.id;", conn
        )
        # They must differ — at minimum the JOIN flag (f[1]) differs
        assert not np.array_equal(feat_simple, feat_join)

    def test_invalidate_single_query(self):
        """Invalidating one query clears only that query's cache entry."""
        cache = QueryCache()
        conn  = make_mock_conn()
        sql1  = "SELECT * FROM students;"
        sql2  = "SELECT * FROM courses;"

        cache.get_features(sql1, conn)
        cache.get_features(sql2, conn)
        assert cache.size == 2

        cache.invalidate(sql1)
        assert cache.size == 1

    def test_invalidate_all(self):
        """Calling invalidate() with no argument clears the entire cache."""
        cache = QueryCache()
        conn  = make_mock_conn()

        for i in range(5):
            cache.get_features(f"SELECT * FROM t{i};", conn)
        assert cache.size == 5

        cache.invalidate()
        assert cache.size == 0

    def test_join_flag_set_correctly(self):
        """Queries with JOIN should have f[1] = 1.0."""
        cache     = QueryCache()
        conn      = make_mock_conn()
        feat_join = cache.get_features(
            "SELECT s.name FROM students s JOIN courses c ON s.course_id = c.id;", conn
        )
        assert feat_join[1] == 1.0

    def test_no_join_flag_not_set(self):
        """Simple queries without JOIN should have f[1] = 0.0."""
        cache        = QueryCache()
        conn         = make_mock_conn()
        feat_no_join = cache.get_features("SELECT * FROM students WHERE age > 20;", conn)
        assert feat_no_join[1] == 0.0