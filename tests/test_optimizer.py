"""
tests/test_optimizer.py
-----------------------
Integration tests for QueryOptimizer using mocked DB and model.
No real database or model file required.
"""

from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

from rl_query_optimizer.execptions import (
    DatabaseConnectionError,
    ModelValidationError,
    QueryExecutionError,
)
from rl_query_optimizer.hints import NUM_ACTIONS, OBS_DIMENSIONS
from rl_query_optimizer.optimizer import QueryOptimizer


def _make_mock_model(action=3):
    """Minimal mock that behaves like a loaded PPO/DQN model."""
    model = MagicMock()
    model.predict.return_value = (action, None)
    model.observation_space.shape = (OBS_DIMENSIONS,)
    model.action_space.n          = NUM_ACTIONS
    return model


def _make_mock_conn():
    """Mock psycopg2 connection that returns fake query rows and EXPLAIN data."""
    plan = [{"Plan": {"Plan Rows": 5000, "Total Cost": 120.5, "Plan Width": 32}}]
    cursor = MagicMock()
    cursor.fetchone.return_value  = [plan]
    cursor.fetchall.return_value  = [("Alice", "Math"), ("Bob", "Physics")]
    conn   = MagicMock()
    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def optimizer():
    """Create a QueryOptimizer with fully mocked DB and model."""
    mock_model = _make_mock_model(action=3)
    mock_conn  = _make_mock_conn()

    with patch("rl_query_optimizer.optimizer.psycopg2.connect", return_value=mock_conn), \
         patch("rl_query_optimizer.optimizer.load_model",        return_value=mock_model):
        opt = QueryOptimizer(
            db_config={"host": "localhost", "database": "test",
                       "user": "u", "password": "p"},
            verbose=False,
        )
    return opt


class TestExecute:

    def test_returns_rows(self, optimizer):
        result = optimizer.execute("SELECT name FROM students;")
        assert "rows" in result
        assert isinstance(result["rows"], list)

    def test_returns_action_info(self, optimizer):
        result = optimizer.execute("SELECT name FROM students;")
        assert "action"      in result
        assert "action_name" in result
        assert isinstance(result["action"],      int)
        assert isinstance(result["action_name"], str)

    def test_returns_timing(self, optimizer):
        result = optimizer.execute("SELECT name FROM students;")
        assert "optimized_ms" in result
        assert result["optimized_ms"] >= 0

    def test_same_query_uses_cache(self, optimizer):
        sql = "SELECT name FROM students WHERE age > 20;"
        optimizer.execute(sql)
        first_size  = optimizer.cache_stats["cached_queries"]
        optimizer.execute(sql)
        second_size = optimizer.cache_stats["cached_queries"]
        assert first_size == second_size  # cache size did not grow


class TestSuggest:

    def test_returns_action_dict(self, optimizer):
        result = optimizer.suggest("SELECT * FROM orders JOIN customers ON ...",)
        assert "action"  in result
        assert "name"    in result
        assert "hints"   in result
        assert isinstance(result["action"], int)
        assert 0 <= result["action"] <= 6

    def test_deterministic_same_query(self, optimizer):
        sql = "SELECT * FROM orders JOIN customers ON orders.cid = customers.id;"
        r1  = optimizer.suggest(sql)
        r2  = optimizer.suggest(sql)
        assert r1["action"] == r2["action"]


class TestReplaceModel:

    def test_replace_with_valid_model_succeeds(self, optimizer):
        new_model = _make_mock_model(action=5)
        with patch("rl_query_optimizer.optimizer.load_model", return_value=new_model):
            optimizer.replace_model("new_model.zip")
        assert optimizer._model is new_model

    def test_replace_clears_cache(self, optimizer):
        optimizer.execute("SELECT * FROM students;")
        assert optimizer.cache_stats["cached_queries"] > 0

        new_model = _make_mock_model(action=1)
        with patch("rl_query_optimizer.optimizer.load_model", return_value=new_model):
            optimizer.replace_model("new_model.zip")

        assert optimizer.cache_stats["cached_queries"] == 0

    def test_replace_with_invalid_model_raises(self, optimizer):
        with patch("rl_query_optimizer.optimizer.load_model",
                   side_effect=ModelValidationError("bad dims")):
            with pytest.raises(ModelValidationError):
                optimizer.replace_model("bad_model.zip")

    def test_original_model_preserved_after_failed_replace(self, optimizer):
        original = optimizer._model
        with patch("rl_query_optimizer.optimizer.load_model",
                   side_effect=ModelValidationError("bad")):
            try:
                optimizer.replace_model("bad.zip")
            except ModelValidationError:
                pass
        assert optimizer._model is original


class TestContextManager:

    def test_context_manager_closes_connection(self):
        mock_model = _make_mock_model()
        mock_conn  = _make_mock_conn()

        with patch("rl_query_optimizer.optimizer.psycopg2.connect", return_value=mock_conn), \
             patch("rl_query_optimizer.optimizer.load_model",        return_value=mock_model):
            with QueryOptimizer(
                db_config={
                    "host": "localhost",
                    "database": "test",
                    "user": "u",
                    "password": "p"}
            ) as opt:
                pass  # use as context manager

        mock_conn.close.assert_called_once()


class TestModelInfo:

    def test_model_info_returns_dict(self, optimizer):
        info = optimizer.model_info
        assert isinstance(info, dict)
        assert "algorithm" in info

    def test_repr_contains_algorithm(self, optimizer):
        r = repr(optimizer)
        assert "QueryOptimizer" in r