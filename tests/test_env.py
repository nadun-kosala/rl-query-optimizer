import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Adjust this import path if your env file is named differently inside the package
from rl_query_optimizer.env import QueryOptimizationEnvV3

@pytest.fixture
def mock_db_config():
    return {"host": "localhost", "dbname": "test", "user": "root", "password": "pwd"}

@pytest.fixture
def sample_queries():
    return [
        "SELECT * FROM students WHERE age > 20",
        "SELECT s.name, c.name FROM students s JOIN courses c ON s.course_id = c.id GROUP BY s.name ORDER BY s.name"
    ]

@pytest.fixture
def mock_env(mock_db_config, sample_queries):
    """
    Fixture that initializes the environment but intercepts psycopg2
    so it never actually connects to a real database during unit tests.
    """
    with patch("psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Mock the JSON output of the EXPLAIN command
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [[{"Plan": {"Plan Rows": 100, "Total Cost": 50.5, "Plan Width": 20}}]]
        mock_conn.cursor.return_value = mock_cursor

        env = QueryOptimizationEnvV3(
            db_config=mock_db_config, 
            queries=sample_queries, 
            n_avg=1
        )
        yield env, mock_cursor

def test_initialization(mock_env):
    env, _ = mock_env
    assert env.action_space.n == 7
    assert env.observation_space.shape == (10,)
    assert len(env.queries) == 2
    assert env.conn is not None

@patch("rl_query_optimizer.env.QueryOptimizationEnvV3._time_query")
def test_reset(mock_time_query, mock_env):
    env, _ = mock_env
    mock_time_query.return_value = 15.0  # Mock baseline time
    
    obs, info = env.reset()
    
    assert obs.shape == (10,)
    assert isinstance(info, dict)
    assert env.current_query in env.queries
    assert env.baseline_time == 15.0

@patch("rl_query_optimizer.env.QueryOptimizationEnvV3._time_query")
def test_step(mock_time_query, mock_env):
    env, _ = mock_env
    # Force a specific state
    env.current_query = env.queries[0]
    env.baseline_time = 100.0

    # Simulate the optimizer finding a faster execution plan (50.0ms)
    mock_time_query.return_value = 50.0

    # Execute Action 3 (Prefer Hash Join)
    obs, reward, terminated, truncated, info = env.step(3)

    assert obs.shape == (10,)
    assert reward > 0.0  # Should be rewarded for halving the time
    assert terminated is True
    assert truncated is False
    assert info["action"] == 3
    assert info["optimized_ms"] == 50.0
    assert info["baseline_ms"] == 100.0

def test_extract_features_no_cache(mock_env):
    env, _ = mock_env
    query = env.queries[1]  # The complex query with JOIN, GROUP BY, ORDER BY
    
    features = env._extract_features(query)

    assert features.shape == (10,)
    assert features[1] == 1.0  # JOIN flag
    assert features[3] == 1.0  # GROUP BY flag
    assert features[4] == 1.0  # ORDER BY flag
    
    # Verify it was added to the cache
    assert query in env._feature_cache

def test_extract_features_with_cache(mock_env):
    env, _ = mock_env
    query = "SELECT * FROM dummy"
    dummy_features = np.ones(10, dtype=np.float32)
    
    # Pre-populate the cache
    env._feature_cache[query] = dummy_features

    features = env._extract_features(query)
    
    # Should bypass regex/DB entirely and return the exact cached numpy array
    np.testing.assert_array_equal(features, dummy_features)

def test_compute_reward_logic(mock_env):
    env, _ = mock_env

    # 1. Baseline is 0 (division by zero safeguard)
    assert env._compute_reward(optimized_ms=10.0, baseline_ms=0.0) == 0.0

    # 2. Improvement is below NOISE_THRESHOLD_PCT (e.g., 1% faster)
    assert env._compute_reward(optimized_ms=99.0, baseline_ms=100.0) == 0.0

    # 3. High baseline, excellent improvement (100ms -> 50ms)
    good_reward = env._compute_reward(optimized_ms=50.0, baseline_ms=100.0)
    assert good_reward > 0.0

    # 4. Low baseline dampening (4.0ms -> 2.0ms)
    # It's a 50% improvement, but starting value is tiny (< MIN_BASELINE_MS)
    dampened_reward = env._compute_reward(optimized_ms=2.0, baseline_ms=4.0)
    assert dampened_reward > 0.0
    assert dampened_reward < good_reward  # Verifies the * 0.1 dampener fired

@patch("time.perf_counter")
def test_time_query_execution_and_hints(mock_perf, mock_env):
    env, mock_cursor = mock_env
    # Simulate time passing: t0=0.0, t1=0.1 (100ms elapsed)
    mock_perf.side_effect = [0.0, 0.1]

    # Run query with Action 6 (Disable Parallel Query)
    median_time = env._time_query("SELECT * FROM test", action=6, n=1)

    assert median_time == 100.0  # (0.1 - 0.0) * 1000
    
    # Verify the specific hint for Action 6 was sent to the DB
    mock_cursor.execute.assert_any_call("SET max_parallel_workers_per_gather = 0")
    
    # Verify cleanup hints were sent afterward
    mock_cursor.execute.assert_any_call("RESET max_parallel_workers_per_gather")

def test_get_explain_info_exception_handling(mock_env):
    env, mock_cursor = mock_env
    
    # Simulate an invalid SQL query crashing the EXPLAIN command
    mock_cursor.execute.side_effect = Exception("Simulated DB Error")

    info = env._get_explain_info("INVALID SQL SYNTAX")

    # Should catch the exception gracefully and return default fallback values
    assert info["rows"] == 1
    assert info["cost"] == 0.0
    assert info["width"] == 0

def test_close_connection(mock_env):
    env, _ = mock_env
    env.close()
    env.conn.close.assert_called_once()