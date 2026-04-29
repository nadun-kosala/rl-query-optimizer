"""
rl-query-optimizer
==================
A reinforcement learning-based PostgreSQL query optimizer.

Automatically selects the best query execution plan hints based on a
pre-trained RL model, improving performance for complex JOINs,
aggregations, and large table scans.

Quick start
-----------
    from rl_query_optimizer import QueryOptimizer

    optimizer = QueryOptimizer(db_config={
        "host":     "localhost",
        "database": "mydb",
        "user":     "admin",
        "password": "secret",
    })

    # Run a query with automatic plan optimization
    result = optimizer.execute("SELECT * FROM orders JOIN customers ON ...")
    print(result["rows"])
    print(f"Action: {result['action_name']}  ({result['optimized_ms']:.1f}ms)")

    # Get a suggestion without executing
    suggestion = optimizer.suggest("SELECT * FROM orders JOIN customers ON ...")
    print(suggestion)   # {"action": 3, "name": "Prefer Hash Join", ...}

    # Replace model without restarting
    optimizer.replace_model("path/to/improved_model.zip")

    # Use as a context manager for automatic cleanup
    with QueryOptimizer(db_config=...) as opt:
        result = opt.execute("SELECT ...")
"""

from .optimizer  import QueryOptimizer
from .execptions import (
    RLOptimizerError,
    ModelValidationError,
    DatabaseConnectionError,
    ModelNotLoadedError,
    QueryExecutionError,
)
from .hints import ACTION_NAMES, NUM_ACTIONS

__version__ = "0.1.0"
__author__  = "Kosala Nadun Madanayaka"
__all__     = [
    "QueryOptimizer",
    # Exceptions — exported so developers can catch them specifically
    "RLOptimizerError",
    "ModelValidationError",
    "DatabaseConnectionError",
    "ModelNotLoadedError",
    "QueryExecutionError",
    # Constants
    "ACTION_NAMES",
    "NUM_ACTIONS",
]