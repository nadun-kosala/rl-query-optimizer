"""
exceptions.py
-------------
Custom exceptions for rl-query-optimizer.

Using named exceptions instead of generic ValueError/RuntimeError means
developers get clear, searchable error messages and can catch specific
failure types in their own error handling.
"""


class RLOptimizerError(Exception):
    """Base exception for all rl-query-optimizer errors."""


class ModelValidationError(RLOptimizerError):
    """
    Raised when a model file fails validation.

    This happens when:
      - The file is not a valid Stable Baselines3 model
      - The observation space dimensions do not match (expected 10)
      - The action space size does not match (expected 7)

    Example
    -------
    >>> optimizer.replace_model("wrong_model.zip")
    ModelValidationError: Model action space is 4, expected 7.
    """


class DatabaseConnectionError(RLOptimizerError):
    """
    Raised when the optimizer cannot connect to PostgreSQL.

    Example
    -------
    >>> QueryOptimizer(db_config={"host": "badhost", ...})
    DatabaseConnectionError: Cannot connect to PostgreSQL at badhost:5432.
    """


class ModelNotLoadedError(RLOptimizerError):
    """
    Raised when execute() or suggest() is called before a model is loaded.
    Should not happen in normal usage since __init__ loads the default model,
    but can occur if replace_model() was called with an invalid path and
    the previous model was already unloaded.
    """


class QueryExecutionError(RLOptimizerError):
    """
    Raised when the optimizer fails to execute a SQL query.
    Wraps the underlying psycopg2 error with additional context.
    """