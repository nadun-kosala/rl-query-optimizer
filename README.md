# rl-query-optimizer

A reinforcement learning-based PostgreSQL query optimizer.

Automatically selects the best query execution plan hints — such as forcing
a Hash Join, Index Scan, or Nested Loop — based on a pre-trained PPO model,
improving performance for complex JOINs, aggregations, and large table scans.

---

## Installation

```bash
pip install rl-query-optimizer
```

Requires Python 3.8+ and a running PostgreSQL instance.

---

## Quick start

```python
from rl_query_optimizer import QueryOptimizer

optimizer = QueryOptimizer(db_config={
    "host":     "localhost",
    "database": "mydb",
    "user":     "admin",
    "password": "secret",
})

# Run a query with automatic plan optimization
result = optimizer.execute(
    "SELECT s.name, c.course_name "
    "FROM students s "
    "JOIN courses c ON s.course_id = c.id "
    "WHERE s.age > 21;"
)

print(result["rows"])
print(f"Action: {result['action_name']}  ({result['optimized_ms']:.1f}ms)")
```

---

## API

### `QueryOptimizer(db_config, model_path=None, verbose=False)`

| Parameter    | Type        | Description                                               |
|-------------|-------------|-----------------------------------------------------------|
| `db_config`  | `dict`      | psycopg2 connection kwargs (host, database, user, password) |
| `model_path` | `str\|None` | Path to a `.zip` model file. Uses bundled default if None. |
| `verbose`    | `bool`      | Print action and timing for each query. Default `False`.  |

---

### `optimizer.execute(sql, params=None, compare=False)`

Execute a query using the optimizer's recommended plan.

```python
result = optimizer.execute("SELECT * FROM orders JOIN customers ON ...", compare=True)

print(result["rows"])            # query results
print(result["action_name"])     # e.g. "Prefer Hash Join"
print(result["optimized_ms"])    # execution time with optimizer hints
print(result["default_ms"])      # execution time without hints (if compare=True)
print(result["improvement_pct"]) # % improvement (if compare=True)
```

---

### `optimizer.suggest(sql)`

Get the optimizer's recommendation without executing the query.

```python
suggestion = optimizer.suggest("SELECT * FROM orders JOIN customers ON ...")
# {
#     "action":     3,
#     "name":       "Prefer Hash Join",
#     "hints":      ["SET enable_hashjoin = ON", ...],
#     "confidence": 8.42
# }
```

---

### `optimizer.replace_model(model_path)`

Hot-swap the model without restarting. The new model is validated before
the old one is replaced — if validation fails, the original keeps running.

```python
optimizer.replace_model("models/improved_model_v2.zip")
```

When you release an improved model, simply bump the package version and
developers get the new model automatically via `pip install --upgrade rl-query-optimizer`.

---

### `optimizer.invalidate_cache(sql=None)`

Clear EXPLAIN and feature caches. Call after major schema changes (new indexes,
ANALYZE, etc.).

```python
optimizer.invalidate_cache()        # clear all
optimizer.invalidate_cache(sql)     # clear one query
```

---

### Context manager

```python
with QueryOptimizer(db_config=...) as opt:
    result = opt.execute("SELECT ...")
# connection closed automatically
```

---

## Error handling

```python
from rl_query_optimizer import (
    QueryOptimizer,
    ModelValidationError,
    DatabaseConnectionError,
    QueryExecutionError,
)

try:
    optimizer.replace_model("bad_model.zip")
except ModelValidationError as e:
    print(f"Model rejected: {e}")

try:
    result = optimizer.execute("SELECT ...")
except QueryExecutionError as e:
    print(f"Query failed: {e}")
```

---

## Replacing the model

The package ships with a default trained model. To use your own:

```python
# Option 1: pass at construction time
optimizer = QueryOptimizer(db_config=..., model_path="my_model.zip")

# Option 2: hot-swap at any time
optimizer.replace_model("my_model.zip")
```

The model must be a Stable Baselines3 PPO or DQN model trained with:
- Observation space: shape `(10,)`
- Action space: `Discrete(7)`

---

## Supported actions

| ID | Action                 | When it helps                              |
|----|------------------------|--------------------------------------------|
| 0  | Default (no hints)     | When PostgreSQL already picks the best plan |
| 1  | Force Index Scan       | Highly selective WHERE clauses              |
| 2  | Force Sequential Scan  | Full table scans with low selectivity       |
| 3  | Prefer Hash Join       | Large table JOINs                          |
| 4  | Prefer Merge Join      | Pre-sorted data JOINs                      |
| 5  | Prefer Nested Loop     | Small inner tables, highly filtered JOINs  |
| 6  | Disable Parallel Query | Simple queries where parallelism adds overhead |

---

## License

MIT