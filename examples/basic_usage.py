"""
examples/basic_usage.py
-----------------------
Minimal example showing how to use rl-query-optimizer in a project.
"""

from rl_query_optimizer import QueryOptimizer, ModelValidationError

# ── 1. Create the optimizer ───────────────────────────────────────────────────
# Uses the bundled default model automatically.
# Replace model_path with your own model if needed.

optimizer = QueryOptimizer(
    db_config={
        "host":     "192.168.8.199",
        "database": "student_db_2",
        "user":     "root",
        "password": "root",
    },
    verbose=True,   # prints chosen action and timing for each query
)

# ── 2. Execute a query with automatic optimization ────────────────────────────

result = optimizer.execute(
    "SELECT s.name, c.course_name "
    "FROM students s "
    "JOIN courses c ON s.course_id = c.id "
    "WHERE s.age > 21 "
    "ORDER BY s.name;"
)

print(f"\nReturned {len(result['rows'])} rows")
print(f"Action used  : {result['action_name']}")
print(f"Executed in  : {result['optimized_ms']:.2f}ms")

# ── 3. Get a suggestion without executing ─────────────────────────────────────

suggestion = optimizer.suggest(
    "SELECT age, COUNT(*) FROM students GROUP BY age ORDER BY age;"
)
print(f"\nSuggested action : [{suggestion['action']}] {suggestion['name']}")
print(f"Confidence       : {suggestion['confidence']:.2f}")
print(f"Hints that would be applied: {suggestion['hints']}")

# ── 4. Compare optimizer vs default plan ──────────────────────────────────────

result = optimizer.execute(
    "SELECT s.name, c.course_name "
    "FROM students s "
    "JOIN courses c ON s.course_id = c.id;",
    compare=True,
)
print(f"\nOptimized : {result['optimized_ms']:.2f}ms")
print(f"Default   : {result['default_ms']:.2f}ms")
print(f"Improvement: {result['improvement_pct']:+.1f}%")

# ── 5. Replace the model with an improved version ─────────────────────────────

try:
    optimizer.replace_model("models/default_model.zip")
    print("\nModel replaced successfully.")
except ModelValidationError as e:
    print(f"\nModel replacement failed: {e}")

# ── 6. Clean up ───────────────────────────────────────────────────────────────

optimizer.close()

# ── Alternative: use as a context manager (auto-close) ────────────────────────

with QueryOptimizer(db_config={
    "host": "192.168.8.199", "database": "student_db_2",
    "user": "root", "password": "root",
}) as opt:
    rows = opt.execute("SELECT * FROM students ORDER BY name;")["rows"]
    print(f"\nFetched {len(rows)} students via context manager.")