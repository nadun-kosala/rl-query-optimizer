"""
hints.py
--------
All PostgreSQL planner hint constants used by the optimizer.

Kept in a dedicated module so:
  - The optimizer, environment, and any future tools all share one source of truth.
  - Adding a new action only requires editing this file.
"""

from typing import Dict, List

# ── Action definitions ────────────────────────────────────────────────────────

ACTION_NAMES: Dict[int, str] = {
    0: "Default (no hints)",
    1: "Force Index Scan",
    2: "Force Sequential Scan",
    3: "Prefer Hash Join",
    4: "Prefer Merge Join",
    5: "Prefer Nested Loop",
    6: "Disable Parallel Query",
}

# PostgreSQL SET commands applied before executing a query for each action
ACTION_HINTS: Dict[int, List[str]] = {
    0: [],
    1: [
        "SET enable_seqscan = OFF",
        "SET enable_indexscan = ON",
        "SET enable_bitmapscan = ON",
    ],
    2: [
        "SET enable_seqscan = ON",
        "SET enable_indexscan = OFF",
        "SET enable_bitmapscan = OFF",
    ],
    3: [
        "SET enable_hashjoin = ON",
        "SET enable_mergejoin = OFF",
        "SET enable_nestloop = OFF",
    ],
    4: [
        "SET enable_hashjoin = OFF",
        "SET enable_mergejoin = ON",
        "SET enable_nestloop = OFF",
    ],
    5: [
        "SET enable_hashjoin = OFF",
        "SET enable_mergejoin = OFF",
        "SET enable_nestloop = ON",
    ],
    6: [
        "SET max_parallel_workers_per_gather = 0",
    ],
}

# Commands to reset all planner settings back to PostgreSQL defaults
RESET_HINTS: List[str] = [
    "SET enable_seqscan = ON",
    "SET enable_indexscan = ON",
    "SET enable_bitmapscan = ON",
    "SET enable_hashjoin = ON",
    "SET enable_mergejoin = ON",
    "SET enable_nestloop = ON",
    "RESET max_parallel_workers_per_gather",
]

# Number of available actions — must match the model's action space
NUM_ACTIONS: int = len(ACTION_NAMES)

# Observation space dimensions — must match the model's input layer
OBS_DIMENSIONS: int = 10