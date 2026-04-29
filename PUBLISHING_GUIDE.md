# rl-query-optimizer — Complete Publishing Guide
## From local setup to PyPI, step by step

---

## Before you start — your final folder structure

Your package folder must look exactly like this before running any commands.
Check every file is present:

```
rl_query_optimizer_pkg/
├── rl_query_optimizer/
│   ├── __init__.py
│   ├── optimizer.py
│   ├── model_loader.py
│   ├── cache.py
│   ├── hints.py
│   ├── exceptions.py
│   └── models/
│       ├── __init__.py
│       └── default_model.zip        ← YOUR TRAINED MODEL (rename it here)
├── tests/
│   ├── __init__.py
│   ├── test_cache.py
│   ├── test_model_loader.py
│   └── test_optimizer.py
├── examples/
│   └── basic_usage.py
├── pyproject.toml
├── README.md
└── LICENSE
```

**What you are missing right now:**
- `default_model.zip` — copy `best_model_ppo.zip` here and rename it
- `tests/__init__.py` — must exist (even if empty) for pytest to find tests

---

## pyproject.toml — dependencies section (updated)

Replace the `dependencies` block in your `pyproject.toml` with this.
The versions are aligned exactly with your `requirements.txt`:

```toml
dependencies = [
    "stable-baselines3==2.1.0",
    "psycopg2-binary>=2.9.9",
    "numpy>=1.26.0,<2.0.0",
    "gymnasium==0.29.1",
    "torch>=2.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "isort",
    "mypy",
    "twine>=4.0",
    "build>=1.0",
]
```

**Why these specific changes from the original:**

| Package | Old | New | Reason |
|---------|-----|-----|--------|
| stable-baselines3 | `>=2.0.0` | `==2.1.0` | Pin exact — SB3 has breaking API changes between minor versions. Your trained model was saved with 2.1.0 and must be loaded with 2.1.0. |
| numpy | `>=1.21.0` | `>=1.26.0,<2.0.0` | SB3 2.1.0 requires numpy <2.0.0. Your requirements.txt already pins this range. |
| gymnasium | `>=0.26.0` | `==0.29.1` | Pin exact — gymnasium has frequent breaking changes in the env API. |
| torch | `>=1.13.0` | `>=2.1.0` | Your requirements use 2.1.0+. No upper bound needed — PyTorch is generally backward-compatible. |
| pandas, matplotlib, faker, tensorboard, tqdm, rich | — | NOT included | These are training tools, not needed at inference time. Do not put them in package dependencies. |

---

## STEP 1 — Place the trained model

```bash
# From inside rl_query_optimizer_pkg/
cp /path/to/your/best_model_ppo.zip rl_query_optimizer/models/default_model.zip
```

Verify it is there:
```bash
ls -lh rl_query_optimizer/models/
# Should show: default_model.zip  (typically 50KB–2MB for PPO [64,64])
```

**If the file is missing, the package will fail to import.**
The model_loader falls back gracefully with a clear error, but tests will fail.

---

## STEP 2 — Create a clean virtual environment

Always test in a fresh venv — this catches missing dependencies that
your global environment might be hiding.

```bash
# Navigate to the package root
cd rl_query_optimizer_pkg

# Create venv (use python3 or python depending on your system)
python3 -m venv .venv_test

# Activate it
source .venv_test/bin/activate        # macOS / Linux
# .venv_test\Scripts\activate         # Windows

# Confirm the venv is active — prompt should show (.venv_test)
which python    # should point inside .venv_test/
```

---

## STEP 3 — Install the package in editable mode with dev dependencies

Editable mode (`-e`) means changes to your source files take effect
immediately without reinstalling. This is how you develop and test.

```bash
# Install package + all dev tools (pytest, black, twine, build)
pip install -e ".[dev]"

# This will also install all runtime dependencies:
# stable-baselines3, psycopg2-binary, numpy, gymnasium, torch
# (torch is large — this may take 5–10 minutes on first install)
```

Verify the install worked:
```bash
pip list | grep rl-query-optimizer
# Should show: rl-query-optimizer  0.1.0

python -c "from rl_query_optimizer import QueryOptimizer; print('Import OK')"
# Should print: Import OK
```

---

## STEP 4 — Run the test suite

### 4a. Run all tests

```bash
pytest tests/ -v
```

Expected output (all green):
```
tests/test_cache.py::TestQueryCache::test_feature_vector_shape          PASSED
tests/test_cache.py::TestQueryCache::test_feature_vector_cached_on_second_call  PASSED
tests/test_cache.py::TestQueryCache::test_different_queries_get_different_features  PASSED
tests/test_cache.py::TestQueryCache::test_invalidate_single_query        PASSED
tests/test_cache.py::TestQueryCache::test_invalidate_all                 PASSED
tests/test_cache.py::TestQueryCache::test_join_flag_set_correctly        PASSED
tests/test_cache.py::TestQueryCache::test_no_join_flag_not_set           PASSED
tests/test_model_loader.py::TestValidateModel::test_valid_model_passes   PASSED
tests/test_model_loader.py::TestValidateModel::test_wrong_obs_shape_raises  PASSED
tests/test_model_loader.py::TestValidateModel::test_wrong_action_space_raises  PASSED
tests/test_model_loader.py::TestValidateModel::test_missing_file_raises  PASSED
tests/test_model_loader.py::TestValidateModel::test_error_message_includes_path  PASSED
tests/test_optimizer.py::TestExecute::test_returns_rows                  PASSED
tests/test_optimizer.py::TestExecute::test_returns_action_info           PASSED
tests/test_optimizer.py::TestExecute::test_returns_timing                PASSED
tests/test_optimizer.py::TestExecute::test_same_query_uses_cache         PASSED
tests/test_optimizer.py::TestSuggest::test_returns_action_dict           PASSED
tests/test_optimizer.py::TestSuggest::test_deterministic_same_query      PASSED
tests/test_optimizer.py::TestReplaceModel::test_replace_with_valid_model_succeeds  PASSED
tests/test_optimizer.py::TestReplaceModel::test_replace_clears_cache     PASSED
tests/test_optimizer.py::TestReplaceModel::test_replace_with_invalid_model_raises  PASSED
tests/test_optimizer.py::TestReplaceModel::test_original_model_preserved_after_failed_replace  PASSED
tests/test_optimizer.py::TestContextManager::test_context_manager_closes_connection  PASSED
tests/test_optimizer.py::TestModelInfo::test_model_info_returns_dict     PASSED
tests/test_optimizer.py::TestModelInfo::test_repr_contains_algorithm     PASSED

25 passed in X.XXs
```

**Note:** All 25 tests use mocked DB and model. No PostgreSQL connection
required to pass them.

### 4b. Run with coverage report

```bash
pytest tests/ -v --cov=rl_query_optimizer --cov-report=term-missing
```

This shows exactly which lines of your source code are covered by tests.
Aim for >80% coverage before publishing.

### 4c. Run a single test file

```bash
pytest tests/test_cache.py -v           # just cache tests
pytest tests/test_model_loader.py -v    # just model loader tests
pytest tests/test_optimizer.py -v       # just optimizer tests
```

### 4d. Run a single test by name

```bash
pytest tests/test_optimizer.py::TestReplaceModel::test_replace_clears_cache -v
```

---

## STEP 5 — Test with a real database (manual integration test)

The pytest tests use mocks. Before releasing, also run the example against
your real PostgreSQL database to confirm end-to-end behaviour.

```bash
# Edit examples/basic_usage.py and fill in your real db_config:
# "host": "your-db-host", "database": "student_db_2", ...

python examples/basic_usage.py
```

Expected output:
```
[RLOptimizer] Loaded PPO model  obs=(10,)  actions=7
[RLOptimizer] action=[5] Prefer Nested Loop  214.32ms  (new query)

Returned 150423 rows
Action used  : Prefer Nested Loop
Executed in  : 214.32ms
...
```

**Common issues at this stage:**

| Error | Cause | Fix |
|-------|-------|-----|
| `DatabaseConnectionError: Cannot connect` | Wrong db_config values | Check host/port/user/password |
| `ModelValidationError: Model file not found` | default_model.zip missing | Re-do Step 1 |
| `ModuleNotFoundError: No module named 'stable_baselines3'` | Install failed | Run `pip install -e ".[dev]"` again |
| `FATAL: role "root" does not exist` | PostgreSQL user wrong | Use your actual PostgreSQL username |

---

## STEP 6 — Test the installed wheel (simulates what users get)

This is the most important local test — it verifies the package works
**exactly as a user would experience it after `pip install`**.

```bash
# 1. Build the distribution files
python -m build

# You will see output like:
# Successfully built rl_query_optimizer-0.1.0-py3-none-any.whl
# and rl_query_optimizer-0.1.0.tar.gz
# Both files appear in dist/

ls dist/
# rl_query_optimizer-0.1.0-py3-none-any.whl
# rl_query_optimizer-0.1.0.tar.gz

# 2. Create a second clean venv to simulate a fresh user install
python3 -m venv .venv_install_test
source .venv_install_test/bin/activate

# 3. Install from the built wheel (NOT editable, just like pip install)
pip install dist/rl_query_optimizer-0.1.0-py3-none-any.whl

# 4. Confirm the bundled model is included
python -c "
from rl_query_optimizer.model_loader import _default_model_path
path = _default_model_path()
print('Model path:', path)
print('Exists:', path.exists())
"
# Should print: Exists: True

# 5. Confirm the public API imports cleanly
python -c "
from rl_query_optimizer import (
    QueryOptimizer,
    ModelValidationError,
    DatabaseConnectionError,
    ACTION_NAMES,
)
print('All imports OK')
print('Actions:', ACTION_NAMES)
"

# 6. Deactivate and return to dev venv
deactivate
source .venv_test/bin/activate
```

If Step 6 passes, your package is ready to publish.

---

## STEP 7 — Publish to GitHub first (recommended before PyPI)

GitHub is free, instant, and lets you share the package with your
supervisor before making it public on PyPI.

```bash
# From the rl_query_optimizer_pkg/ root:
git init
git add .
git commit -m "Initial release: rl-query-optimizer v0.1.0"

# Create a repo on github.com (name: rl-query-optimizer), then:
git remote add origin https://github.com/kosalanadun/rl-query-optimizer.git
git branch -M main
git push -u origin main
```

Anyone can now install it with:
```bash
pip install git+https://github.com/kosalanadun/rl-query-optimizer.git
```

---

## STEP 8 — Publish to PyPI (public release)

### 8a. Create a PyPI account

Go to https://pypi.org/account/register/ and create a free account.
Verify your email address.

### 8b. Create an API token (more secure than password)

1. Log in to pypi.org
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Name it: `rl-query-optimizer-publish`
5. Scope: "Entire account" (for first upload — can restrict later)
6. Copy the token — it starts with `pypi-` — you only see it once

### 8c. Configure your credentials locally

```bash
# Create the credentials file
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
EOF

chmod 600 ~/.pypirc    # restrict permissions
```

### 8d. Upload to PyPI

```bash
# Make sure you're in the package root and dev venv is active
cd rl_query_optimizer_pkg
source .venv_test/bin/activate

# Check the package before uploading (catches common mistakes)
twine check dist/*

# Upload
twine upload dist/*
```

You will see:
```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading rl_query_optimizer-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.2 MB
Uploading rl_query_optimizer-0.1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.1 MB

View at: https://pypi.org/project/rl-query-optimizer/0.1.0/
```

### 8e. Verify the live package

```bash
# In a completely fresh environment (not your dev venv):
pip install rl-query-optimizer

python -c "
from rl_query_optimizer import QueryOptimizer
print('Install from PyPI: OK')
"
```

---

## Releasing a better model later (3 commands)

When you finish training a more accurate model:

```bash
# 1. Replace the model
cp new_best_model_ppo.zip rl_query_optimizer/models/default_model.zip

# 2. Bump version in pyproject.toml
#    version = "0.1.0"  →  version = "0.2.0"

# 3. Rebuild and re-upload
python -m build
twine upload dist/*
```

Developers get the improved model automatically:
```bash
pip install --upgrade rl-query-optimizer
```

---

## Checklist before every release

- [ ] `default_model.zip` is present in `rl_query_optimizer/models/`
- [ ] Version bumped in `pyproject.toml`
- [ ] All 25 tests pass: `pytest tests/ -v`
- [ ] Wheel installs cleanly in fresh venv (Step 6)
- [ ] `twine check dist/*` passes with no warnings
- [ ] README updated if API changed