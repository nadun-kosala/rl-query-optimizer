"""
model_loader.py
---------------
Handles loading, validating, and hot-swapping RL models.

This is the module that makes model replacement safe.  Every model that
enters the optimizer — whether the bundled default or a developer-supplied
custom model — is validated here before it is allowed to serve queries.

Validation checks:
  1. File exists and is a valid Stable Baselines3 archive
  2. Observation space shape == (10,)   [must match query_env_v4 features]
  3. Action space size       == 7       [must match ACTION_HINTS keys]
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .execptions import ModelNotLoadedError, ModelValidationError
from .hints import NUM_ACTIONS, OBS_DIMENSIONS

# Lazy imports — stable_baselines3 is heavy; only import when needed
_PPO = None
_DQN = None


def _import_sb3():
    """Import stable_baselines3 lazily so the package loads fast."""
    global _PPO, _DQN
    if _PPO is None:
        from stable_baselines3 import PPO, DQN
        _PPO = PPO
        _DQN = DQN


def _default_model_path() -> Path:
    """
    Return the path to the bundled default model.
    Works whether the package is installed (egg/wheel) or run from source.
    """
    try:
        # Python 3.9+ path
        ref = importlib.resources.files("rl_query_optimizer") / "models" / "default_model.zip"
        with importlib.resources.as_file(ref) as p:
            return Path(p)
    except (AttributeError, TypeError):
        # Fallback for Python 3.8
        here = Path(__file__).parent
        return here / "models" / "default_model.zip"


def load_model(model_path: Optional[Union[str, Path]] = None):
    """
    Load and validate an RL model.

    Parameters
    ----------
    model_path : str | Path | None
        Path to a .zip model file (PPO or DQN from Stable Baselines3).
        If None, loads the bundled default model.

    Returns
    -------
    model : PPO | DQN
        A validated, ready-to-use model.

    Raises
    ------
    ModelValidationError
        If the file is missing, corrupt, or has wrong dimensions.
    """
    _import_sb3()

    if model_path is None:
        path = _default_model_path()
    else:
        path = Path(model_path)

    if not path.exists():
        raise ModelValidationError(
            f"Model file not found: {path}\n"
            "Pass a valid path to replace_model() or ensure the package "
            "was installed correctly."
        )

    # Try loading as PPO first, then DQN
    model = None
    last_error = None
    for ModelClass in [_PPO, _DQN]:
        try:
            model = ModelClass.load(str(path))
            break
        except Exception as e:
            last_error = e
            continue

    if model is None:
        raise ModelValidationError(
            f"Could not load model from {path}.\n"
            f"Make sure it is a valid Stable Baselines3 PPO or DQN model.\n"
            f"Last error: {last_error}"
        )

    validate_model(model, path)
    return model


def validate_model(model, path: Path) -> None:
    """
    Validate that a loaded model is compatible with this optimizer.

    Checks observation space and action space dimensions match the
    feature extractor and hint set defined in hints.py.

    Raises
    ------
    ModelValidationError
        With a clear message explaining exactly what is wrong.
    """
    # Check observation space
    obs_shape = model.observation_space.shape
    if obs_shape != (OBS_DIMENSIONS,):
        raise ModelValidationError(
            f"Model observation space is {obs_shape}, "
            f"expected ({OBS_DIMENSIONS},).\n"
            f"This model was trained with a different feature set and "
            f"cannot be used with this version of rl-query-optimizer.\n"
            f"Model path: {path}"
        )

    # Check action space
    n_actions = model.action_space.n
    if n_actions != NUM_ACTIONS:
        raise ModelValidationError(
            f"Model action space has {n_actions} actions, "
            f"expected {NUM_ACTIONS}.\n"
            f"This model was trained with a different set of query hints.\n"
            f"Model path: {path}"
        )


def get_model_info(model) -> dict:
    """
    Return a human-readable summary of a loaded model.
    Useful for logging and debugging.
    """
    algo = type(model).__name__
    obs  = model.observation_space.shape
    acts = model.action_space.n

    policy_arch = "unknown"
    try:
        policy_arch = str(
            model.policy.mlp_extractor.policy_net
            if hasattr(model.policy, "mlp_extractor")
            else model.policy.net_arch
        )
    except Exception:
        pass

    return {
        "algorithm":         algo,
        "observation_shape": obs,
        "num_actions":       acts,
        "policy_arch":       policy_arch,
    }