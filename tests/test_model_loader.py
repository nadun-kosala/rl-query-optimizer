"""
tests/test_model_loader.py
--------------------------
Tests for model loading and validation.

These tests do NOT require a database connection or GPU.
They test that the model loader correctly validates model dimensions
and raises the right errors for bad inputs.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rl_query_optimizer.execptions import ModelValidationError
from rl_query_optimizer.hints import NUM_ACTIONS, OBS_DIMENSIONS
from rl_query_optimizer.model_loader import validate_model


class MockObsSpace:
    def __init__(self, shape):
        self.shape = shape


class MockActSpace:
    def __init__(self, n):
        self.n = n


def make_mock_model(obs_shape=(10,), n_actions=7):
    m = MagicMock()
    m.observation_space = MockObsSpace(obs_shape)
    m.action_space      = MockActSpace(n_actions)
    return m


class TestValidateModel:

    def test_valid_model_passes(self):
        """A model with correct dimensions should pass without error."""
        model = make_mock_model(obs_shape=(OBS_DIMENSIONS,), n_actions=NUM_ACTIONS)
        validate_model(model, Path("dummy.zip"))  # should not raise

    def test_wrong_obs_shape_raises(self):
        """A model with wrong observation dimensions should raise ModelValidationError."""
        model = make_mock_model(obs_shape=(8,), n_actions=NUM_ACTIONS)
        with pytest.raises(ModelValidationError) as exc_info:
            validate_model(model, Path("dummy.zip"))
        assert "observation space" in str(exc_info.value).lower()
        assert "8" in str(exc_info.value)

    def test_wrong_action_space_raises(self):
        """A model with wrong number of actions should raise ModelValidationError."""
        model = make_mock_model(obs_shape=(OBS_DIMENSIONS,), n_actions=4)
        with pytest.raises(ModelValidationError) as exc_info:
            validate_model(model, Path("dummy.zip"))
        assert "action space" in str(exc_info.value).lower()
        assert "4" in str(exc_info.value)

    def test_missing_file_raises(self):
        """Loading a non-existent file should raise ModelValidationError."""
        from rl_query_optimizer.model_loader import load_model
        with pytest.raises(ModelValidationError) as exc_info:
            load_model("/nonexistent/path/model.zip")
        assert "not found" in str(exc_info.value).lower()

    def test_error_message_includes_path(self):
        """Error messages should always include the file path for debugging."""
        model = make_mock_model(obs_shape=(5,), n_actions=NUM_ACTIONS)
        path  = Path("some/path/model.zip")
        with pytest.raises(ModelValidationError) as exc_info:
            validate_model(model, path)
        assert str(path) in str(exc_info.value)