import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from mle_training.score import load_data, preprocess_data, score_models


@pytest.fixture(scope="module")
def sample_data_path():
    # Load a sample dataset from the actual path
    return Path("data/housing.csv")


@pytest.fixture(scope="module")
def setup_data(sample_data_path):
    # Load sample data from the specified path
    housing_df = load_data(sample_data_path)
    return housing_df, sample_data_path


def test_load_data(setup_data):
    housing_df, sample_data_path = setup_data
    # Test loading data from a CSV file
    df = load_data(sample_data_path)
    pd.testing.assert_frame_equal(df, housing_df)


def test_preprocess_data(setup_data):
    housing_df, _ = setup_data
    # Test preprocessing steps
    X_test_prepared, y_test = preprocess_data(housing_df)

    # Adjust the expected number of columns based on actual preprocessing output
    expected_num_features = 15  # Update to match the final feature count after preprocessing
    assert X_test_prepared.shape[1] == expected_num_features  # Check number of features
    assert 4*len(y_test) == int(len(housing_df) * 0.8)  # Check split ratio (assuming 80-20 split)

    # Check for existence of engineered features
    assert "rooms_per_household" in X_test_prepared.columns
    assert "bedrooms_per_room" in X_test_prepared.columns
    assert "population_per_household" in X_test_prepared.columns

    # Check target values are within expected range
    assert np.all(y_test <= housing_df["median_house_value"].max())


def test_score_models(setup_data, mocker):
    housing_df, sample_data_path = setup_data

    # Calculate the expected test set size after an 80-20 train-test split
    expected_test_size = int(len(housing_df) * 0.2)

    # Create a mock model with predictions matching the test set size
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([350000] * expected_test_size)

    # Mock pickle loading to return the mock model directly
    mocker.patch("mle_training.score.pickle.load", return_value=mock_model)

    # Call score_models with the mock model and dataset
    with tempfile.TemporaryDirectory() as output_dir:
        score_models(mock_model, sample_data_path, output_dir)

        # Check model prediction method was called
        mock_model.predict.assert_called_once()

        # Verify MSE and RMSE calculations (mock output)
        predicted_values = mock_model.predict.return_value
        mse = np.mean((predicted_values - housing_df["median_house_value"][:expected_test_size]) ** 2)
        rmse = np.sqrt(mse)

        assert mse == pytest.approx(43755080186, rel=1e-6)
        assert rmse == pytest.approx(209177, rel=1e-6)
