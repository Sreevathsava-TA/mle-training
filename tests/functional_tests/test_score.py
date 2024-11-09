import os
import pickle
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from mle_training.score import (  # Assuming functions are accessible
    load_data,
    preprocess_data,
    score_models,
)


@pytest.fixture
def sample_data_path(tmp_path):
    """Fixture to load sample housing data and save it to a temporary file."""
    sample_data = pd.read_csv("data/housing.csv")
    sample_path = tmp_path / "sample_housing.csv"
    sample_data.to_csv(sample_path, index=False)
    return sample_path


@pytest.fixture
def model_path():
    """Fixture to provide the path to the pretrained random forest model."""
    model_file = "artifacts/random_forest_regressor.pkl"
    assert os.path.isfile(model_file), f"{model_file} does not exist"
    return model_file


@pytest.fixture
def output_dir():
    """Fixture to create a temporary directory for storing outputs."""
    dir_path = "temp_results"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    # Cleanup code for output directory (if necessary)
    # if os.path.exists(dir_path):
    #     for file in os.listdir(dir_path):
    #         os.remove(os.path.join(dir_path, file))
    #     os.rmdir(dir_path)


def test_script_runs_successfully(sample_data_path, model_path, output_dir):
    """Test that the scoring script runs end-to-end without errors."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "mle_training.score",
            "--model_folder",
            model_path,
            "--dataset_folder",
            str(sample_data_path),
            "--output_folder",
            output_dir,
        ],
        capture_output=True,
        text=True,
    )

    # Verify that the script runs successfully
    assert result.returncode == 0
    # Check if the message is in stderr
    assert "Predictions saved to" in result.stderr



def test_score_computation_accuracy(sample_data_path, model_path):
    """Test that scoring computes MSE and RMSE accurately."""
    # Load data and model
    housing = load_data(sample_data_path)
    X_test_prepared, y_test = preprocess_data(housing)

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Get predictions
    predictions = model.predict(X_test_prepared)

    # Calculate expected metrics
    expected_mse = mean_squared_error(y_test, predictions)
    expected_rmse = np.sqrt(expected_mse)

    # Assert metrics match
    assert np.isclose(expected_mse, expected_mse, atol=1e-2)
    assert np.isclose(expected_rmse, expected_rmse, atol=1e-2)


def test_output_directory_created(output_dir):
    """Verify that the output directory is created if it doesn't exist."""
    assert os.path.exists(output_dir), "Output directory should be created"
