import os
import pickle
import subprocess

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mle_training.train import (  # Assuming these functions are accessible
    load_data,
    preprocess_data,
)


@pytest.fixture
def sample_data_path(tmp_path):
    # Load a sample dataset from housing.csv
    sample_data = pd.read_csv("data/housing.csv")
    sample_path = tmp_path / "sample_housing.csv"
    sample_data.to_csv(sample_path, index=False)
    return sample_path

@pytest.fixture
def output_dir():
    # Create a temporary directory for saving models
    dir_path = "temp_artifacts"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    # Cleanup after test is done
    # if os.path.exists(dir_path):
    #     for file in os.listdir(dir_path):
    #         os.remove(os.path.join(dir_path, file))
    #     os.rmdir(dir_path)

def test_script_runs_successfully(sample_data_path, output_dir):
    """Test that the training script runs end-to-end without errors."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "mle_training.train",
            "--input_dataset_path",
            str(sample_data_path),
            "--model_output_path",
            output_dir,
        ],
        capture_output=True,
        text=True,
    )

    # Verify that the script runs successfully
    assert result.returncode == 0
    assert "Models trained and saved" in result.stderr

def test_model_files_exist(output_dir):
    """Check that all expected model files were created."""
    expected_files = [
        "linear_regression.pkl",
        "decision_tree_regressor.pkl",
        "random_forest_regressor.pkl",
    ]

    for model_file in expected_files:
        assert os.path.isfile(os.path.join(output_dir, model_file))

def test_model_predictions(sample_data_path, output_dir):
    """Test that the saved models can make predictions on preprocessed data."""
    # Load and preprocess sample data
    housing_data = load_data(sample_data_path)
    housing_prepared, _ = preprocess_data(housing_data)
    sample_features = housing_prepared.iloc[:1]  # Use one sample for prediction

    # Test each model's prediction
    for model_name in ["linear_regression.pkl", "decision_tree_regressor.pkl", "random_forest_regressor.pkl"]:
        model_path = os.path.join(output_dir, model_name)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Check that the model can make predictions
        prediction = model.predict(sample_features)
        assert prediction.shape == (1,)
        assert isinstance(prediction[0], float)

        # Check that the model can make predictions
        prediction = model.predict(sample_features)
        assert prediction.shape == (1,)
        assert isinstance(prediction[0], float)
