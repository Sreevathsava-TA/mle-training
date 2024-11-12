import os
import pickle
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from mle_training.train import load_data, preprocess_data, train_and_save_models


# Fixture to provide the path to the actual housing dataset
@pytest.fixture
def sample_data_path():
    # Load a sample dataset from the actual path
    return Path("data/housing.csv")

# Fixture to create a temporary directory for saving models
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

# Test load_data to ensure data loads correctly
def test_load_data(sample_data_path):
    loaded_data = load_data(sample_data_path)
    assert not loaded_data.empty, "Loaded data should not be empty"
    assert "median_house_value" in loaded_data.columns, "Expected column 'median_house_value' not found in loaded data"

# Test preprocess_data to ensure transformations are applied correctly
import numpy as np


def test_preprocess_data(sample_data_path):
    housing = load_data(sample_data_path)
    housing_prepared, housing_labels = preprocess_data(housing)

    # Check that the labels have the correct length and are a subset of the original labels
    assert len(housing_labels) > 0, "Housing labels should not be empty."
    assert len(housing_labels) == int(len(housing) * 0.8), "Housing labels should match 80% of the original data length."
    assert np.all(housing_labels.isin(housing["median_house_value"])), "Housing labels should be a subset of 'median_house_value' column"

    # Check that prepared data has the expected columns after preprocessing
    expected_columns = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "rooms_per_household", "bedrooms_per_room", "population_per_household",
        "ocean_proximity_INLAND", "ocean_proximity_ISLAND",
        "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN"
    ]
    for column in expected_columns:
        assert column in housing_prepared.columns, f"Missing expected column '{column}' in prepared data"


# Test train_and_save_models to ensure models are trained and saved correctly
def test_train_and_save_models(temp_dir, sample_data_path):
    # Load and preprocess data
    housing = load_data(sample_data_path)
    housing_prepared, housing_labels = preprocess_data(housing)

    # Run train_and_save_models function
    train_and_save_models(housing_prepared, housing_labels, temp_dir)

    # Check if model files are created
    model_files = ["linear_regression.pkl", "decision_tree_regressor.pkl", "random_forest_regressor.pkl"]
    for model_file in model_files:
        model_path = temp_dir / model_file
        assert model_path.exists(), f"{model_file} should be created in the output path."

    # Check if the saved models can be loaded and are of expected types
    with open(temp_dir / "linear_regression.pkl", "rb") as f:
        lin_reg_model = pickle.load(f)
        assert isinstance(lin_reg_model, LinearRegression), \
            "Loaded model should be an instance of LinearRegression"

    with open(temp_dir / "decision_tree_regressor.pkl", "rb") as f:
        tree_model = pickle.load(f)
        assert isinstance(tree_model, DecisionTreeRegressor), \
            "Loaded model should be an instance of DecisionTreeRegressor"

    with open(temp_dir / "random_forest_regressor.pkl", "rb") as f:
        forest_model = pickle.load(f)
        assert isinstance(forest_model, RandomForestRegressor), \
            "Loaded model should be an instance of RandomForestRegressor"
