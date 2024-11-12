import os
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from mle_training.ingest_data import (
    create_datasets,
    fetch_housing_data,
    load_housing_data,
)


# Fixture to create a temporary directory for testing
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

# Mock for URL download to avoid actual download
@pytest.fixture
def mock_urlretrieve(mocker):
    return mocker.patch("six.moves.urllib.request.urlretrieve")

# Fixture to load a sample dataset from the updated housing dataset path
@pytest.fixture
def sample_data_path():
    # Load a sample dataset from the actual path
    return Path("data/housing.csv")

# Test load_housing_data to check if data is loaded correctly
def test_load_housing_data(temp_dir, sample_data_path):
    # Copy the sample data to the temporary directory
    mock_csv_path = temp_dir / "housing.csv"
    sample_data = pd.read_csv(sample_data_path)
    sample_data.to_csv(mock_csv_path, index=False)

    # Load data using the function
    loaded_data = load_housing_data(housing_path=temp_dir)
    pd.testing.assert_frame_equal(loaded_data, sample_data, check_dtype=False)

# Test create_datasets to check if datasets are created and saved correctly
def test_create_datasets(temp_dir, mocker, sample_data_path):
    # Copy the sample data to the temporary directory
    mock_csv_path = temp_dir / "housing.csv"
    sample_data = pd.read_csv(sample_data_path)
    sample_data.to_csv(mock_csv_path, index=False)

    # Run create_datasets function
    create_datasets(output_path=temp_dir)

    # Check if train_set.csv and test_set.csv exist
    train_path = temp_dir / "train_set.csv"
    test_path = temp_dir / "test_set.csv"
    assert train_path.exists(), "train_set.csv should exist in output path."
    assert test_path.exists(), "test_set.csv should exist in output path."

    # Check that the CSV files have data and contain expected columns
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    assert "median_income" in train_data.columns, "train_set.csv should contain the column 'median_income'"
    assert "median_house_value" in test_data.columns, "test_set.csv should contain the column 'median_house_value'"
    assert "median_house_value" in test_data.columns, "test_set.csv should contain the column 'median_house_value'"
