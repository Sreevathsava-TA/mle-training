import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mle_training.ingest_data import (
    create_datasets,
    fetch_housing_data,
    load_housing_data,
)


@pytest.fixture
def sample_data_path():
    # Load a sample dataset from the actual path
    return Path("data/housing.csv")


@pytest.fixture
def output_path(tmp_path, sample_data_path):
    # Temporary directory for output and copy sample data
    output_dir = tmp_path / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_data_path, output_dir / "housing.csv")
    return output_dir


def test_fetch_housing_data(output_path):
    fetch_housing_data(housing_path=output_path)
    tgz_path = output_path / "housing.tgz"

    # Check if housing.tgz was downloaded
    assert tgz_path.is_file(), "housing.tgz should be downloaded"


def test_load_housing_data(sample_data_path):
    data = load_housing_data(sample_data_path.parent)

    # Basic validation
    assert isinstance(data, pd.DataFrame), "Data should be a DataFrame"
    assert not data.empty, "Loaded data should not be empty"


def test_create_datasets(output_path, mocker):
    # Mock fetch_housing_data to prevent downloading in tests
    mocker.patch("mle_training.ingest_data.fetch_housing_data", return_value=None)

    create_datasets(output_path)

    # Check if the output files were created
    train_path = output_path / "train_set.csv"
    test_path = output_path / "test_set.csv"

    assert train_path.is_file(), "train_set.csv should be created"
    assert test_path.is_file(), "test_set.csv should be created"

    # Load the datasets and perform basic checks
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    assert isinstance(train_data, pd.DataFrame), "Train set should be a DataFrame"
    assert isinstance(test_data, pd.DataFrame), "Test set should be a DataFrame"
    assert not train_data.empty, "Train set should not be empty"
    assert not test_data.empty, "Test set should not be empty"
    test_data = pd.read_csv(test_path)

    assert isinstance(train_data, pd.DataFrame), "Train set should be a DataFrame"
    assert isinstance(test_data, pd.DataFrame), "Test set should be a DataFrame"
    assert not train_data.empty, "Train set should not be empty"
    assert not test_data.empty, "Test set should not be empty"
