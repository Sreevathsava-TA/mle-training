import argparse
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

def fetch_housing_data(housing_url=HOUSING_URL, housing_path="."):
    """
    Downloads and extracts the housing data from a specified URL.

    Parameters
    ----------
    housing_url : str
        URL of the housing dataset.
    housing_path : str
        Directory path where the dataset will be downloaded and extracted.

    Returns
    -------
    None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    logging.info(f"Downloading housing data from {housing_url}...")

    urllib.request.urlretrieve(housing_url, tgz_path)
    logging.info("Download complete. Extracting housing data...")

    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

    logging.info(f"Housing data extracted to {housing_path}.")

def load_housing_data(housing_path="."):
    """
    Loads the housing data from a specified directory.

    Parameters
    ----------
    housing_path : str
        Directory path where the housing dataset is stored.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the housing data.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    logging.info(f"Loading housing data from {csv_path}...")

    data = pd.read_csv(csv_path)
    logging.info("Housing data loaded successfully.")
    return data

def create_datasets(output_path):
    """
    Creates stratified training and test datasets from the housing data.

    Parameters
    ----------
    output_path : str
        Directory path where the datasets will be saved.

    Returns
    -------
    None
    """
    logging.info(f"Creating datasets in {output_path}...")

    # Fetch and load data
    fetch_housing_data(housing_path=output_path)
    housing = load_housing_data(housing_path=output_path)

    # Create income category for stratified split
    logging.info("Creating income category for stratified split...")
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # Perform stratified split
    logging.info("Splitting data into stratified training and test sets...")
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
    )

    # Drop the income_cat column
    logging.info("Dropping 'income_cat' column from datasets...")
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Save the datasets to CSV files
    train_path = os.path.join(output_path, "train_set.csv")
    test_path = os.path.join(output_path, "test_set.csv")

    logging.info(f"Saving training set to {train_path}...")
    strat_train_set.to_csv(train_path, index=False)
    logging.info(f"Training set saved to {train_path}.")

    logging.info(f"Saving test set to {test_path}...")
    strat_test_set.to_csv(test_path, index=False)
    logging.info(f"Test set saved to {test_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and create training and validation datasets."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The output folder to save the datasets.",
        default="data",
    )
    args = parser.parse_args()

    logging.info("Starting dataset creation process...")
    create_datasets(args.output_path)
    logging.info(f"Datasets created and saved to {args.output_path}")
