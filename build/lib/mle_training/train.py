import argparse
import logging
import os
import pickle
import tarfile  # noqa : F401

import numpy as np
import pandas as pd
from scipy.stats import randint  # noqa : F401
from six.moves import urllib  # noqa : F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # noqa : F401
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error  # noqa : F401
from sklearn.metrics import mean_squared_error  # noqa : F401
from sklearn.model_selection import GridSearchCV  # noqa : F401
from sklearn.model_selection import StratifiedShuffleSplit  # noqa : F401
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

def load_data(input_path):
    """
    Load housing data from a CSV file.

    Parameters
    ----------
    input_path : str
        The path to the CSV file containing the housing data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded housing data.
    """
    logging.info(f"Loading data from {input_path}...")
    data = pd.read_csv(input_path)
    logging.info("Data loaded successfully.")
    return data

def preprocess_data(housing):
    """
    Preprocess the housing data for model training.

    Parameters
    ----------
    housing : pd.DataFrame
        The raw housing data.

    Returns
    -------
    tuple
        - pd.DataFrame: The preprocessed features for training.
        - pd.Series: The target labels (median house values).
    """
    logging.info("Preprocessing data...")

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )
    logging.info("Train-test split completed.")

    # Creating income category for stratified split
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    logging.info("Stratified train-test split completed.")

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # Feature engineering
    housing = strat_train_set.copy()
    logging.info("Performing feature engineering...")
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing_labels = strat_train_set["median_house_value"].copy()
    housing = strat_train_set.drop("median_house_value", axis=1)

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )
    logging.info("Data preprocessing completed.")

    return housing_prepared, housing_labels

def train_and_save_models(housing_prepared, housing_labels, output_path):
    """
    Train regression models and save them as pickle files.

    Parameters
    ----------
    housing_prepared : pd.DataFrame
        The preprocessed feature data for training.
    housing_labels : pd.Series
        The target labels (median house values).
    output_path : str
        The path to save the model pickle files.

    Returns
    -------
    None
    """
    logging.info("Training models...")

    # Train a Linear Regression model
    logging.info("Training Linear Regression model...")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    with open(os.path.join(output_path, "linear_regression.pkl"), "wb") as f:
        pickle.dump(lin_reg, f)
    logging.info("Linear Regression model trained and saved.")

    # Train a Decision Tree Regressor
    logging.info("Training Decision Tree Regressor...")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    with open(
        os.path.join(output_path, "decision_tree_regressor.pkl"), "wb"
    ) as f:
        pickle.dump(tree_reg, f)
    logging.info("Decision Tree Regressor trained and saved.")

    # Train a Random Forest Regressor with Randomized Search
    logging.info(
        "Training Random Forest Regressor with hyperparameter tuning..."
    )
    param_distribs = {
        "n_estimators": np.arange(1, 200),
        "max_features": np.arange(1, 8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    with open(
        os.path.join(output_path, "random_forest_regressor.pkl"), "wb"
    ) as f:
        pickle.dump(rnd_search.best_estimator_, f)
    logging.info("Random Forest Regressor trained and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train regression models on the housing dataset."
    )
    parser.add_argument(
        "--input_dataset_path",
        type=str,
        help="The input folder containing the dataset.",
        default="data/housing.csv",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        help="The output folder to save the model pickles.",
        default="artifacts",
    )
    args = parser.parse_args()

    # Load and preprocess data
    logging.info("Starting model training pipeline...")
    housing = load_data(args.input_dataset_path)
    housing_prepared, housing_labels = preprocess_data(housing)

    # Create output directory if it doesn't exist
    os.makedirs(args.model_output_path, exist_ok=True)

    # Train models and save them
    train_and_save_models(
        housing_prepared, housing_labels, args.model_output_path
    )
    logging.info(f"Models trained and saved to {args.model_output_path}")
