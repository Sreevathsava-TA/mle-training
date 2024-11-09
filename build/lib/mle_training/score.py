import argparse
import os
import pickle
import tarfile  # noqa : F401

import numpy as np
import pandas as pd
from scipy.stats import randint  # noqa : F401
from six.moves import urllib  # noqa : F401
from sklearn.ensemble import RandomForestRegressor  # noqa : F401
from sklearn.impute import SimpleImputer  # noqa : F401
from sklearn.linear_model import LinearRegression  # noqa : F401
from sklearn.metrics import mean_absolute_error  # noqa : F401
from sklearn.metrics import mean_squared_error  # noqa : F401
from sklearn.model_selection import GridSearchCV  # noqa : F401
from sklearn.model_selection import RandomizedSearchCV  # noqa : F401
from sklearn.model_selection import StratifiedShuffleSplit  # noqa : F401
from sklearn.model_selection import train_test_split  # noqa : F401
from sklearn.tree import DecisionTreeRegressor  # noqa : F401
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_data(input_path):
    """
    Load dataset from a given CSV file.

    Parameters
    ----------
    input_path : str
        The file path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the loaded data.

    Raises
    ------
    Exception
        If the file cannot be loaded (e.g., file not found or invalid format).
    """
    logging.info(f"Loading data from {input_path}")
    try:
        data = pd.read_csv(input_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(housing):
    """
    Preprocess the housing dataset by handling missing values,
    adding new features, and encoding categorical variables.

    Parameters
    ----------
    housing : pd.DataFrame
        The raw housing data with features and target values.

    Returns
    -------
    X_test_prepared : pd.DataFrame
        The prepared features for the test dataset after preprocessing.

    y_test : pd.Series
        The target values (median_house_value) for the test dataset.

    Notes
    -----
    This function performs stratified splitting of the data based on income category
    and generates new features such as rooms per household and population per household.
    """
    logging.info("Preprocessing the data.")

    # Stratified Shuffle Split
    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    # Drop income_cat
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    logging.info("Fitting imputer on numerical data.")
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
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test[["ocean_proximity"]], drop_first=True)
    )
    logging.info("Data preprocessing complete.")
    return X_test_prepared, y_test


def score_models(model_folder, dataset_folder, output_folder):
    """
    Load a trained model, preprocess the test data, and evaluate the model.

    Parameters
    ----------
    model_folder : str
        The file path to the trained model to be evaluated.

    dataset_folder : str
        The file path to the dataset to score the model on.

    output_folder : str
        The folder where the prediction results will be saved.

    Raises
    ------
    Exception
        If the model cannot be loaded or an error occurs during scoring.
    """
    logging.info(f"Scoring models with dataset from {dataset_folder}.")

    # Load the test data
    housing = load_data(dataset_folder)

    # Load the model
    logging.info(f"Loading model from {model_folder}")
    try:
        with open(model_folder, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    # Preprocess the test data
    X_test_prepared, y_test = preprocess_data(housing)

    # Make predictions
    final_predictions = model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    logging.info(f"Model evaluation results:")
    logging.info(f"Mean Squared Error: {final_mse}")
    logging.info(f"Root Mean Squared Error: {final_rmse}")

    # Save the predictions
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "predictions.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(final_predictions, f)

    logging.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    """
    Main entry point of the script. It parses command-line arguments
    and calls the scoring function to evaluate the model on the provided dataset.
    """
    parser = argparse.ArgumentParser(
        description="Score models on the housing dataset."
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="The folder containing the trained models.",
        default="artifacts/random_forest_regressor.pkl",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        help="The folder containing the dataset for scoring.",
        default="data/housing.csv",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="The folder to save the prediction outputs.",
        default="result",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Score the models
    score_models(args.model_folder, args.dataset_folder, args.output_folder)
    logging.info(f"Predictions saved to {args.output_folder}")
