---

# Housing Price Prediction

This project aims to predict median housing prices based on various features like location, number of rooms, and proximity to the ocean. It employs machine learning models such as Linear Regression, Decision Trees, and Random Forests with hyperparameter tuning for improved accuracy.

## Table of Contents
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Required Libraries](#required-libraries)
- [Running the Code](#running-the-code)
- [Model Performance](#model-performance)

## Dataset
The dataset used for this project is the [California Housing Prices dataset](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz). It contains information about the housing market in California, including features like house age, income, and proximity to the ocean.

## Setup Instructions

### 1. Clone the Repository

To get started with the project, clone this repository:

```bash
git clone https://github.com/Sreevathsava-TA/mle-training.git
cd mle-training
```

### 2. Create a Conda Environment

Create a new Conda environment using the provided `env.yml` file:

```bash
conda env create -f env.yml
conda activate mle-dev
```

### 3. Install Required Libraries

Make sure to install the required Python libraries. If not using the `env.yml`, you can install them manually:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### 4. Download the Dataset

The script automatically downloads the dataset from the following link:

```text
https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz
```

You can also manually download and extract the data using this link if needed.
## 5.How to setup a virtual environment
```bash 
conda env create -f environment.yaml
conda activate env
```
## 6. Made the Code Packageable
```bash

repo-name/
├── src/
│   └── your_package/
│       ├── __init__.py
│       ├── ingest_data.py
│       ├── train.py
│       └── score.py
├── tests/
├── README.md
├── env.yaml
├── setup.py
```

## Installing package
```bash 
python setup.py sdist bdist_wheel
pip install .
```
## Running the Scripts

### Data Ingestion: Run ingest_data.py 
```bash
python src/ingest_data.py
```
By default the datsets are saved into the data folder

By using the below code we can save the datasets in the different folder

```bash 
python src/mle_training/ingest_data.py --output_path folder_name
```

### Trining the models: Run train.py
```bash
python src/train.py
```
The trained models will be saved in the folder named artifacts
### Score: Run score.py
```bash
python src/score.py
```
The score is saved in the results folder

## Running tests

### Unit tests

#### Test data ingestion : Run test_ingest_data
```bash 
pytest -v tests/unit_tests/test_ingest_data.py
```
#### Test trained models : Run test_train
```bash 
pytest -v tests/unit_tests/test_train.py
```
#### Test data ingestion : Run test_score
```bash 
pytest -v tests/unit_tests/test_score.py
```

### Functional tests

#### Test data ingestion : Run test_ingest_data
```bash 
pytest -v tests/functional_tests/test_ingest_data.py
```
#### Test trained models : Run test_train
```bash 
pytest -v tests/functional_tests/test_train.py
```
#### Test data ingestion : Run test_score
```bash 
pytest -v tests/functional_tests/test_score.py
```

## Generate Sphinx build
``` bash
cd docs
make html
```
## Model Performance

The final models are evaluated using Mean Squared Error (MSE) to assess prediction accuracy. The Random Forest model typically yields better performance due to its ability to capture complex relationships in the data.

## Screenshots

Below are some sample outputs from the model training:

<img width="1440" alt="Model Output 1" src="https://github.com/user-attachments/assets/5eb0109c-f365-4b85-bac6-4605158d07b7">

<img width="1440" alt="Model Output 2" src="https://github.com/user-attachments/assets/dbbfcbd6-0840-4532-942d-291ac2ec55ee">

---
