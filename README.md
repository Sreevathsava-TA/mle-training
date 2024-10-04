# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.
Here is the updated **README.md** file with the correct repository link and Python filename:

---

# Housing Price Prediction

This project aims to predict housing prices based on various features such as location, number of rooms, and proximity to the ocean. It utilizes machine learning models like Linear Regression, Decision Trees, and Random Forests with hyperparameter tuning.

## Table of Contents
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Required Libraries](#required-libraries)
- [Running the Code](#running-the-code)
- [Model Performance](#model-performance)
  
## Dataset
The dataset is from the [California housing prices dataset](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz) and consists of data about the housing market in California.

## Setup Instructions

### 1. Clone the Repository

To get started, clone this repository:

```bash
git clone https://github.com/Sreevathsava-TA/mle-training.git
cd mle-training
```

### 2. Install Dependencies

Ensure you have `Python 3.12` installed. Use conda install to install the required libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`

```
## To excute the script
python nonstandardcode.py
