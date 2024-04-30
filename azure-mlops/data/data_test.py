import os
import numpy as np
import pandas as pd

# get absolute path of csv files from data folder
def get_absPath(filename):
    """Return the path of the notebooks folder"""
    path = os.path.abspath(
        os.path.join(
            os.path.dirname(
                __file__), os.path.pardir, "data", filename
        )
    )
    return path

# number of features
expected_columns = 10

# distribution of features in the training set
historical_mean = np.array([
    -3.63962254e-16,
    1.26972339e-16,
    -8.01646331e-16,
    1.28856202e-16,
    -8.99230414e-17,
    1.29609747e-16,
    -4.56397112e-16,
    3.87573332e-16,
    -3.84559152e-16,
    -3.39848813e-16,
    1.52133484e02,
])

historical_std = np.array([
    4.75651494e-02,
    4.75651494e-02,
    4.75651494e-02,
    4.75651494e-02,
    4.75651494e-02,
    4.75651494e-02,
    4.75651494e-02,
    4.75651494e-02,
    4.75651494e-02,
    4.75651494e-02,
    7.70057459e01,
])

# maximmal relative change in feature mean or standard deviation
# that we can tollerate
shift_tolerance = 3

def test_check_schema():
    datafile = get_absPath('diabetes.csv')
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns == expected_columns

def test_check_bad_schema():
    datafile = get_absPath('diabetes_bad_schema.csv')
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns != expected_columns

def test_check_missing_values():
    datafile = get_absPath("diabetes_missing_values.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    n_nan = np.sum(np.isnan(dataset.values))
    assert n_nan > 0

def test_check_distribution():
    datafile = get_absPath("diabetes_bad_dist.csv")
    # check that file exists
    assert os.path.exists(datafile)
    dataset = pd.read_csv(datafile)
    mean = np.mean(dataset.values, axis=0)
    std = np.mean(dataset.values, axis=0)
    assert (
        np.sum(abs(mean - historical_mean)
               > shift_tolerance * abs(historical_mean))
        or np.sum(abs(std - historical_std)
                  > shift_tolerance * abs(historical_std)) > 0
    )