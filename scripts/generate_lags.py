import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_PATH = '../data/input/'


def get_lag_optimized(timeseries, lag_values, n_days):
    """Generate lagging features for a single timeseries

    Arguments:
        timeseries {np.ndarray} -- Array with timeseries with shape (n_days, 1)
        lag_values {list} -- lags in days for new features
        n_days {int} -- number of days in original timeseries

    Returns:
        np.ndarray -- Array of shape (n_days, n_features)
    """

    n_features = len(lag_values) + 1
    features = np.empty((n_days, n_features))
    median = np.nanmedian(timeseries)
    for i in range(timeseries.shape[0]):
        x = np.zeros((n_features))
        x[0] = timeseries[i]
        for j, lag in enumerate(lag_values):

            lag_index = i - lag

            if lag_index < 0:
                x[j + 1] = median
            else:
                x[j + 1] = timeseries[lag_index]

        features[i] = x

    return features


def create_features_optimized(DATA_PATH):
    """Create lagging features for all timeseries

    Arguments:
        DATA_PATH {str} -- path to data location

    Returns:
        np.ndarray -- Array with shape (n_series, n_days, n_features)
    """

    train_df = pd.read_csv(DATA_PATH + 'train_2.csv')
    n_series = train_df.shape[0]
    n_days = train_df.shape[1] - 1
    lag_values = [365, 180, 90]
    n_features = len(lag_values) + 1

    train_array = train_df.drop('Page', axis=1).values

    X_train = np.empty((n_series, n_days, n_features))
    for i in tqdm(range(len(train_array))):
        x_t = train_array[i]
        features = get_lag_optimized(x_t, lag_values, n_days)
        X_train[i, ] = features

    np.save('features_array.npy', X_train)
    return X_train


n_features = 5
X_train = create_features_optimized(DATA_PATH)
