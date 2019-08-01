import pandas as pd
import numpy as np

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
    for i in range(timeseries.shape[0]):
        x = np.zeros((n_features))
        x[0] = timeseries[i]
        for j, lag in enumerate(lag_values):

            lag_index = i - lag

            if lag_index < 0:
                x[j + 1] = 0
            else:
                x[j + 1] = timeseries[lag_index]
        features[i] = x

    return features


def create_features_optimized(DATA_PATH, n_series, n_days, n_features):
    """Create lagging features for all timeseries

    Arguments:
        DATA_PATH {str} -- path to data location
        n_series {int} -- number of timeseries in data
        n_days {int} -- number of days in timeseries
        n_features {int} -- number of total features

    Returns:
        np.ndarray -- Array with shape (n_series, n_days, n_features)
    """

    train_df = pd.read_csv(DATA_PATH + 'train_2.csv')
    train_array = train_df.drop('Page', axis=1).values

    X_train = np.empty((n_series, n_days, n_features))
    for i in range(len(train_array)):
        if (i % 10000) == 0:
            print(i)
        x_t = train_array[i]
        features = get_lag_optimized(x_t, [365, 180, 90], n_days, n_features)

        X_train[i, ] = features

    np.save('features_array.npy', X_train)
    return X_train
