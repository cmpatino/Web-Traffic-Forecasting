import pandas as pd
import numpy as np
from tqdm import tqdm


class DataConfig():

    def __init__(self, window_train=90, window_pred=62, batch_size=128):

        self.window_train = window_train
        self.window_pred = window_pred
        self.batch_size = batch_size


def generate_windows(features, data_config):

    pair = []
    low_boundary = features.shape[1] - (data_config.window_train + data_config.window_pred)
    for i in range(0, low_boundary, data_config.window_pred):
        X = features[:, i:i + data_config.window_train, :]
        Y = features[:, i + data_config.window_train:i + data_config.window_train + data_config.window_pred,:]
        #median = np.median(X, axis=1, keepdims=True)
        #X = X - median
        #Y = Y - median
        pair.append([X, Y])

    return pair


def generator(pairs, window_train, windows_pred, batch_size):
    # epochs is the len of pairs
    while (1):
        for X, Y in pairs:
            m, n, _ = X.shape
            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size, :, :]
                y_batch = Y[i:i + batch_size, :, 0]
                assert pd.isna(X_batch).sum().sum() == 0
                yield X_batch, y_batch


def get_lag_optimized(timeseries, lag_values, n_days):
    """Generate lagging features for a single timeseries

    Arguments:
        timeseries {np.ndarray} -- Array with timeseries with shape (n_days, 1)
        lag_values {list} -- lags in days for new features
        n_days {int} -- number of days in original timeseries

    Returns:
        np.ndarray -- Array of shape (n_days, n_features)
    """

    n_features = len(lag_values) + 1 + 2
    median = np.nanmedian(timeseries)
    mean = np.nanmean(timeseries)
    timeseries[np.isnan(timeseries)] = median
    timeseries = timeseries.astype('int32')
    timeseries = np.log1p(timeseries)
    assert np.isnan(timeseries).sum() == 0
    features = np.empty((n_days, n_features))
    for i in range(timeseries.shape[0]):
        x = np.zeros((n_features))
        x[0] = timeseries[i]
        for j, lag in enumerate(lag_values):

            lag_index = i - lag

            if lag_index < 0:
                x[j + 1] = median
            else:
                x[j + 1] = timeseries[lag_index]

        x[-2] = median
        x[-1] = mean
        features[i] = x

    return features


def create_features_optimized(DATA_PATH, MATRIX_PATH, generate_matrix=False):
    """Create lagging features for all timeseries

    Arguments:
        DATA_PATH {str} -- path to data location
        MATRIX_PATH {str} -- path to matrix

    Returns:
        np.ndarray -- Array with shape (n_series, n_days, n_features)
    """

    if not generate_matrix:
        print('Loading Matrix from Memory')
        X_train = np.load(MATRIX_PATH + 'feature_matrix.npy')
        return X_train

    train_df = pd.read_csv(DATA_PATH + 'train_2.csv')
    n_series = train_df.shape[0]
    n_days = train_df.shape[1] - 1
    lag_values = [365, 180, 90]
    n_features = len(lag_values) + 1 + 2

    train_array = train_df.drop('Page', axis=1).values
    del train_df

    X_train = np.empty((n_series, n_days, n_features))
    for i in tqdm(range(len(train_array)), ascii=True):
        x_t = train_array[i]
        features = get_lag_optimized(x_t, lag_values, n_days)
        X_train[i, ] = features

    print('Writing Matrix File')
    X_train = X_train.astype('int32')
    np.save(MATRIX_PATH + 'feature_matrix.npy', X_train)
    return X_train
