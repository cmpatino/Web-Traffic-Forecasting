import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import data_utils as du
import models

batch_size = 64


def train(DATA_PATH, MATRIX_PATH, MODEL_PATH, data_config, model_config):

    features = du.create_features_optimized(DATA_PATH, MATRIX_PATH, generate_matrix=True)

    pair = du.generate_windows(features, data_config)
    m = pair[0][0].shape[0]

    train_generator = du.generator(pair,
                                   data_config.window_train,
                                   data_config.window_pred,
                                   data_config.batch_size)
    print('*'*50, 'Training Model',  '*'*50)
    model = models.get_baseline_model(data_config)

    steps_per_epoch = (m // data_config.batch_size)*11
    history = model.fit_generator(train_generator,
                                  epochs=model_config.epochs,
                                  steps_per_epoch=steps_per_epoch)

    model.save_weights(MODEL_PATH + f'weight_{model_config.model_name}.h5')
    with open(MODEL_PATH + f'history_{model_config.model_name}.pkl',
              'wb') as f:
        pickle.dump(history.history, f)

    return features, model


def predict(model, features, model_config, DATA_PATH):

    print('*'*50, 'Generating Predictions',  '*'*50)
    train = pd.read_csv(DATA_PATH + 'train_2.csv')
    keys = train.Page
    del train
    preds = []
    for i in tqdm(range(len(keys)), ascii=True):
        preds.append(model.predict(features[i, -90:, :][None, ...]))

    #pred_median = np.median(features[:, -90:, :],
    #                        axis=1, keepdims=True)[..., None]
    preds = np.array(preds)
    preds = np.expm1(preds)
    #preds = preds + pred_median

    preds = np.array(preds).squeeze()
    subm = pd.read_csv(DATA_PATH + 'key_2.csv')
    dict_val = dict(zip(list(range(62)),
                        sorted(subm.iloc[0:62].Page.str[-10:])))

    aux = pd.DataFrame(preds)
    aux = aux.set_index(keys)
    aux = aux.rename(columns=dict_val)
    aux = aux.unstack().reset_index()
    aux['Name'] = aux.Page + '_' + aux.level_0
    aux = aux[['Name', 0]]

    sub = pd.merge(subm, aux, left_on='Page', right_on='Name')
    sub[['Id', 0]].rename(columns={0: 'Visits'})\
                  .to_csv(f'../submissions/submission_{model_config.model_name}.csv',
                          index=False)


def train_and_predict():

    DATA_PATH = '../data/input/'
    MATRIX_PATH = '../data/'
    MODEL_PATH = '../models/'

    data_config = du.DataConfig()

    model_name = 'baseline'
    model_config = models.ModelConfig(model_name=model_name)

    features, model = train(DATA_PATH, MATRIX_PATH, MODEL_PATH,
                            data_config, model_config)

    predict(model, features, model_config, DATA_PATH)


train_and_predict()
