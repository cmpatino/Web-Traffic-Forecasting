from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense


class ModelConfig():

    def __init__(self, model_name, epochs=5):

        self.model_name = model_name
        self.epochs = epochs


def get_baseline_model(data_config):

    model = Sequential()
    model.add(CuDNNLSTM(100, input_shape=(data_config.window_train, 4)))
    model.add(Dense(data_config.window_pred))
    model.compile(optimizer='adam', loss='mse')

    print(model.summary())

    return model