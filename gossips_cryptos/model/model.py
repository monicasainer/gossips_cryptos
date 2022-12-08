## Baseline Model
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Sequence
from tensorflow.keras.layers import Lambda
from tensorflow.keras import models, layers, optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping
from gossips_cryptos.model.preprocess import preprocess_features #scaler_y


def init_baseline():
    """ returns a model ready to initialize.
    It predicts the last seen value for the future value forecast.
    """
    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x[:,-1,0,None]))

    return model

def init_model(y_train_scaled):
    """RNN architecture and compile model
    """

    model = models.Sequential()

    ## 1.1 - Recurrent Layer
    model.add(layers.LSTM(64,
                          activation='tanh',
                          return_sequences = False,
                          recurrent_dropout = 0.2))

    ## 1.2 - Predictive Dense Layers
    output_length = y_train_scaled.shape[1]
    model.add(layers.Dense(output_length, activation='relu'))

    ##2.0 Defining the optimizer
    adam = optimizers.Adam(learning_rate = 0.0001)
    model.compile(loss = 'mse',
                  optimizer = adam,
                  metrics = ['mae'])

    return model


def fit_model(X_train_scaled, y_train_scaled, model: tf.keras.Model, verbose=1) -> Tuple[tf.keras.Model, dict]:

    """
    Fit model and return a the tuple (fitted_model, history)
    """

    es = EarlyStopping(monitor = "val_loss",
                      patience = 10,
                      mode = "min",
                      restore_best_weights = True)


    history = model.fit(X_train_scaled, y_train_scaled,
                        validation_split = 0.3,
                        shuffle = False,
                        batch_size = 16,
                        epochs = 100,
                        callbacks = [es],
                        verbose = verbose)

    return model, history



## Alternative in gossips_cryptos/api/fast.py
## Is it actually needed?
"""
def predict(X_test_scaled,model: tf.keras.Model,scaler_y):
    predicted = model.predict(X_test_scaled)
    unscaled_pred = scaler_y.inverse_transform(predicted)
    return unscaled_pred
"""
