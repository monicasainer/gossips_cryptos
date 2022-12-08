## Baseline Model
import numpy as np
from tensorflow.keras.layers import Lambda
from tensorflow.keras import models, layers, optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping

def init_baseline()-> Model:
    """ returns a model ready to initialize.
    It predicts the last seen value for the future value forecast.
    """
    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x[:,-1,0,None]))

    return model

def compile_model(model: Model, learning_rate=0.1) -> Model:
    """
    Compile the Neural Network
    """
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])
    return model


def train_model(model: Model,
                X_train: np.ndarray,
                y_train: np.ndarray,
                batch_size=64,  #Pending
                patience=2,
                # validation_split=0.3,
                validation_data=None) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """


    es = EarlyStopping(monitor="val_loss", #The monitored value
                       patience=patience,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X_train,
                        y_train,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=100,  #Pending
                        batch_size=batch_size, #Pending
                        callbacks=[es],
                        verbose=0)

    return model, history

def evaluate_model(model: Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on test dataset
    """

    if model is None:
        return None

    metrics = model.evaluate(
        x=X_test,
        y=y_test,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    mae = metrics["mae"]

    return metrics
