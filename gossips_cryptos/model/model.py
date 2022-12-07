## Baseline Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics


def init_baseline():
    """ returns a model ready to initialize.
    It predicts the last seen value for the future value(s).
    """

    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x[:,-1,0,None]))

    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model

def run_baseline_model(X_test,y_test_scaled):
    baseline_model = init_baseline()
    baseline_score = baseline_model.evaluate(X_test, y_test_scaled)
    return baseline_score
