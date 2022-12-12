import numpy as np
import pandas as pd
from gossips_cryptos.model.data import fgindex, prices
from gossips_cryptos.model.preprocess import data_cleaning, preprocess_features
from gossips_cryptos.model.model import init_model, fit_model, evaluate_model
from gossips_cryptos.model.registry import get_model_version
from gossips_cryptos.model.registry import load_model, save_model

def preprocess():
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """
    raw_data_prices = prices()
    raw_data_sentiment = fgindex()

    data_cleaned = data_cleaning(raw_data_prices,raw_data_sentiment)


    X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y = preprocess_features(data_cleaned)

    return X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y

def train():
    """
    Train a new model on the full (already preprocessed) dataset ITERATIVELY, by loading it
    chunk-by-chunk, and updating the weight of the model after each chunks.
    Save final model once it has seen all data, and compute validation metrics on a holdout validation set
    common to all chunks.
    """

    X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y = preprocess()

    model = load_model()  # production model
    model = init_model(y_train_scaled)
    model, history=fit_model(X_train_scaled, y_train_scaled, model)
    predicted = model.predict(X_test_scaled)
    unscaled_pred = scaler_y.inverse_transform(predicted)
    unscaled_y = scaler_y.inverse_transform(y_test_scaled)


    val_mae = abs(unscaled_pred-unscaled_y).mean()
    metrics_dict = {'val_mae':val_mae}
    #Save model
    save_model(model=model,metrics=metrics_dict)
    return abs(unscaled_pred[-1]-unscaled_y[-1])


def pred() -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    from gossips_cryptos.model.registry import load_model
    price = prices()
    index = fgindex()
    data_cleaned= data_cleaning(price,index)
    X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y = preprocess_features(data_cleaned,40,1,0.8)
    model = load_model()
    predicted = model.predict(X_test_scaled)
    unscaled_pred = scaler_y.inverse_transform(predicted)
    return dict(price=np.concatenate(unscaled_pred, axis=0).tolist())


def pred_new_day() -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    from gossips_cryptos.model.registry import load_model
    full_price = prices()
    index = fgindex()
    data_cleaned= data_cleaning(full_price,index)
    X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y = preprocess_features(data_cleaned,40,1,0.8)

    model = load_model()
    predicted = model.predict(X_test_scaled[-1:])
    unscaled_pred = scaler_y.inverse_transform(predicted)
    return dict(predicted_price=np.concatenate(unscaled_pred, axis=0).tolist(),actual_price=data_cleaned['price'][-1])



if __name__ == '__main__':
    mae,y,prediction=train()
    # #pred()
    predicted_dict=pred_new_day()#['predicted_price'][0]
    print(predicted_dict,mae,y,prediction)
