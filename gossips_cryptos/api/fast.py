import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gossips_cryptos.model.data import fgindex, prices
from gossips_cryptos.model.preprocess import data_cleaning,window_data,folds,scaling,preprocess_features


import pandas as pd
#from gossips_cryptos.model import model, data, preprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?crypto_currency=BTC

@app.get("/predict")
def predict(crypto_currency ="BTC"):

    #Data Retrieval
    fg_index_df = fgindex()
    price = prices(crypto_currency)

    #Data Cleaning
    df = data_cleaning(price,fg_index_df)
    X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y = preprocess_features(df)
    model = model.init_baseline().compile_model()
    fit_model, history = model.train_model()
    fit_model = app.state.model
    y_pred = fit_model.predict(X_processed)

    return y_pred


@app.get("/")
def root():
    return dict(greeting="Gossips and Cryptos")
