import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
"""
@app.get("/predict")
def predict(crypto_currency: "BTC"):

    #Data Retrieval
    fg_index_df = data.fgindex()
    BTC_prices_df = data.prices()

    #Data Cleaning
    df = preprocess.data_cleaning()
    df_cleaned = preprocess.window_data()
    model = model.init_baseline().compile_model()
    fit_model, history = model.train_model()
    fit_model = app.state.model
    X_processed = preprocess(X_pred)
    y_pred = fit_model.predict(X_processed)

    return y_pred
"""

@app.get("/")
def root():
    return dict(greeting="Gossips and Cryptos")
