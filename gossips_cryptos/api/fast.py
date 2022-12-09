
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gossips_cryptos.model.data import fgindex, prices
from gossips_cryptos.model.preprocess import data_cleaning,preprocess_features
from gossips_cryptos.model.model import init_model,fit_model
from gossips_cryptos.model.registry import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?crypto_currency=BTC

app.state.model = load_model()

@app.get("/predict")
def predictor(crypto='BTC',horizon=1):
    price = prices(crypto)
    index = fgindex()
    data_cleaned= data_cleaning(price,index)
    X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled,scaler_y = preprocess_features(data_cleaned,20,horizon)
    model = app.state.model
    predicted = model.predict(X_test_scaled)
    unscaled_pred = scaler_y.inverse_transform(predicted)
    return dict(price=float(unscaled_pred))


@app.get("/")
def root():
    return dict(greeting="Gossips and Cryptos")
