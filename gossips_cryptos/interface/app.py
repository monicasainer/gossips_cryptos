import streamlit as st
import datetime
import pandas as pd
import numpy as np
from gossips_cryptos.model.data import prices
import plotly.graph_objects as go
from gossips_cryptos.api.fast import predictor


st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def nearest_business_day(DATE: datetime.date):
    """
    Takes a date and transform it to the nearest business day
    """
    if DATE.weekday() == 5:
        DATE = DATE - datetime.timedelta(days=1)

    if DATE.weekday() == 6:
        DATE = DATE + datetime.timedelta(days=1)
    return DATE


window_selection_c = st.sidebar.container()
window_selection_c.markdown("## Specifics")
CRYPTOS = np.array([ "Bitcoin", "Ethereum", "Tether", "Dogecoin", "Cardano" ])

crypto_dict = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Tether": "USDT",
    "Dogecoin": "DOGE",
    "Cardano": "ADA"
}

selected_crypto = window_selection_c.selectbox('Select crypto currency', CRYPTOS)

st.title(f'{selected_crypto} Price Prediction')

data = prices(crypto=crypto_dict[selected_crypto])
st.line_chart(data["close"].iloc[-365:-1])

data_pred = predictor(crypto=crypto_dict[selected_crypto])
print[len(data["close"])]
print[len(data_pred["price"])]

data["pred_close"] = data_pred["price"]
st.line_chart(data["pred_close"].iloc[-365:-1])
