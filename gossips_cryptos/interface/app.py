import streamlit as st
import time
import pandas as pd
import numpy as np
from gossips_cryptos.model.data import prices, currency_converter, fgindex
from gossips_cryptos.api.fast import predictor
from gossips_cryptos.model.preprocess import preprocess_features, data_cleaning
import mlflow

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

window_selection_c = st.sidebar.container()
with st.sidebar:
    with st.expander("Financial disclaimer", expanded=True):
        st.write("""
        The Content is for informational purposes only, you should not construe any such information or other material as legal, tax, investment, financial, or other advice.
        Nothing contained on our Site constitutes a solicitation, recommendation, endorsement, or offer by us or any third party service provider to buy or sell any securities or other
        financial instruments in this or in in any other jurisdiction in which such solicitation or offer would be unlawful under the securities laws of such jurisdiction.
        """)
    window_selection_c.markdown("## Specifics")
    CRYPTOS = np.array([ "Bitcoin", "Ethereum", "Tether", "Dogecoin", "Cardano" ])
    CURRENCIES = np.array(["US Dollar", "European Euro", "British Pound", "Swiss Franc", "Japanese Yen", "Canadian Dollar" ])

crypto_dict = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Tether": "USDT",
    "Dogecoin": "DOGE",
    "Cardano": "ADA"
}
currency_dict = {
    "US Dollar": "USD",
    "European Euro": "EUR",
    "Japanese Yen": "JPY",
    "British Pound": "GBP",
    "Swiss Franc": "CHF",
    "Canadian Dollar": "CAD"
}

selected_crypto = window_selection_c.selectbox('Select crypto currency', CRYPTOS)
selected_currency = window_selection_c.selectbox('Select fiat currency', CURRENCIES)

st.title(f'{selected_crypto} Price Prediction')
data = prices(crypto=crypto_dict[selected_crypto])
index = fgindex()

with st.spinner('Building historic graph...'):
    df1 = data["close"]
    dict = predictor(crypto=crypto_dict[selected_crypto])
    df2 = pd.DataFrame.from_dict(data=dict)
    df1 = df1[-len(df2):]
    df2 = df2.set_index(df1.index[-len(df2):])
    df3 = df2.assign(real_price = df1)
    df3 = df3.rename(columns={"real_price": "Real Prices", "price": "Predicted Prices"})

    st.line_chart(df3)

currency_df = currency_converter()

col1,col2,col3,col4 = st.columns(4)

with col1:
    with st.spinner('Retrieve Current Price From API...'):
        time.sleep(5)
        st.metric(label=f"Current Price in {selected_currency}:", \
            value=round(data["close"].iloc[-1]*currency_df[currency_dict[selected_currency]], 2), \
            delta= round((data["close"].iloc[-1]/data["close"].iloc[-2]-1)*100, 2))

with col2:
    with st.spinner('Predicting Tomorrow''s Price Through AI Model...'):
        pred_price_dict=predictor(crypto=crypto_dict[selected_crypto])
        tomorrows_price = pred_price_dict["price"][-1]
        st.metric(label=f"Tomorrow""s Price Prediction:", \
            value=round(tomorrows_price*currency_df[currency_dict[selected_currency]], 2), \
            delta=round((tomorrows_price/data["close"].iloc[-1]-1)*100,2))

with col3:
    with st.spinner('Calculating The Lowest Deviation'):
        #run_id2 = mlflow.search_runs(experiment_ids=["6964"])["run_id"][0]
        mae = mlflow.get_run(run_id="1df5425fbc7143de8ca3a4be56a47192").data.metrics["mae"] #hard coded
        time.sleep(3)
        st.metric(label=f"Lowest Price Prediction:", \
            value=round(tomorrows_price*currency_df[currency_dict[selected_currency]]*(1-mae), 2), \
            delta=round((tomorrows_price*currency_df[currency_dict[selected_currency]]*(1-mae)/(data["close"].iloc[-1]*currency_df[currency_dict[selected_currency]])-1)*100,2)[0])

with col4:
    with st.spinner('Calculating The Highest Deviation'):
        time.sleep(3)
        st.metric(label=f"Highest Price Prediction:", \
            value=round(tomorrows_price*currency_df[currency_dict[selected_currency]]*(1+mae), 2), \
            delta=round((tomorrows_price*currency_df[currency_dict[selected_currency]]*(1+mae)/(data["close"].iloc[-1]*currency_df[currency_dict[selected_currency]])-1)*100,2)[0])
