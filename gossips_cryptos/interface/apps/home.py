from pathlib import PureWindowsPath
import streamlit as st
import os
from os import environ
from dotenv import dotenv_values

import numpy as np
import pandas as pd
from PIL import Image
import datetime
#!/usr/bin/env python

from urllib.request import urlopen
import json
### THIS WILL GO ON THE FIRST PAGe


def get_jsonparsed_data():
    """
    Receive the content of ``url``, parse it as JSON and return the object.
    Parameters
    ----------
    url : str
    Returns
    -------
    dict
    """
    env_variables = dotenv_values(".env")
    api_key = env_variables['FINANCIAL_MODELLING_API_KEY']
    url = (f"https://financialmodelingprep.com/api/v3/quote/BTCUSD?apikey={api_key}")
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

def app():

    st.markdown("""# Smart Cryptocurrency Trading"""),
    st.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1631179116/699-cryptocurrency/Buy-Bitcoin_aciwoa.jpg')
    st.write("""## Our platform predicts whether the next day price of Bitcoin will go Up ⬆️ or Down ⬇️, taking into account sentiment analysis from the social media websites, Reddit and Twitter.""")
    st.write("""Do you wanna have a go yourself? Then navigate to the Predict page on the sidebar!""")

    # st.write('This page tells you some stuff about our model')
    st.write("""## Here is the latest BTC-USD Trading Price👇:""")

    # print(get_jsonparsed_data(url)[0]['price'])
    bitcoin_live_data = get_jsonparsed_data()[0]
    bitcoin_current_price = bitcoin_live_data ['price']
    bitcoin_change = bitcoin_live_data ['changesPercentage']


    # number = st.number_input('How much money are you investing???')

    # st.write('You are investing: ', number)


    col1, col2, col3 = st.columns(3)
    col1.metric("", "", "")
    col2.metric("BITCOIN", f"${round(bitcoin_current_price,3)}", f"{round(bitcoin_change,2)}%")
    col3.metric("", "", "")
    # perhaps insert here the current value of bitcoin?
