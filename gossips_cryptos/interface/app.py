'''
import streamlit as st

import datetime
import requests

'''
# TaxiFareModel front

This front queries the Le Wagon [taxi fare model API](https://taxifare.lewagon.ai/predict?pickup_datetime=2012-10-06%2012:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2)
'''

with st.form(key='params_for_api'):

    pickup_date = st.date_input('pickup datetime', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
    pickup_time = st.time_input('pickup datetime', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
    pickup_datetime = f'{pickup_date} {pickup_time}'
    pickup_longitude = st.number_input('pickup longitude', value=40.7614327)
    pickup_latitude = st.number_input('pickup latitude', value=-73.9798156)
    dropoff_longitude = st.number_input('dropoff longitude', value=40.6413111)
    dropoff_latitude = st.number_input('dropoff latitude', value=-73.7803331)
    passenger_count = st.number_input('passenger_count', min_value=1, max_value=8, step=1, value=1)

    st.form_submit_button('Make prediction')

params = dict(
    pickup_datetime=pickup_datetime,
    pickup_longitude=pickup_longitude,
    pickup_latitude=pickup_latitude,
    dropoff_longitude=dropoff_longitude,
    dropoff_latitude=dropoff_latitude,
    passenger_count=passenger_count)

wagon_cab_api_url = 'https://gossipscryptos-ghz7tzfv5a-ew.a.run.app/'
response = requests.get(wagon_cab_api_url)

prediction = response.json()

pred = prediction[0]

st.header(f'Fare amount: ${round(pred, 2)}')
'''
# pip install pandas numpy matplotlib streamlit pystan fbprophet cryptocmd plotly
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
from datetime import date, datetime

from cryptocmd import CmcScraper
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('Crypto Forecaster')

st.markdown("This application enables you to predict on the future value of any cryptocurrency (available on Coinmarketcap.com), for \
	any number of days into the future! The application is built with Streamlit (the front-end) and the Facebook Prophet model, \
	which is an advanced open-source forecasting model built by Facebook, running under the hood. You can select to train the model \
	on either all available data or a pre-set date range. Finally, you can plot the prediction results on both a normal and log scale.")

### Change sidebar color
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#D6EAF8,#D6EAF8);
    color: black;
}
</style>
""",
    unsafe_allow_html=True,
)

### Set bigger font style
st.markdown(
	"""
<style>
.big-font {
	fontWeight: bold;
    font-size:22px !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<p class='big-font'><font color='black'>Forecaster Settings</font></p>", unsafe_allow_html=True)

### Select ticker & number of days to predict on
selected_ticker = st.sidebar.text_input("Select a ticker for prediction (i.e. BTC, ETH, LINK, etc.)", "BTC")
period = int(st.sidebar.number_input('Number of days to predict:', min_value=0, max_value=1000000, value=365, step=1))
training_size = int(st.sidebar.number_input('Training set (%) size:', min_value=10, max_value=100, value=100, step=5)) / 100

### Initialise scraper without time interval
@st.cache
def load_data(selected_ticker):
	init_scraper = CmcScraper(selected_ticker)
	df = init_scraper.get_dataframe()
	min_date = pd.to_datetime(min(df['Date']))
	max_date = pd.to_datetime(max(df['Date']))
	return min_date, max_date

data_load_state = st.sidebar.text('Loading data...')
min_date, max_date = load_data(selected_ticker)
data_load_state.text('Loading data... done!')


### Select date range
date_range = st.sidebar.selectbox("Select the timeframe to train the model on:", options=["All available data", "Specific date range"])

if date_range == "All available data":

	### Initialise scraper without time interval
	scraper = CmcScraper(selected_ticker)

elif date_range == "Specific date range":

	### Initialise scraper with time interval
	start_date = st.sidebar.date_input('Select start date:', min_value=min_date, max_value=max_date, value=min_date)
	end_date = st.sidebar.date_input('Select end date:', min_value=min_date, max_value=max_date, value=max_date)
	scraper = CmcScraper(selected_ticker, str(start_date.strftime("%d-%m-%Y")), str(end_date.strftime("%d-%m-%Y")))

### Pandas dataFrame for the same data
data = scraper.get_dataframe()


st.subheader('Raw data')
st.write(data.head())

### Plot functions
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

def plot_raw_data_log():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.update_yaxes(type="log")
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

### Plot (log) data
plot_log = st.checkbox("Plot log scale")
if plot_log:
	plot_raw_data_log()
else:
	plot_raw_data()

### Predict forecast with Prophet
if st.button("Predict"):

	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	### Create Prophet model
	m = Prophet(
		changepoint_range=training_size, # 0.8
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality=False,
        seasonality_mode='multiplicative', # multiplicative/additive
        changepoint_prior_scale=0.05
		)

	### Add (additive) regressor
	for col in df_train.columns:
	    if col not in ["ds", "y"]:
	        m.add_regressor(col, mode="additive")

	m.fit(df_train)

	### Predict using the model
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)

	### Show and plot forecast
	st.subheader('Forecast data')
	st.write(forecast.head())

	st.subheader(f'Forecast plot for {period} days')
	fig1 = plot_plotly(m, forecast)
	if plot_log:
		fig1.update_yaxes(type="log")
	st.plotly_chart(fig1)

	st.subheader("Forecast components")
	fig2 = m.plot_components(forecast)
	st.write(fig2)
