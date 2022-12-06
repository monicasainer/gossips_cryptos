import pandas as pd
import datetime
import seaborn as sns
import requests
import os

#Currency Converter
def currency_converter()-> pd.DataFrame:
    """Finding a fitting currency conversion API (Based on USD) and requesting the data as a function """

    APIkey = "j4YVbY4DsPe2bRdqchwWzvccCljEhg2EhlPyfrEN"
    url = f"https://api.freecurrencyapi.com/v1/latest?apikey={APIkey}"
    response = requests.get(url).json()
    df = pd.DataFrame(response['data'], index=[0])
    return df


#FearGridIndex
def fgindex() -> pd.DataFrame:
    """returns a dataframe of fear and greed index with columns:
    [value],[value_classification],[timestamp],[time_until_update] """

    url = 'https://api.alternative.me/fng/'
    params = {'limit':100000,'date_format':'world'}
    response = requests.get(url,params).json()
    fg = pd.DataFrame(response['data'])
    return fg


#Closing prices
def prices(crypto)-> pd.DataFrame:
    """returns a dataframe with columns:
    ['time_period_start'], ['time_period_end'], ['time_open'], ['time_close'],
    ['rate_open'], ['rate_high'], ['rate_low'], ['rate_close'] """
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    key = os.environ.get('API_Key')
    url = f'https://rest.coinapi.io/v1/exchangerate/{crypto}/USD/history?period_id=1DAY&time_start=2018-02-01T00:00:00&time_end={today}T00:00:00&limit=100000'
    headers = {'X-CoinAPI-Key' : key} #API-key
    response = requests.get(url, headers=headers).json()
    crypto = pd.DataFrame(response)
    return crypto
