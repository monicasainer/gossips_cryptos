import pandas as pd
import datetime
import seaborn as sns
import requests
import os

# Greed and Fear index

## Getting data:

def fgindex() -> pd.DataFrame:
    """returns a dataframe of fear and greed index with columns:
    [value],[value_classification],[timestamp],[time_until_update] """

    url = 'https://api.alternative.me/fng/'
    params = {'limit':100000,'date_format':'world'}
    response = requests.get(url,params).json()
    fg = pd.DataFrame(response['data'])
    return fg


# Closing prices
def prices(crypto)-> pd.DataFrame:
    """returns a dataframe with columns:
    ['time_period_start'], ['time_period_end'], ['time_open'], ['time_close'],
    ['rate_open'], ['rate_high'], ['rate_low'], ['rate_close'] """

    today = datetime.datetime.today()
    key = os.environ.get('API_Key')
    url = 'https://rest.coinapi.io/v1/exchangerate/{crypto}/USD/history?period_id=1DAY&time_start=2018-02-01T00:00:00&time_end={today}T00:00:00&limit=100000'
    headers = {'X-CoinAPI-Key' : key} #API-key
    response = requests.get(url, headers=headers).json()
    crypto = pd.DataFrame(response)
    return crypto
