import pandas as pd
import datetime
import seaborn as sns
import requests
import os

# Greed and Fear index

## Getting data:

def fgindex() -> pd.DataFrame:
    """returns a dataframe of fear and Greed index with columns:
    [value],[value_classification],[timestamp],[time_until_update] """

    url = 'https://api.alternative.me/fng/'
    params = {'limit':100000,'date_format':'world'}
    response = requests.get(url,params).json()
    fg = pd.DataFrame(response['data'])
    return fg
# # Converting string to timestamp.
# Cleaning part: fg['timestamp'] = pd.to_datetime(fg['timestamp'],dayfirst=True)




# Closing prices
def prices(crypto):
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


## Cleaning part
##BTC_EUR.rename(columns={'time_close':'timestamp'},inplace=True)

##BTC_EUR['timestamp'] = BTC_EUR['timestamp'].apply(lambda x: x[0:10]) #Removing the time

##BTC_EUR['timestamp']=pd.to_datetime(BTC_EUR['timestamp'],yearfirst=True)
##BTC_EUR=BTC_EUR[['timestamp','rate_close']] #Getting two columns


# Index and Crypto together
#df=fg.merge(BTC_EUR,how='left',on='timestamp')
#df



# Exchange rates base asset identifier
#url = 'https://rest.coinapi.io/v1/assets'
#headers = {'X-CoinAPI-Key' : 'CC32A4FC-471F-4200-8788-6C980B2C8CAF'} #API-key
#response = requests.get(url, headers=headers).json()


#market=[i['asset_id'] for i in response]
#sns.scatterplot(df['value_classification'],df['rate_close'])