import pandas as pd
import datetime
import seaborn as sns

# Greed and Fear index

## Getting data:
import requests

url = 'https://api.alternative.me/fng/'
params = {'limit':100000,'date_format':'world'}
response = requests.get(url,params).json()

## Turning into DataFrame
fg= pd.DataFrame(response['data'])
fg['timestamp'] = pd.to_datetime(fg['timestamp'],dayfirst=True) # Converting string to timestamp.



# Closing prices

url = 'https://rest.coinapi.io/v1/exchangerate/BTC/USD/history?period_id=1DAY&time_start=2018-02-01T00:00:00&time_end=2022-12-02T00:00:00&limit=100000'
headers = {'X-CoinAPI-Key' : 'Replace_by_your_api_key'} #API-key
response = requests.get(url, headers=headers).json()

## BTC-EUR
BTC_EUR=pd.DataFrame(response)
BTC_EUR.tail()

BTC_EUR.rename(columns={'time_close':'timestamp'},inplace=True)

BTC_EUR['timestamp'] = BTC_EUR['timestamp'].apply(lambda x: x[0:10]) #Removing the time

BTC_EUR['timestamp']=pd.to_datetime(BTC_EUR['timestamp'],yearfirst=True)
BTC_EUR=BTC_EUR[['timestamp','rate_close']] #Getting two columns


# Index and Crypto together
df=fg.merge(BTC_EUR,how='left',on='timestamp')
df



# Exchange rates base asset identifier
url = 'https://rest.coinapi.io/v1/assets'
headers = {'X-CoinAPI-Key' : 'CC32A4FC-471F-4200-8788-6C980B2C8CAF'} #API-key
response = requests.get(url, headers=headers).json()


market=[i['asset_id'] for i in response]
sns.scatterplot(df['value_classification'],df['rate_close'])
