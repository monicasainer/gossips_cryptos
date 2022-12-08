import pandas as pd
import numpy as np
from gossips_cryptos.model.data import prices, fgindex
from sklearn.preprocessing import  MinMaxScaler

# from gossips_cryptos.model.preprocess import data_cleaning


def data_cleaning(raw_data_prices,raw_data_sentiment):
    '''The function returns a dataframe containing:
    price: the historical crypto price
    index: the Grid/fear index value
    '''
    #cleaning the price data

    data = raw_data_prices # raw_data_prices = prices (crypto)
    data = data['close']

    #cleaning the sentiment data
    sentiment_data = raw_data_sentiment # raw_data_sentiment = fgindex()
    sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
    sentiment_data['value'] = sentiment_data['value'].astype('float')
    fg= pd.DataFrame(sentiment_data[['value', 'timestamp']])
    fg.set_index('timestamp', inplace=True)


    #merging the price and sentiment data
    df = fg.join(data)

    #cleaning the merged dataframe
    df.dropna(inplace=True)
    df.rename(columns = {'close': 'price', 'value': 'index'}, inplace = True)
    df = df.tail(-1)
    return df


def window_data(cleaned_data,window=20,horizon=1):
    """returns two arrays:
    X : Array of lists. Each list contains n_window observations of features.
    y: Array of lists. Each list contains the price of obs n_window + 1
    """
    df = cleaned_data
    cols_to_find =['index','price']
    feature_column = [df.columns.get_loc(col) for col in cols_to_find] #CHANGE
    target_column = df.columns.get_loc('price')
    X = []
    y = []

    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window),feature_column]
        target = df.iloc[(i + window+ horizon), target_column]
        X.append(features)
        y.append(target)

    return np.array(X), np.array(y).reshape(-1, 1)


def folds(X,y,window=20):
    """ returns four arrays:
    X_train : array of lists with the 70% of the observed feature values
    X_test : array of lists with the 30% of the observed feature values
    y_train : array of lists with the 70% of the observed target values
    y_test : array of lists with the 30% of the observed target values
    """
    split = int(.7 * len(X))
    X_train = X[:split - 1]
    X_test = X[split:]

    # y split
    y_train = y[:split - 1]
    y_test = y[split:]

    return X_train,X_test,y_train,y_test

def scaling(X_train,X_test,y_train,y_test):
    """ returns four arrays:
    X_train_scaled : array of lists with the 70% of the observed feature values scaled,
    X_test_scaled : array of lists with the 30% of the observed feature values scaled,
    y_train_scaled : array of lists with the 70% of the observed target values scaled,
    y_test_scaled : array of lists with the 30% of the observed target values scaled.
    """

    scaler = MinMaxScaler()
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    X_train = X_train_scaled.reshape((X_train_scaled.shape[0], 20, 2))
    X_test = X_test_scaled.reshape((X_test_scaled.shape[0],  20, 2))

    return X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled


def preprocess_features(cleaned_data: pd.DataFrame,window=20) -> np.ndarray:
    data_scaled = scaling(cleaned_data,window=10)
    data = window_data(data_scaled,window=10)
    X_train,X_test,y_train,y_test = folds(data ,window)
    return X_train,X_test,y_train,y_test
