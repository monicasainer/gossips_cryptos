import pandas as pd
import numpy as np
from gossips_cryptos.model.data import prices, fgindex
from sklearn.preprocessing import  StandardScaler
# from gossips_cryptos.model.preprocess import data_cleaning


def data_cleaning(crypto):
    '''The function returns a dataframe containing:
    price: the historical crypto price
    index: the Grid/fear index value
    '''
    #cleaning the price data

    BTC_USD = prices(crypto)
    BTC_USD= BTC_USD['close']

    #cleaning the sentiment data
    sentiment_data = fgindex()
    sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
    fg= pd.DataFrame(sentiment_data[['value', 'timestamp']])
    fg.set_index('timestamp', inplace=True)


    #merging the price and sentiment data
    df = fg.join(BTC_USD)

    #cleaning the merged dataframe
    df.dropna(inplace=True)
    df.rename(columns = {'close': 'price', 'value': 'index'}, inplace = True)

    return df



def window_data(crypto='BTC',window=10):
    """returns two arrays:
    X : Array of lists. Each list contains n_window observations of features.
    y: Array of lists. Each list contains the price of obs n_window + 1
    """
    df = data_cleaning(crypto)
    feature_column = df.columns.get_loc('index')
    target_column = df.columns.get_loc('price')
    X = []
    y = []

    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_column]
        target = df.iloc[(i + window), target_column]
        X.append(features)
        y.append(target)

    return np.array(X), np.array(y).reshape(-1, 1)


def folds(crypto='BTC',window=10):
    """ returns four arrays:
    X_train : array of lists with the 70% of the observed feature values
    X_test : array of lists with the 30% of the observed feature values
    y_train : array of lists with the 70% of the observed target values
    y_test : array of lists with the 30% of the observed target values
    """

    X, y = window_data(crypto,window)
    split = int(.7 * len(X))
    X_train = X[:split - 1]
    X_test = X[split:]

    # y split
    y_train = y[:split - 1]
    y_test = y[split:]

    return X_train,X_test,y_train,y_test


def scaling(crypto='BTC',window=10):
    """ returns four arrays:
    X_train_scaled : array of lists with the 70% of the observed feature values scaled,
    X_test_scaled : array of lists with the 30% of the observed feature values scaled,
    y_train_scaled : array of lists with the 70% of the observed target values scaled,
    y_test_scaled : array of lists with the 30% of the observed target values scaled.
    """

    scaler = StandardScaler()
    X_train,X_test,y_train,y_test = folds(crypto,window)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)

    return X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled


def reshape(crypto='BTC',window=10):
    """ returns two arrays:
    X_train : array of lists with the 70% of the observed feature values scaled,
    and reshaped.
    X_test : array of lists with the 30% of the observed feature values
    scaled, and reshaped
    """
    X_train_scaled,X_test_scaled = scaling(crypto,window)[0],scaling(crypto,window)[1]
    X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    return X_train,X_test
