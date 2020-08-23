import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from yahoo_fin import stock_info as si
from collections import deque


#the feature columns are just how the data from Yahoo Fin is organized
#the lookup step just means how many days ahead how many days in the future we look ahead to predict
#the test data ratio is just the percentage (out of 1) of data that we look at

def load_data(ticker_code, feature_columns=['adjclose', 'volume', 'open', 'high', 'low'], days = 50, 
              lookup_step = 1, test_data_ratio = .2):
 
    if isinstance(ticker_code, str):                 #if the ticker_code is a string ("DPZ", "GOOGL")...
        df = si.get_data(ticker_code)               #import data from Yahoo Fin (df = company dataframe)
    elif isinstance(ticker_code, pd.DataFrame):      #if the data has been previously loaded...
        df = ticker_code                            #get data directly
    else:
        raise TypeError("Ticker must be either a string or a `pd.DataFrame` instance")

    results = {}                                      #create set to hold properly loaded data
    results['df'] = df.copy()                         #return original company dataframe 

    for col in feature_columns:
        assert col in cdf.columns, f"'{col}' does not exist in the dataframe."

    if True:
        scaled_columns = {}
        for column in feature_columns:               #scale stock prices from 0 to 1
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(cdf[column].values, axis=1))
            scaled_columns[column] = scaler

        results["scaled_columns"] = scaled_columns    # add the MinMaxScaler calculations to the result returned

    df['future'] = df['adjclose'].shift(-lookup_step)  # add the label by shifting by `lookup_step`


    last_seq = np.array(df[feature_columns].tail(lookup_step)) #retrieve last "lookup_step" columns
    df.dropna(inplace=True)                                    #remove NaNs from future columns

    seq_data = []
    seqs = deque(maxlen = days)

    for entry, target in zip(df[feature_columns].values, df['future'].values):
        seqs.append(entry)
        if len(seqs) == n_steps:
            seq_data.append([np.array(seqs), target])

    
    last_seq = list(seq) + list(last_seq)                             # append last  "days" number of steps sequnce with lookup_step
    last_seq = np.array(pd.DataFrame(last_seq).shift(-1).dropna())    # shift last sequence by -1
    results['last_seq'] = last_seq                                    # add to result
    
    # construct the x and y arrays, which respectively divide into sequence and target label
    x, y = [], []
    for seq, target in sequence_data:
        x.append(seq)
        y.append(target)
    x = np.array(x)
    y = np.array(y)
    x = x.reshape((x.shape[0], x.shape[2], x.shape[1]))   #reshapen in order to account for shape of neural network
    
    # split the dataset appropriately into training and testing data
    results["x_train"], results["x_test"], results["y_train"], results["y_test"] = train_test_split(x, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    # return the result
    return results

