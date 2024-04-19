import pandas as pd
import pandas_ta as ta
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict

from keras.models import load_model
import os
import re
from agent import *

def preprocess(stock, lookback = 30, features = 9, split_by = 'ratio', split = '0.8'):
        # Opening stock csv file
        file = 'data/'+stock+'.csv'
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace = True)
        
        # Add indicators
        df['RSI'] = ta.rsi(df.Close, length=15)
        df['EMAF'] = ta.ema(df.Close, length = 20)
        df['EMAM']=ta.ema(df.Close, length = 100)
        df['EMAS']=ta.ema(df.Close, length = 150)
        stoch = ta.stoch(df.High, df.Low, df.Close)
        df['STOS_k'] = stoch['STOCHk_14_3_3']
        df['STOS_d'] = stoch['STOCHk_14_3_3']
        df['Delta_Vol'] = df['Volume'].shift(-1) - df['Volume']
        # df['Day_of_Month'] = df['Date'].dt.strftime('%d')
        # df['Day_of_Month'] = df['Day_of_Month'].astype(np.float64)
        
        #Add heavy indicators to force outcome
        df['Tgt'] = (df['Adj Close'] - df['Open']).shift(-1)
        df['Tgt_Class'] = (df['Tgt'] > 0).astype(int)
        
        #Add output labels
        df['Tgt_Close'] = df['Adj Close'].shift(-1)

        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        
        if(split_by == 'date'):
            split_at= df[df['Date'] == split].index[0]
        else:
            split_at = int(float(split) * len(df))

        train_df = df.iloc[:split_at].copy()
        test_df = df.iloc[split_at:].copy()
        
        #Drop unnecessary columns
        train_df.drop(['Volume', 'Adj Close', 'Date'], 
                      axis=1, 
                      inplace=True)
        
        test_df.drop(['Volume', 'Adj Close', 'Date'], 
                     axis=1,
                     inplace=True)

        # Scaling the df
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train_df.values)
        test_data = scaler.fit_transform(test_df.values)

        # Prepare final data
        X_train, X_test = [], []
        y_train, y_test = [], []
        
        for j in range(features):
            X_train.append([])
            X_test.append([])
            for i in range(lookback, len(train_data)):
                X_train[j].append(train_data[i - lookback:i, j])
            for i in range(lookback, len(test_data)):
                X_test[j].append(test_data[i - lookback:i, j])
                
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_train = np.moveaxis(X_train, [0], [2])
        X_test = np.moveaxis(X_test, [0], [2])

        y_train = train_data[lookback:, -1].reshape(-1, 1)
        y_test = test_data[lookback:, -1].reshape(-1, 1)
        
        return X_train, y_train, X_test, y_test

def generate_models(stocks):
    models = []
    all_models = os.listdir(os.path.join('models'))
    for stock in stocks:
        pattern = r'model_' + stock + r'_\d+lb.keras'
        exists = False
        for model in all_models:
            if re.match(pattern, model):
                models.append(load_model('models/'+model))
                exists = True
                break
    
        if not exists:
            print("Creating model for ", stock)
            agent = TradeAgent(stock) #This warning can be ignored as this prevents circular import 
            agent.train(32, 30)
            agent.save_model()
            new_models = os.listdir('models')
            for model in new_models:
                if re.match(pattern, model):
                    models.append(load_model('models/'+model))
        else:
            print("Model found for ", stock)
        
    return models

def dict2vec(odict):
    return np.array(list(odict.values()))[:,:]

def allStates(min_len, stocks = ['AAPL', 'GOOGL', 'AMZN', 'ADBE'], split_ratio = 0.8):
    mdls = generate_models(stocks)
    for stock in stocks:
        _, __ , x_test, ___ = preprocess(stock, 30, 9)
        min_len = min(min_len, len(x_test))

    stock_data = OrderedDict()
    true_data = OrderedDict()
    differences = OrderedDict()

    for stock in stocks:
        df = pd.read_csv('data/'+stock+'.csv')['Adj Close'].values[-min_len:] #last min_len elements
        _, __, x_test, y_test = preprocess(stock, 30, 9)
        split =int(split_ratio * min_len)
        
        x_test = x_test[-min_len+1:]
        y_test = y_test[-min_len:]
        y_test = y_test[:-1] #shifted
        
        # stock_data[stock] = (x_test[: split], y_test[: split], x_test[split: ], y_test[split: ])
        # true_data[stock] = (df[:split], df[split:])
        true_data[stock] = df
        # print(len(stock_data[stock][0]), len(stock_data[stock][1]), len(stock_data[stock][2]), len(stock_data[stock][3]) )
        # print(len(true_data[stock][0]), len(true_data[stock][1]))

        model_index = stocks.index(stock)
        model = mdls[model_index]
        predictions = model.predict(x_test)
        differences[stock] = (y_test -predictions)
        
        #differences[stock] = ()
    # stock_data_train = dict2vec(stock_data.values()[])
    # stock_data_test = dict2vec(stock_data[1])
    true_data = dict2vec(true_data)
    differences = dict2vec(differences)

    return true_data, differences

def getstate(differences, i):
    return differences[:, i]