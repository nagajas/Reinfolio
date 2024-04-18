import numpy as np
import random
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import os

from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Input, Activation
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

from utilities import *

class TradeAgent:
    def __init__(self, ticker, lookback = 30, features = 9):
        self.lookback = lookback
        self.features = features
        self.input_shape  = (lookback, features)
        self.action_size = 1 #predict scaled close difference
        self.stock = ticker
        
        self.model_exists = False
        if("model_" + ticker + "_" + str(lookback) + "lb.keras" in os.listdir('models')):
            filename = "models/model_"+ticker+"_"+str(lookback)+"lb.keras"
            self.model = load_model(filename)
            self.model_exists = True
        else:
            self.model = self._model()
            print("----------------------------------------------------------------Model Created----------------------------------------------------------------")
            
        # To be used for evaluation only
        self.X_test = None
        self.Y_test = None
    
    def _model(self):
        input_layer = Input(self.input_shape)
        lstm_layer = LSTM(150)(input_layer)
        # dense_layer = Dense(64, activation = 'linear')(lstm_layer)
        output_layer = Dense(1, activation = 'linear')(lstm_layer)
        model = Model(inputs = input_layer, outputs = output_layer)
        
        model.compile(optimizer = Adam(), loss = 'mse')
        model.summary()
        
        return model
    
    def train(self,  batch_size = 128, epochs = 100):
        if(not self.model_exists):
            X_train, Y_train, self.X_test, self.Y_test = preprocess(self.stock, self.lookback, self.features)
            # print(X_train.shape, Y_train.shape, self.Y_test.shape, self.X_test.shape)
            self.model.fit(x = X_train, 
                        y = Y_train, 
                        batch_size = batch_size, 
                        epochs = epochs, 
                        shuffle = True,
                        validation_split = 0.1)
            print("----------------------------------------------------------------Model Trained----------------------------------------------------------------")
        else:
            print("Model already trained.")
    
    def evaluate(self, graph = True):
        if not self.model_exists:
            print("Model doesn't exist or is not trained.")
            return None
        if(self.X_test is None):
            _, __, self.X_test, self.Y_test = preprocess(self.stock)
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, y_pred)
        print("Test size: ", self.X_test.shape[0])
        print("Mean Squared Error: ", mse)
        
        if(graph):
            plt.figure(figsize=(16,8))
            plt.plot(self.Y_test, color = 'black', label = 'Test')
            plt.plot(y_pred, color = 'blue', label = 'pred')
            plt.title('Stock Price Trend for '+ self.stock)
            plt.xlabel('# of Test Days')
            plt.ylabel('Scaled Price Change Value')
            plt.legend()
            plt.show()
        return mse
    
    def save_model(self):
        if(not self.model_exists):
            filename = "models/model_" + self.stock + "_" + str(self.lookback) + "lb.keras"
            self.model.save(filename)
            print("Model saved successfully at", filename)
            self.model_exists = True

                


class DQNAgent:
    
    def __init__(self, num_stocks):
        self.state_size = (num_stocks, 1)
        self.action_size = 3 #buy, sell, sit
        self.replay_buffer = deque(maxlen = 50) 
        self.inventory = []
        
        self.gamma = 0.9 # discount rate
        self.epsilon = 0.8
        self.epsilon_threshold = 0.01
        self.epsilon_decay = 0.995
        
        self.model = self._model()
        
    def _model(self):
        input_layer = Input(self.state_size)
        dense1 = Dense(64, activation = 'relu')(input_layer)
        dense2 = Dense(32, activation = 'relu')(dense1)
        output = Dense(self.action_size, activation = 'softmax')(dense2)
        model = Model(inputs = input_layer, outputs = output)
        model.compile(optimizer = Adam(), loss = 'mse')
        model.summary()
        return model
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def epsilon_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            return np.array([random.randrange(self.action_size) for _ in range(self.state_size[0])])
        
        prediction = self.model.predict(state, verbose = 0)
        return np.array([np.argmax(prediction[i]) for i in range(len(prediction))])
    
    def exp_replay(self, batch_size):
        mini_batch = []
        l = len(self.replay_buffer)
        
        for i in range(max(0,l-batch_size),l):
            mini_batch.append(self.replay_buffer[i])
        
        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose = 0)))
            else:
                target_Q = reward
            
            Q_values = self.model.predict(state, verbose = 0)

            for i in range(self.action_size):
                Q_values[i][0][actions[i]] = target_Q[i]
            
            self.model.fit(state, Q_values, epochs = 1, verbose = 0)
            
        if(self.epsilon < self.epsilon_threshold):
            self.epsilon *= self.epsilon_decay
                
                