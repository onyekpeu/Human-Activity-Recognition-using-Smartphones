# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:52:52 2020

@author: onyekpeu
"""
# import pywt

import os
import numpy as np
import os.path
#import pandas as pd
#from sklearn.metrics import accuracy_score,roc_auc_score
import tensorflow.keras
#from keras.layers import Dense, Flatten
#from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, History, LearningRateScheduler
from tensorflow.keras import regularizers
#from keras.layers import Recurrent_Dropout
# from tensorflow.keras.utils.vis_utils import plot_model
#from tensorflow.keras import backend as K
from tensorflow.keras import optimizers 
import time
import matplotlib.pyplot as plt
def read_signals_ucihar(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def read_labels_ucihar(filename):        
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities

def load_ucihar_data():
    train_folder = os.path.join("train", "InertialSignals")
    test_folder = os.path.join("test", "InertialSignals")
    labelfile_train = os.path.join("train", "y_train.txt")
    labelfile_test = os.path.join("test", "y_test.txt")
    train_signals, test_signals = [], []
    for input_file in os.listdir(train_folder):
        signal = read_signals_ucihar(os.path.join(train_folder, input_file))
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    for input_file in os.listdir(test_folder):
        signal = read_signals_ucihar(os.path.join(test_folder, input_file))
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
    train_labels = read_labels_ucihar(labelfile_train)
    test_labels = read_labels_ucihar(labelfile_test)
    return train_signals, train_labels, test_signals, test_labels

def seq_data_man(data, batch_size, seq_dim, input_dim, output_dim):
    X,Y=data
    X=np.array(X)
    Y=np.array(Y)
    lx=len(Y)
    x = []
    y = []
    for i in range(seq_dim,lx):
        x.append(X[i-seq_dim:i, :])
        y.append(Y[i-1])
    x, y, = np.array(x), np.array(y)
    return (x, y)
train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = load_ucihar_data()

from scipy.fftpack import fft
def get_fft_values(y_values, N):
    fft_values_ = fft(y_values)
    fft_values = np.abs(fft_values_[0:N//2])
#    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return fft_values


def to_fourier(data,N,seq,input_dim):
    Data = np.zeros((data.shape[0], seq, input_dim))
    for ii in range(len(data)):
    #        if ii % 1000 == 0:
    ##            print(ii)
        for jj in range(0,input_dim):
            signal = data[ii, :, jj]
            coeff = get_fft_values(signal, N)
            coeff_ = coeff[:seq]
            coeff_ =np.reshape(coeff_ ,(1,len(coeff_ )))
            Data[ii, :, jj] = coeff_
    return Data

y_train=list(train_labels_ucihar)
y_test=list(test_labels_ucihar)

#batch size
#32-96.97/96.42
#64-97.1/96.97
#96-96.69/96.83
#128-96.76/96.18
#256-96.87/96.33
#320-97.04/96.08
#416-96.76/96.49
#512- 96.52/95.91
#720-96.35\96.21
#1024- 96.22/95.47
#1280- 95.54/95.71
#weights

#8-95.53
#16-93.08
#32-96.45
#64-96.73
#96-97.31
#128-97.68
#256-97.24
#512-97.21


# par=[1,2,1]
#par=[96, 128]

par=np.arange(1, 2,1)#14
opt_runs=np.zeros((len(par),1))
opt_runs1=np.zeros((len(par),1))
for opt_ in range(len(par)):
    opt=par[opt_] 
    seq=4#int(opt)#0#number of timesteps
    N=24
    input_shape=9
    num_classes = 6
    batch_size = 16#64
    num_classes = 7
    epochs = 100
    h2=100#32
    dropout=0.25
    learning_rate=0.0002502  
    nor=1
    seq_dim=12

    
    x_train1=to_fourier(train_signals_ucihar,N,seq_dim,input_shape) 
    x_test1=to_fourier(test_signals_ucihar,N,seq_dim,input_shape) 
    
    n_runs=np.zeros((int(nor),1))    
    n_runs1=np.zeros((int(nor),1)) 
    for nfr in range(nor):
        print('full training run: '+ str(nfr))
        print('optimisation run: '+ str(opt))    
     
        history = History()
     
        y_train1 = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test1 = tensorflow.keras.utils.to_categorical(y_test, num_classes)
        
        start=time.time()
        regressor = Sequential()
        regressor.add(GRU(units =h2,input_shape = (x_train1.shape[1], x_train1.shape[2]), activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, return_sequences = False))
#        regressor.add(Dropout(dropout))
#        regressor.add(GRU(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, return_sequences = True))
#        regressor.add(Dropout(dropout))
#        regressor.add(GRU(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, return_sequences = False))
#            
        regressor.add(Dense(units = num_classes, activation='sigmoid'))
        adamax=optimizers.Adam(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
        regressor.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=adamax,
                      metrics=['accuracy'])   
        print(regressor.summary())
        
        regressor.fit(x_train1, y_train1, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(x_test1, y_test1), callbacks=[history])
        plt.plot(history.history['loss'], label='train')
        
        plt.show()
        
        class AccuracyHistory(tensorflow.keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.acc = []
        
            def on_epoch_end(self, batch, logs={}):
                self.acc.append(logs.get('acc'))
         
        history = AccuracyHistory()
                
        score = regressor.evaluate(x_test1, y_test1, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        n_runs[nfr]=score[1]
    opt_runs[opt_]=np.amax(n_runs)
    print(max(n_runs))
