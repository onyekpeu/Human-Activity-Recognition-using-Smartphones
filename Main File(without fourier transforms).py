# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 17:52:52 2020

@author: onyekpeu
"""
# import pywt
import os
import numpy as np
import os.path
import pandas as pd

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

train_signals_ucihar, train_labels_ucihar, test_signals_ucihar, test_labels_ucihar = load_ucihar_data()


y_train=list(train_labels_ucihar)
y_test=list(test_labels_ucihar)


# par=[1,2,1]
#par=[96, 128]

par=np.arange(1, 2,1)#14
opt_runs=np.zeros((len(par),1))
for opt_ in range(len(par)):
    opt=par[opt_] 
    N=24
    input_shape=9
    num_classes = 6
    batch_size = 16#64
    epochs = 100
    hidden_dim=32
    dropout=0.25
    learning_rate=0.0002502  
    no_of_training_runs=20


    y_train1 = tensorflow.keras.utils.to_categorical(np.array(y_train)-1, num_classes)
    y_test1 = tensorflow.keras.utils.to_categorical(np.array(y_test)-1, num_classes)
    x_train1=train_signals_ucihar
    x_test1=test_signals_ucihar
        
    n_runs=np.zeros((int(no_of_training_runs),1))    
    for nfr in range(no_of_training_runs):
        print('full training run: '+ str(nfr))
        print('optimisation run: '+ str(opt))    
        import tensorflow.keras

        from tensorflow.keras.models import Sequential
        
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.callbacks import EarlyStopping, History, LearningRateScheduler
        from tensorflow.keras import regularizers

        from tensorflow.keras import backend as K
        from tensorflow.keras import optimizers 
        import time
        import matplotlib.pyplot as plt
        
        history = History()
                
        start=time.time()
        regressor = Sequential()
        regressor.add(GRU(units =hidden_dim,input_shape = (x_train1.shape[1], x_train1.shape[2]), activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, return_sequences = False))
#        regressor.add(Dropout(dropout))
#        regressor.add(GRU(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, return_sequences = True))
#        regressor.add(Dropout(dropout))
#        regressor.add(GRU(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, return_sequences = False))
#            
        #adamax=optimizers.Adam(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
        regressor.add(Dense(units = num_classes, activation='sigmoid'))
        adamax=optimizers.Adamax(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
       
        regressor.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=adamax,
                      metrics=['accuracy'])   
        print(regressor.summary())
        
        regressor.fit(x_train1, y_train1, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[history])
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







