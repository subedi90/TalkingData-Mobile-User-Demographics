import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.cluster import DBSCAN
from sklearn import metrics as skmetrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from keras.layers.advanced_activations import PReLU
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import ensemble
from sklearn.decomposition import PCA
import os
import gc
from scipy import sparse
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

import pickle

#------------------------------------------------- Write functions ----------------------------------------

def rstr(df): return df.dtypes, df.head(3) ,df.apply(lambda x: [x.unique()]), df.apply(lambda x: [len(x.unique())]),df.shape

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        if not isinstance(X, np.ndarray):
			X_batch = X[batch_index,:].toarray()
        else:
			X_batch = X[batch_index,:]
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        if not isinstance(X, np.ndarray):
			X_batch = X[batch_index,:].toarray()
        else:
			X_batch = X[batch_index,:]
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

##################
#  Build Model
##################
			
def baseline_model(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(150, input_dim=input_dim, init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=input_dim, init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  #logloss
    return model
	
def ensemble_model(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(35, input_dim=input_dim, init='normal', activation='tanh'))
    #model.add(PReLU())
    #model.add(Dropout(0.1))
    model.add(Dense(12, init='normal', activation='softmax'))
    # Compile model
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  #logloss
    return model
	
def ensemble_model2(input_dim):
    # create model
    model = Sequential()
    #model.add(Dense(30, input_dim=input_dim, init='normal', activation='tanh'))
    #model.add(PReLU())
    #model.add(Dropout(0.1))
    model.add(Dense(12, init='normal', input_dim=input_dim, activation='softmax'))
    # Compile model
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  #logloss
    return model
			
def run_keras(X_train, y_train, X_val, y_val, epochs=50, ensemble=False, nea=False, predict=[]):
	if(ensemble):
		if(nea):
			model = ensemble_model2(X_train.shape[1])
		else:
			model = ensemble_model(X_train.shape[1])
	else:
		model=baseline_model(X_train.shape[1])
	save_model = ModelCheckpoint('../models/keras.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
	early_stopping=EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
	if not isinstance(X_val, np.ndarray):
		X_val = X_val.todense()
	batch_size = 400
	fit= model.fit_generator(generator=batch_generator(X_train, y_train, batch_size, True),
							 nb_epoch=epochs,
							 samples_per_epoch=np.ceil(X_train.shape[0]/batch_size)*batch_size,
							 callbacks=[save_model, early_stopping],
							 validation_data=(X_val, y_val), verbose=0
							 )
	model = load_model('../models/keras.h5')
	# evaluate the model
	X_valid_prob_mlp = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
	score = round(log_loss(y_val, X_valid_prob_mlp),5)
	print('logloss val {}'.format(score))
	returns = []
	for d in predict:
		returns.append(model.predict_generator(generator=batch_generatorp(d, 400, False), val_samples=d.shape[0]))
	return score, returns
