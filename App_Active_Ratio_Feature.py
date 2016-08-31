import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.externals import joblib
from sknn.mlp import Classifier, Layer
from logistic_sgd import run_logistic, predict_logistic
import matplotlib.pyplot as plt
from mlp import run_mlp, predict
import pickle
import math

from sklearn.grid_search import GridSearchCV

def sparse_df_to_array(df):
    num_rows = df.shape[0]   

    data = []
    row = []
    col = []

    for i, col_name in enumerate(df.columns):
        if isinstance(df[col_name], pd.SparseSeries):
            column_index = df[col_name].sp_index
            if isinstance(column_index, BlockIndex):
                column_index = column_index.to_int_index()

            ix = column_index.indices
            data.append(df[col_name].sp_values)
            row.append(ix)
            col.append(len(df[col_name].sp_values) * [i])
        else:
            data.append(df[col_name].values)
            row.append(np.array(range(0, num_rows)))
            col.append(np.array(num_rows * [i]))

    data_f = np.concatenate(data)
    row_f = np.concatenate(row)
    col_f = np.concatenate(col)

    arr = coo_matrix((data_f, (row_f, col_f)), df.shape, dtype=np.float64)
    return arr.tocsr()
	
def feature_generator():
	# Create bag-of-apps in character string format
	# first by event
	# then merge to generate larger bags by device

	##################
	#   App Labels
	##################
	'''
	print("# Read App Labels")
	app_lab = pd.read_csv("../input/app_labels.csv", dtype={'device_id': np.str})
	app_lab = app_lab.groupby("app_id")["label_id"].apply(
		lambda x: " ".join(str(s) for s in np.unique(x.values)))
	'''
	##################
	#   App Events
	##################
	print("# Read App Events")
	app_ev = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str, 'app_id': np.str})
	##################
	#     Events
	##################
	print("# Read Events")
	events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
	events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
	events = pd.merge(app_ev, events[['device_id','event_id','counts']], how='left', on='event_id', left_index=True)
	active_events = events[events['is_active']==1]
	print(len(events))
	print(len(active_events))
	print(len(active_events['app_id'].unique()))
	events.drop_duplicates(subset=['device_id','event_id','app_id'], inplace=True)
	events['active_counts'] = events.groupby(['device_id','app_id'])['is_active'].transform(lambda x: x[x==1].count())
	events.drop_duplicates(subset=['device_id','app_id'], inplace=True)
	events['active_ratio'] = events['active_counts']/events['counts']
	print(events[['device_id','app_id','active_ratio','counts','active_counts']])
	events = events[['device_id','app_id','active_ratio']]	
	print(len(active_events['app_id'].unique()))
	events = events.pivot(index='device_id', columns='app_id', values='active_ratio')
	events.reset_index(inplace=True)
	events.fillna(0, inplace=True)
	print(events[events['device_id']=='-9221026417907250000'])
	
	del app_ev
	#del app_small
	#del e1
	##################
	#   Phone Brand
	##################
	print("# Read Phone Brand")
	pbd = pd.read_csv("../input/phone_brand_device_model.csv",
					  dtype={'device_id': np.str})
	pbd.drop_duplicates('device_id', keep='first', inplace=True)
	pbd['device_model'] = pbd['device_model']
	pbd['phone_brand'] = pbd['phone_brand']	
	pbd = pbd[['device_id','phone_brand','device_model']]
	pbd = pd.get_dummies(pbd, columns = ['phone_brand', 'device_model'], sparse=False)
	pbd.fillna(0, inplace=True)
	#print(pbd.iloc[0])
	print(type(pbd))
	print(len(pbd.columns))
	##################
	#  Train and Test
	##################
	print("# Generate Train and Test")
	features = pd.merge(pbd, events, on='device_id', left_index=True)
	del pbd
	del events
	
	train = pd.read_csv("../input/gender_age_train.csv",
						dtype={'device_id': np.str})
	train = features[features['device_id'].isin(train['device_id'])]

	test = pd.read_csv("../input/gender_age_test.csv",
					   dtype={'device_id': np.str})
	test = features[features['device_id'].isin(test['device_id'])]
	
	train.drop('device_id', axis=1, inplace=True)
	test.drop('device_id', axis=1, inplace=True)
	
	train = sparse_df_to_array(train)
	test = sparse_df_to_array(test)
	
	col_sum = train.sum(axis=0).values
	print(col_sum.shape)
	nrows = len(train)
	cols = np.where((col_sum > 0) & (col_sum < nrows))[0]
	print(len(train.columns))
	print(cols.shape)
	train = train[:,cols]
	test = test[:,cols]
	
	print('# Save the Featured Train/Test data..')
	with open("../cache/sparse_train_2.p", 'wb') as f:
		pickle.dump(train, f)
	with open("../cache/sparse_test_2.p", 'wb') as f:
		pickle.dump(test, f)
	return train, test
		
	#print('# Compute TF-IDF')
	#get_hash_data(train,test)
	
feature_generator()