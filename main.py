# coding=utf8
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from scipy import sparse
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MaxAbsScaler
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble  import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LSHForest
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import dump_svmlight_file
from sknn.mlp import Classifier, Layer
from logistic_sgd import run_logistic, predict_logistic
import matplotlib.pyplot as plt
from mlp import run_mlp, predict
import pickle
import math
from pyfm import pylibfm as fm
import pywFM

from sklearn.grid_search import GridSearchCV
#from xgboost.sklearn import XGBClassifier
from keras_ import run_keras

import os
os.environ["PATH"] += os.pathsep + 'C:\mingw-w64\x86_64-6.1.0-win32-seh-rt_v5-rev0\mingw64\bin'

def get_hash_data(train, test):
	df = pd.concat((train, test), axis=0, ignore_index=True)
	split_len = len(train)
	y = train['group']
	device_id = test["device_id"].values
	
	counts = df[['counts','appcounts1']].fillna(-1).as_matrix()
	#print(counts)
	
	# TF-IDF Feature
	tfv = TfidfVectorizer(min_df=1)
	df = df[["phone_brand", "device_model", "app_lab", "Cluster", "hour"]].astype(np.str).apply(
		lambda x: " ".join(s for s in x), axis=1).fillna("Missing")
	df_tfv = tfv.fit_transform(df)
	print(type(df_tfv))
	df_tfv = sparse.csr_matrix(hstack([df_tfv, counts]))
	
	train = df_tfv[:split_len, :]
	test = df_tfv[split_len:, :]
	print('# Save the Featured Train/Test data..')
	with open("../cache/sparse_train_xgb.p", 'wb') as f:
		pickle.dump(train, f)
	with open("../cache/sparse_test_xgb.p", 'wb') as f:
		pickle.dump(test, f)
	with open("../cache/y.p", 'wb') as f:
		pickle.dump(y, f)
	with open("../cache/device.p", 'wb') as f:
		pickle.dump(device_id, f)
	return train, test, y, device_id

def feature_generator():
	# Create bag-of-apps in character string format
	# first by event
	# then merge to generate larger bags by device

	##################
	#   App Labels
	##################

	print("# Read App Labels")
	app_lab = pd.read_csv("../input/app_labels.csv", dtype={'device_id': np.str})
	app_lab = app_lab.groupby("app_id")["label_id"].apply(
		lambda x: " ".join(str(s) for s in x))

	##################
	#   App Events
	##################
	print("# Read App Events")
	app_ev = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})
	
	app_ev['appcounts'] = app_ev.groupby(['event_id'])['app_id'].transform('count')
	app_small = app_ev.groupby(['event_id']).agg({'appcounts':'mean'}).reset_index(level=[0])
	
	app_ev["app_lab"] = app_ev["app_id"].map(app_lab)
	app_ev = app_ev.groupby("event_id")["app_lab"].apply(
		lambda x: " ".join(str(s) for s in x))

	del app_lab

	##################
	#     Events
	##################
	print("# Read Events")
	events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
	
	events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
	events_small = events.groupby(['device_id']).agg({'counts':'mean'}).reset_index(level=[0])
	e1=pd.merge(events, app_small, how='left', on='event_id', left_index=True)
	e1.loc[e1.isnull()['appcounts'] ==True, 'appcounts']=0
	e1['appcounts1'] = e1.groupby(['device_id'])['appcounts'].transform('sum')
	e1_small = e1[['device_id', 'appcounts1']].drop_duplicates('device_id', keep='first')
	
	events["app_lab"] = events["event_id"].map(app_ev)
	events_loc = events[['device_id', 'longitude','latitude']]
	events = events.groupby("device_id")["app_lab"].apply(
		lambda x: " ".join(str(s) for s in x))

	del app_ev
	del app_small
	del e1
	##################
	#   Phone Brand
	##################
	print("# Read Phone Brand")
	pbd = pd.read_csv("../input/phone_brand_device_model.csv",
					  dtype={'device_id': np.str})
	pbd.drop_duplicates('device_id', keep='first', inplace=True)
	pbd['device_model'] = 'M_'+pbd['device_model']
	
	#GeoLocation
	print('Read Feature Geolocation...')
	gl = pd.read_csv('../features/geolocation.csv', dtype={'Cluster':np.str})
	e2 = pd.merge(events_loc, gl, how='left', on=['longitude','latitude'], left_index=True)
	#e2['loc'] = e2.groupby(['device_id'])['Cluster'].apply(lambda x: stats.mode(x)[0][0])
	loc = e2[['device_id','Cluster']].drop_duplicates('device_id',keep='first')
	loc['Cluster'] = 'C_' + loc['Cluster']
	del events_loc
	
	#Hour
	print('Read Feature Hour...')
	hour = pd.read_csv('../features/hour.csv', dtype={'device_id': np.str})
	
	##################
	#  Train and Test
	##################
	print("# Generate Train and Test")
	train = pd.read_csv("../input/gender_age_train.csv",
						dtype={'device_id': np.str})
	train["app_lab"] = train["device_id"].map(events)
	train = pd.merge(train, pbd, how='left',
					 on='device_id', left_index=True)
	train = pd.merge(train, loc, how='left', on='device_id', left_index=True)
	train = pd.merge(train, hour, how='left', on='device_id', left_index=True)
	train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
	train = pd.merge(train, e1_small, how='left', on='device_id', left_index=True)

	test = pd.read_csv("../input/gender_age_test.csv",
					   dtype={'device_id': np.str})
	test["app_lab"] = test["device_id"].map(events)
	test = pd.merge(test, pbd, how='left',
					on='device_id', left_index=True)
	test = pd.merge(test, loc, how='left', on='device_id', left_index=True)
	test = pd.merge(test, hour, how='left', on='device_id', left_index=True)
	test = pd.merge(test, events_small, how='left', on='device_id', left_index=True)
	test = pd.merge(test, e1_small, how='left', on='device_id', left_index=True)
	
	del pbd
	del events
		
	print('# Compute TF-IDF')
	get_hash_data(train,test)
	
def modelfit(alg, dtrain, y,useTrainCV=True, cv_folds=2, early_stopping_rounds=50):

	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(dtrain, label=y)
		xgb_param['num_class'] = len(np.unique(y))
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics='mlogloss', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
		alg.set_params(n_estimators=cvresult.shape[0])

	#Fit the algorithm on the data
	alg.fit(dtrain, y,eval_metric='mlogloss')
		
	#Predict training set:
	dtrain_predictions = alg.predict(dtrain)
	dtrain_predprob = alg.predict_proba(dtrain)
		
	#Print model report:
	print "\nModel Report"
	print "Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions)
	print "LogLoss Score (Train): %f" % metrics.log_loss(y, dtrain_predprob)
					
	feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')
	plt.show()
	
def xgboost_tuning():
	train,test,Y,device_id = None,None,None,None
	print('Load the featured Train/Test data..')
	with open('../cache/sparse_train_2.p', 'rb') as f:
		train = pickle.load(f)
	with open('../cache/sparse_test_2.p', 'rb') as f:
		test = pickle.load(f)
	with open('../cache/y.p', 'rb') as f:
		Y = pickle.load(f)
	with open('../cache/device.p', 'rb') as f:
		device_id = pickle.load(f)
		
	# Group Labels
	lable_group = LabelEncoder()
	Y = lable_group.fit_transform(Y)
	'''
	xgb2 = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=4,
	 min_child_weight=11, gamma=0, subsample=0.8, colsample_bytree=0.8,
	 objective= 'multi:softprob', nthread=3, scale_pos_weight=1, seed=27)
	modelfit(xgb2, train, Y)
	exit()
	'''
	##################
	#     XGBoost Tune
	##################
	param_test1 = {
		'max_depth':[3,4,5],
		'min_child_weight':[9,10,11]
	}
	gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=230, max_depth=4,
	 min_child_weight=11, gamma=0, subsample=0.8, colsample_bytree=0.3,
	 objective= 'multi:softprob', nthread=4, scale_pos_weight=1, seed=27), 
	 param_grid = param_test1, scoring='log_loss',n_jobs=2,iid=False, cv=2)
	gsearch1.fit(train, Y)
	print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
	
def xgboost(params):
	train,test,Y,device_id = None,None,None,None
	print('Load the featured Train/Test data..')
	with open('../cache/sparse_train_1_event_subset.p', 'rb') as f:
		train = pickle.load(f)
	with open('../cache/sparse_test_1.p', 'rb') as f:
		test = pickle.load(f)
	with open('../cache/y_event_subset.p', 'rb') as f:
		Y = pickle.load(f)
	with open('../cache/device.p', 'rb') as f:
		device_id = pickle.load(f)
	
	# Group Labels
	lable_group = LabelEncoder()
	Y = lable_group.fit_transform(Y)
		
	X_train, X_val, y_train, y_val = train_test_split(train, Y, test_size=.20, stratify=Y)

	##################
	#     XGBoost
	##################
	print('Build XGBoost Tree Model')
	dtrain = xgb.DMatrix(X_train, y_train)
	dvalid = xgb.DMatrix(X_val, y_val)
	'''
	params = {
		"objective": "multi:softprob",
		"num_class": 12,
		"booster": 'gbtree',
		"eval_metric": "mlogloss",
		"eta": 0.0055,
		"silent": 1,
		'max_depth': 4,
		'min_child_weight': 11, 	
		'subsample': 0.8,
		'colsample_bytree': 0.3,
		'reg_alpha': 10,
		'gamma': 0.0
	}
	'''
	num_boost_round = 200
	early_stopping_rounds = 10
		
	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	print('# XGBoost Tree - Train')
				
	gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True, learning_rates = lr)
	#gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
	gbm = xgb.train(params, xgb.DMatrix(train, Y), gbm.best_iteration, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True, learning_rates = lr)
	print('# XGBoost Tree - Validate')
	check = gbm.predict(xgb.DMatrix(X_val))
	score = log_loss(y_val.tolist(), check)
	print('# XGBoost Tree - Predict')
	y_pre = gbm.predict(xgb.DMatrix(test))

	# Write results
	print('# XGBoost Tree - Submit')
	result = pd.DataFrame(y_pre, columns=lable_group.classes_)
	result["device_id"] = device_id
	result = result.set_index("device_id")
	sub_file = 'submission_' + str(score) + '_' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')) + '.csv'
	result.to_csv(sub_file, index=True, index_label='device_id')
	
	print('XGBoost Tree - Save Model')
	gbm.save_model('../model/xgbtree.model')
	with open('../model/xgbtree_best_iteration.txt', 'wb') as f:
		f.write(str(gbm.best_iteration))
		
	#print('XGBoost Tree - Save Predictions..')
	#X_train_prob = gbm.predict(xgb.DMatrix(X_train), ntree_limit=gbm.best_iteration)
	#X_valid_prob = gbm.predict(xgb.DMatrix(X_val), ntree_limit=gbm.best_iteration)
	#np.save('../predictions/xgbt_pred_train', np.vstack((X_train_prob, X_valid_prob)))
	#np.save('../predictions/xgbt_pred_test', y_pre)
	#np.save('../predictions/y', np.hstack((y_train, y_val)))
	
def extratrees():
	train,test,Y,device_id = None,None,None,None
	print('Load the featured Train/Test data..')
	with open('../cache/sparse_train_xgb.p', 'rb') as f:
		train = pickle.load(f)
	with open('../cache/sparse_test_xgb.p', 'rb') as f:
		test = pickle.load(f)
	with open('../cache/y.p', 'rb') as f:
		Y = pickle.load(f)
	with open('../cache/device.p', 'rb') as f:
		device_id = pickle.load(f)
		
	# Group Labels
	lable_group = LabelEncoder()
	Y = lable_group.fit_transform(Y)
	
	X_train, X_val, y_train, y_val = train_test_split(train, Y, test_size=.30)

	##################
	#     ExtraTrees
	##################
	model = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0, criterion='entropy', n_jobs=32, verbose=20)
	model.fit(X_train, y_train)
	x_val_prob = model.predict_proba(X_val)
	score = log_loss(y_val.tolist(), x_val_prob)
	print("ExtraTrees - Score : " + str(score))
	
def save_pca_data():
	train,test,Y,device_id = None,None,None,None
	print('Load the featured Train/Test data..')
	with open('../cache/sparse_train_1_event_subset.p', 'rb') as f:
		train = pickle.load(f)
	with open('../cache/sparse_test_1.p', 'rb') as f:
		test = pickle.load(f)
	with open('../cache/y_event_subset.p', 'rb') as f:
		Y = pickle.load(f)
	with open('../cache/device.p', 'rb') as f:
		device_id = pickle.load(f)
		
	svd = TruncatedSVD(n_components = 200)
	train_len = train.shape[0]
	data = svd.fit_transform(vstack([train, test]))
	#print(pca.n_components_)
	print(svd.explained_variance_ratio_)
	print(svd.explained_variance_ratio_.sum())
	train = data[:train_len]
	test = data[train_len:]
	
	print('# Save the Featured Train/Test data..')
	with open("../cache/pca_train_1_event_subset.p", 'wb') as f:
		pickle.dump(train, f)
	with open("../cache/pca_test_1_event_subset.p", 'wb') as f:
		pickle.dump(test, f)
		
def kNN():
	train,test,Y,device_id = None,None,None,None
	print('Load the featured Train/Test data..')
	with open('../cache/sparse_train_xgb_1.p', 'rb') as f:
		train = pickle.load(f)
	with open('../cache/sparse_test_xgb_1.p', 'rb') as f:
		test = pickle.load(f)
	with open('../cache/y.p', 'rb') as f:
		Y = pickle.load(f)
	with open('../cache/device.p', 'rb') as f:
		device_id = pickle.load(f)

	# Group Labels
	lable_group = LabelEncoder()
	Y = lable_group.fit_transform(Y)
	
	svd = TruncatedSVD(n_components = 500)
	train_len = train.shape[0]
	data = svd.fit_transform(vstack([train, test]))
	#print(pca.n_components_)
	print(svd.explained_variance_ratio_)
	print(svd.explained_variance_ratio_.sum())
	train = data[:train_len]
	test = data[train_len:]
	
	#print(train)
	'''
	for i in np.unique(Y):
		plt.scatter(train[:,0], train[:,1], c=Y.tolist())
	plt.show()
	'''
	X_train, X_val, y_train, y_val = train_test_split(train, Y, test_size=.20)
	print '# Build kNN Model'
	##################
	#  kNN
	##################
	n_neighbors = 20
	model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', metric='minkowski', p=2, n_jobs=2)
	model.fit(X_train, y_train)
	x_val_prob = model.predict_proba(X_val)
	X_train_prob = model.predict_proba(X_train)
	valid_score = log_loss(y_val.tolist(), x_val_prob)
	train_score = log_loss(y_train.tolist(), X_train_prob)
	print("kNN-%s - Train Score: %s; Valid Score: %s" % (n_neighbors, str(train_score), str(valid_score)))
	
def randomforest():
	train,test,Y,device_id = None,None,None,None
	print('Load the featured Train/Test data..')
	with open('../cache/sparse_train_1.p', 'rb') as f:
		train = pickle.load(f)
	with open('../cache/sparse_test_1.p', 'rb') as f:
		test = pickle.load(f)
	with open('../cache/y.p', 'rb') as f:
		Y = pickle.load(f)
	with open('../cache/device.p', 'rb') as f:
		device_id = pickle.load(f)
		
	# Group Labels
	lable_group = LabelEncoder()
	Y = lable_group.fit_transform(Y)
		
	X_train, X_val, y_train, y_val = train_test_split(train, Y, test_size=.20, stratify=Y)
	n_estimators = 200
	model = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=500, min_samples_leaf=2, n_jobs=3, verbose=0)
	model = model.fit(X_train, y_train)
	x_val_prob = model.predict_proba(X_val)
	X_train_prob = model.predict_proba(X_train)
	valid_score = log_loss(y_val.tolist(), x_val_prob)
	train_score = log_loss(y_train.tolist(), X_train_prob)
	print("RF-%s - Train Score: %s; Valid Score: %s" % (n_estimators, str(train_score), str(valid_score)))

def mlp():
	train,test,Y,device_id = None,None,None,None
	print('Load the featured Train/Test data..')
	with open('../cache/sparse_train_2.p', 'rb') as f:
		train = pickle.load(f)
	with open('../cache/sparse_test_2.p', 'rb') as f:
		test = pickle.load(f)
	with open('../cache/y.p', 'rb') as f:
		Y = pickle.load(f)
	with open('../cache/device.p', 'rb') as f:
		device_id = pickle.load(f)
	with open('../cache/kfold_5.p', 'rb') as f:
		kf = pickle.load(f)
	'''
	for train_index, test_index in kf:
		#X_train, X_val, y_train, y_val = train_test_split(train, Y, test_size=.30)
		mlp = train_mlp(train[train_index], y[train_index], train[test_index], y[test_index])
		X_valid_prob_mlp = mlp.predict_proba(sparse.csr_matrix(train[test_index]))
	
	return
	
	#Standardize data...
	train_len = train.shape[0]
	data = vstack([sparse.csr_matrix(train), sparse.csr_matrix(test)])
	scaler = StandardScaler(with_mean=False)
	data = scaler.fit_transform(data)
	scaler = MaxAbsScaler()
	data = scaler.fit_transform(data)
	train = data[:train_len]
	test = data[train_len:]
	'''
	# Group Labels
	lable_group = LabelEncoder()
	Y = lable_group.fit_transform(Y)
	X_train, X_val, y_train, y_val = train_test_split(train, Y, test_size=.30)
	
	print('Build MLP Model')
	'''mlp = Classifier(
			layers=[
				Layer("Tanh", units=200, dropout=0.35),
				Layer("Softmax")],
			learning_rate=0.01, n_iter=1000, batch_size=50, n_stable=10, f_stable=0.001,
			valid_set=(X_val,y_val), regularize='dropout', loss_type='mcc', verbose=True)
	mlp.fit(X_train, y_train)
	
	X_train_prob = mlp.predict_proba(X_train)
	X_valid_prob = mlp.predict_proba(X_val)
	test_prob = mlp.predict_proba(sparse.csr_matrix(test))'''

	test_prob, X_train_prob, X_valid_prob, best_iter = run_mlp(X_train.toarray(), X_val.toarray(), y_train, y_val, test.toarray(), learning_rate=0.0015, n_epochs=1000, n_hidden=200, activation='tanh')
	
	score = log_loss(y_val, X_valid_prob)
	print("MLP - Training set score: %f" % log_loss(y_train, X_train_prob))
	print("MLP - Test set score: %f" % score)	

	# Write results
	print('# MLP Tree - Submit')
	result = pd.DataFrame(test_prob, columns=lable_group.classes_)
	result["device_id"] = device_id
	result = result.set_index("device_id")
	sub_file = 'submission_mlp_' + str(score) + '_' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')) + '.csv'
	result.to_csv(sub_file, index=True, index_label='device_id')
	
	joblib.dump(mlp, '../model/mlp.model')
	
	#X_test_prob = mlp.predict_proba(test)
	#np.save('../predictions/mlp_pred_train', np.vstack((X_train_prob, X_valid_prob)))
	#np.save('../predictions/mlp_pred_test', X_test_prob)
	
def train_kNN(X_train, y_train, n_neighbors, weights, metric, algorithm='auto'):
	model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, weights=weights, metric=metric, n_jobs=2)
	model.fit(X_train, y_train)
	return model
	
def train_mlp(X_train, y_train, X_valid = None, y_valid=None, iterations=150):
	mlp = None
	if(X_valid is None):
		print('Run without Validation..')
		mlp = Classifier(
		layers=[
			Layer("Rectifier", units=100, dropout=0.1),
			Layer("Softmax")],
		learning_rate=0.05, n_iter=iterations, batch_size=500, n_stable=5,
		f_stable=0.0003, regularize='dropout', loss_type='mcc', verbose=True)
	else:
		mlp = Classifier(
			layers=[
				Layer("Rectifier", units=100, dropout=0.1),
				Layer("Softmax")],
			learning_rate=0.05, n_iter=iterations, batch_size=500, n_stable=5, f_stable=0.0003,
			valid_set=(X_valid, y_valid), regularize='dropout', loss_type='mcc', verbose=True)
	mlp.fit(X_train, y_train)
	return mlp
	
def ensemble_mlp(X_train, y_train, X_valid = None, y_valid=None, iterations=500):
	mlp = Classifier(
			layers=[
				Layer("Sigmoid", units=35),
				Layer("Softmax")],
			learning_rate=0.03, n_iter=iterations, batch_size=500, n_stable=5, f_stable=0.0003,
			valid_set=(X_valid, y_valid), loss_type='mcc', verbose=True)
	mlp.fit(X_train, y_train)
	return mlp
	
def train_xgb(X_train, y_train, X_valid = None, y_valid = None, iterations=1000):
	dtrain = xgb.DMatrix(X_train, y_train)
	if(X_valid is not None):
		dvalid = xgb.DMatrix(X_valid, y_valid)
		watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
		gbm = xgb.train(xgb_params, dtrain, iterations, evals=watchlist, early_stopping_rounds=1, verbose_eval=True, learning_rates = lr)	
	else:
		gbm = xgb.train(xgb_params, dtrain, iterations, learning_rates = lr)	
	return gbm
	
def train_rf(X_train, y_train, n_estimators=300):
	model = RandomForestClassifier(n_estimators=n_estimators, max_depth=500, min_samples_leaf=2, n_jobs=3, verbose=0)
	model = model.fit(X_train, y_train)
	return model
	
def train_extra_trees(X_train, y_train, n_estimators=300):
	model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=2, random_state=0, criterion='entropy', n_jobs=3, verbose=0)
	model.fit(X_train, y_train)
	return model
	
def first_level_probs(test_only=False):		
	train,Y = None,None
	print('Load the featured Train data..')
	with open('../cache/sparse_train2_1_event_availability.p', 'rb') as f:
		ea = pickle.load(f)
	with open('../cache/sparse_train_1.p', 'rb') as f:
		train1 = pickle.load(f)
	with open('../cache/sparse_test_1.p', 'rb') as f:
		test1 = pickle.load(f)
	with open('../cache/sparse_train_2.p', 'rb') as f:
		train2 = pickle.load(f)
	with open('../cache/sparse_test_2.p', 'rb') as f:
		test2 = pickle.load(f)
	with open('../cache/sparse_train_3.p', 'rb') as f:
		train3 = pickle.load(f)
	with open('../cache/sparse_test_3.p', 'rb') as f:
		test3 = pickle.load(f)
	with open('../cache/sparse_train_4.p', 'rb') as f:
		train4 = pickle.load(f)
	with open('../cache/sparse_test_4.p', 'rb') as f:
		test4 = pickle.load(f)
	with open('../cache/sparse_train_5.p', 'rb') as f:
		train5 = pickle.load(f)
	with open('../cache/sparse_test_5.p', 'rb') as f:
		test5 = pickle.load(f)
	#with open('../cache/sparse_train_6.p', 'rb') as f:
	#	train6 = pickle.load(f)
	with open('../cache/sparse_test_6.p', 'rb') as f:
		test6 = pickle.load(f)
	with open('../cache/sparse_train_7.p', 'rb') as f:
		train7 = pickle.load(f)
	with open('../cache/sparse_test_7.p', 'rb') as f:
		test7 = pickle.load(f)
	with open('../cache/sparse_train_8.p', 'rb') as f:
		train8 = pickle.load(f)
	with open('../cache/sparse_test_8.p', 'rb') as f:
		test8 = pickle.load(f)
	with open('../cache/sparse_train_9.p', 'rb') as f:
		train9 = pickle.load(f)
	with open('../cache/sparse_test_9.p', 'rb') as f:
		test9 = pickle.load(f)
	with open('../cache/y.p', 'rb') as f:
		Y = pickle.load(f)
	'''with open("../cache/pca_train_2.p", 'rb') as f:
		train_pca = pickle.load(f)
	with open("../cache/pca_test_2.p", 'rb') as f:
		test_pca = pickle.load(f)'''
	with open('../cache/kfold_5.p', 'rb') as f:
		kf = pickle.load(f)
	with open('../cache/sparse_train2_1_event_availability.p', 'rb') as f:
		train_ea = pickle.load(f)
	with open('../cache/sparse_test2_1_event_availability.p', 'rb') as f:
		test_ea = pickle.load(f)
	train_ea = train_ea.values.reshape((train_ea.values.shape[0], 1))
	test_ea = test_ea.values.reshape((test_ea.values.shape[0], 1))
		
	lable_group = LabelEncoder()
	y = lable_group.fit_transform(Y)
	print(train1.shape, train2.shape, train3.shape, train4.shape)
	
	train = sparse.csr_matrix(hstack((train1, train2, train3, train4)))
	test = sparse.csr_matrix(hstack((test1, test2, test3, test4)))
	#train = sparse.csr_matrix(hstack((train9, train2, train4)))
	#test = sparse.csr_matrix(hstack((test9, test2, test4)))
	
	model = SelectKBest(chi2, k=5000)
	train = model.fit_transform(train, y)
	test = model.transform(test)
	
	model = SelectKBest(chi2, k=50)
	train9 = model.fit_transform(train9, y)
	test9 = model.transform(test9)
	
	#train5[train5.nonzero()] = 1
	train = sparse.csc_matrix(hstack((train, train5, train8, train9, train_ea)))
	test = sparse.csc_matrix(hstack((test, test5, test8, test9, test_ea)))
	
	print(train.shape)
	print(test.shape)
	
	'''
	dist_test = None
	dist_train = None
	for i in range(y.max()):
		print('Getting Distance for Label %i'%i)
		train_ = train[np.where(y == i)[0]]
		lshf = LSHForest(n_estimators=20, n_candidates=100, random_state=42)
		lshf.fit(train_, y)
		print('Train Distance...')
		distances, indices = lshf.kneighbors(train, n_neighbors=2)
		distances_train = np.mean(distances, axis=1)
		print('Test Distance...')
		distances, indices = lshf.kneighbors(test, n_neighbors=2)
		distances_test = np.mean(distances, axis=1)
		if dist is None:
			dist_test = distances_test
			dist_train = distances_train
		else:
			dist_test = np.hstack(dist_test, distances_test)
			dist_train = np.hstack(dist_train, distances_train)
	print('# Save the Featured Train/Test data..')
	with open("../cache/train_dist2.p", 'wb') as f:
		pickle.dump(train, f)
	with open("../cache/test_dist2.p", 'wb') as f:
		pickle.dump(test, f)
	'''
	'''
	svd = TruncatedSVD(n_components = 500)
	train_len = train.shape[0]
	data = svd.fit_transform(vstack([train, test]))
	#print(pca.n_components_)
	print(svd.explained_variance_ratio_)
	print(svd.explained_variance_ratio_.sum())
	train = data[:train_len]
	test = data[train_len:]
	print('# Save the Featured Train/Test data..')
	with open("../cache/pca_train_2.p", 'wb') as f:
		pickle.dump(train, f)
	with open("../cache/pca_test_2.p", 'wb') as f:
		pickle.dump(test, f)
	'''
	#train = train7
	#test = test7
	
	ea_rows = np.where(ea == 1)[0]
		
	print('# Run 5-Fold Models')	
	n_neighbors = 1000
	best_xgb_ntree = 800
	best_mlp_iter = 1
	dummy_y = [0 for i in range(test.shape[0])]
	train2 = None
	y2 = []
	if not test_only:
		score = 0.0
		test_len = test.shape[0]
		for train_index, test_index in kf:
			#train_index = np.intersect1d(train_index, ea_rows)
			#test_index_ea = np.intersect1d(test_index, ea_rows)
			'''
			fm = pywFM.FM(task='classification',num_iter=1,learning_method='sgda',rlog=False)
			test = vstack((test, train[test_index]))
			y = np.hstack((dummy_y, y[test_index]))
			print(test.shape,y.shape)
			model = fm.run(train[train_index], np.array(y[train_index]), test, y, train[test_index], np.array(y[test_index]))
			print(model.predictions[:10], len(model.predictions))
			#X_test_prob_knn
			'''
			'''
			#KNN...
			#model = train_kNN(train_pca[train_index], y[train_index], 50, 'uniform', 'euclidean')
			X_valid_prob_knn = model.predict_proba(train_pca[test_index])
			#X_train_prob_knn = model.predict_proba(train_pca[train_index])
			score_ = log_loss(y[test_index].tolist(), X_valid_prob_knn)
			#train_score = log_loss(y[train_index].tolist(), X_train_prob_knn)
			print("kNN-%s - Valid Score: %s" % (n_neighbors, str(score_)))
			X_test_prob_knn = model.predict_proba(test_pca)
			
			#RF..
			model = train_rf(train[train_index], y[train_index])
			X_valid_prob_rf = model.predict_proba(train[test_index])
			X_train_prob = model.predict_proba(train[train_index])
			score_ = log_loss(y[test_index_ea].tolist(), model.predict_proba(train[test_index_ea]))
			train_score = log_loss(y[train_index].tolist(), X_train_prob)
			print("RF-%s - Train Score: %s; Valid Score: %s" % (len(model.estimators_), str(train_score), str(score_)))
			X_test_prob_rf = model.predict_proba(test)		
			'''
			'''
			#EXTRA TRESS..
			model = train_extra_trees(train[train_index], y[train_index])
			X_valid_prob_rf = model.predict_proba(train[test_index])
			X_train_prob = model.predict_proba(train[train_index])
			score_ = log_loss(y[test_index].tolist(), model.predict_proba(train[test_index]))
			train_score = log_loss(y[train_index].tolist(), X_train_prob)
			print("XT - Train Score: %s; Valid Score: %s" % (str(train_score), str(score_)))
			X_test_prob_rf = model.predict_proba(test)	
			'''
			
			score_, predictions = run_keras(train[train_index], y[train_index], train[test_index], y[test_index], predict=[train[test_index], test])
			X_valid_prob_mlp, X_test_prob_mlp = predictions			
			'''
			#XGBOOST...
			gbm = train_xgb(train[train_index], y[train_index], train[test_index], y[test_index])
			print('Predict Valid..')
			X_valid_prob_xgb = gbm.predict(xgb.DMatrix(train[test_index]))
			print('Predict Test...')
			X_test_prob_xgb = gbm.predict(xgb.DMatrix(test))
			score_ = log_loss(y[test_index].tolist(), gbm.predict(xgb.DMatrix(train[test_index])))
			'''
			score = score + score_
			#Combine Probs
			#train2s = np.hstack((X_valid_prob_xgb, X_valid_prob_mlp, X_valid_prob_knn))
			#train2s = np.hstack((X_valid_prob_xgb, X_valid_prob_mlp))
			train2s = X_valid_prob_mlp
			if train2 is not None:
				train2 = np.vstack((train2, train2s))
				X_test_prob_mlp_sum = np.add(X_test_prob_mlp_sum, X_test_prob_mlp)
			else:
				train2 = train2s
				X_test_prob_mlp_sum = X_test_prob_mlp
			y2 = y2 + y[test_index].tolist()
		
		print('KERAS: {}'.format(round(score/5.0, 5)))
		print('# Save the 1st Level Train Probabilities..')
		with open("../cache/sparse_train2_37.p", 'wb') as f:
			pickle.dump(train2, f)
		#with open("../cache/y2_1.p", 'wb') as f:
		#	pickle.dump(y2, f)

	print('# Predict Test Set..')	
	
	test_prob_mlp = np.multiply(X_test_prob_mlp_sum, 0.2)
	print('# Save the 1st Level Test Probabilities..')
	with open("../cache/sparse_test2_37.p", 'wb') as f:
		#pickle.dump(np.hstack((test_prob_xgb, test_prob_mlp, test_prob_knn)), f)
		#pickle.dump(np.hstack((test_prob_xgb, test_prob_mlp)), f)
		pickle.dump(test_prob_mlp,f)	
	
def ensemble():
	train,test,Y,device_id = None,None,None,None
	print('Load the featured Train/Test data..')
	with open('../cache/sparse_test_5.p', 'rb') as f:
		hour_test = pickle.load(f)
	with open('../cache/sparse_train2_1.p', 'rb') as f:
		train1 = pickle.load(f)
	with open('../cache/sparse_test2_1.p', 'rb') as f:
		test1 = pickle.load(f)
	with open('../cache/sparse_train2_2.p', 'rb') as f:
		train2 = pickle.load(f)
	with open('../cache/sparse_test2_2.p', 'rb') as f:
		test2 = pickle.load(f)
	with open('../cache/sparse_train2_19.p', 'rb') as f:
		train19 = pickle.load(f)
	with open('../cache/sparse_test2_19.p', 'rb') as f:
		test19 = pickle.load(f)
	with open('../cache/sparse_train2_20.p', 'rb') as f:
		train20 = pickle.load(f)
	with open('../cache/sparse_test2_20.p', 'rb') as f:
		test20 = pickle.load(f)
	with open('../cache/sparse_train2_3.p', 'rb') as f:
		train3 = pickle.load(f)
	with open('../cache/sparse_test2_3.p', 'rb') as f:
		test3 = pickle.load(f)
	with open('../cache/sparse_train2_4.p', 'rb') as f:
		train4 = pickle.load(f)
	with open('../cache/sparse_test2_4.p', 'rb') as f:
		test4 = pickle.load(f)
	with open('../cache/sparse_train2_21.p', 'rb') as f:
		train21 = pickle.load(f)
	with open('../cache/sparse_test2_21.p', 'rb') as f:
		test21 = pickle.load(f)
	with open('../cache/sparse_train2_22.p', 'rb') as f:
		train22 = pickle.load(f)
	with open('../cache/sparse_test2_22.p', 'rb') as f:
		test22 = pickle.load(f)
	with open('../cache/sparse_train2_14.p', 'rb') as f:
		train14 = pickle.load(f)
	with open('../cache/sparse_test2_14.p', 'rb') as f:
		test14 = pickle.load(f)
	with open('../cache/sparse_train2_15.p', 'rb') as f:
		train15 = pickle.load(f)
	with open('../cache/sparse_test2_15.p', 'rb') as f:
		test15 = pickle.load(f)
	with open('../cache/sparse_train2_17.p', 'rb') as f:
		train17 = pickle.load(f)
	with open('../cache/sparse_test2_17.p', 'rb') as f:
		test17 = pickle.load(f)
	with open('../cache/sparse_train2_18.p', 'rb') as f:
		train18 = pickle.load(f)
	with open('../cache/sparse_test2_18.p', 'rb') as f:
		test18 = pickle.load(f)
	with open('../cache/sparse_train2_23.p', 'rb') as f:
		train23 = pickle.load(f)
	with open('../cache/sparse_test2_23.p', 'rb') as f:
		test23 = pickle.load(f)
	with open('../cache/sparse_train2_24.p', 'rb') as f:
		train24 = pickle.load(f)
	with open('../cache/sparse_test2_24.p', 'rb') as f:
		test24 = pickle.load(f)
	with open('../cache/sparse_train2_25.p', 'rb') as f:
		train25 = pickle.load(f)
	with open('../cache/sparse_test2_25.p', 'rb') as f:
		test25 = pickle.load(f)
	with open('../cache/sparse_train2_26.p', 'rb') as f:
		train26 = pickle.load(f)
	with open('../cache/sparse_test2_26.p', 'rb') as f:
		test26 = pickle.load(f)
	with open('../cache/sparse_train2_27.p', 'rb') as f:
		train27 = pickle.load(f)
	with open('../cache/sparse_test2_27.p', 'rb') as f:
		test27 = pickle.load(f)
	with open('../cache/sparse_train2_28.p', 'rb') as f:
		train28 = pickle.load(f)
	with open('../cache/sparse_test2_28.p', 'rb') as f:
		test28 = pickle.load(f)
	with open('../cache/sparse_train2_29.p', 'rb') as f:
		train29 = pickle.load(f)
	with open('../cache/sparse_test2_29.p', 'rb') as f:
		test29 = pickle.load(f)
	with open('../cache/sparse_train2_30.p', 'rb') as f:
		train30 = pickle.load(f)
	with open('../cache/sparse_test2_30.p', 'rb') as f:
		test30 = pickle.load(f)
	with open('../cache/sparse_train2_31.p', 'rb') as f:
		train31 = pickle.load(f)
	with open('../cache/sparse_test2_31.p', 'rb') as f:
		test31 = pickle.load(f)
	with open('../cache/sparse_train2_32.p', 'rb') as f:
		train32 = pickle.load(f)
	with open('../cache/sparse_test2_32.p', 'rb') as f:
		test32 = pickle.load(f)
	with open('../cache/sparse_train2_33.p', 'rb') as f:
		train33 = pickle.load(f)
	with open('../cache/sparse_test2_33.p', 'rb') as f:
		test33 = pickle.load(f)
	with open('../cache/sparse_train2_34.p', 'rb') as f:
		train34 = pickle.load(f)
	with open('../cache/sparse_test2_34.p', 'rb') as f:
		test34 = pickle.load(f)
	with open('../cache/sparse_train2_35.p', 'rb') as f:
		train35 = pickle.load(f)
	with open('../cache/sparse_test2_35.p', 'rb') as f:
		test35 = pickle.load(f)
	with open('../cache/sparse_train2_36.p', 'rb') as f:
		train36 = pickle.load(f)
	with open('../cache/sparse_test2_36.p', 'rb') as f:
		test36 = pickle.load(f)
	with open('../cache/sparse_train2_37.p', 'rb') as f:
		train37 = pickle.load(f)
	with open('../cache/sparse_test2_37.p', 'rb') as f:
		test37 = pickle.load(f)
	with open('../cache/sparse_train2_38.p', 'rb') as f:
		train38 = pickle.load(f)
	with open('../cache/sparse_test2_38.p', 'rb') as f:
		test38 = pickle.load(f)
	with open('../cache/y2_1.p', 'rb') as f:
		y = np.array(pickle.load(f))
	with open('../cache/device.p', 'rb') as f:
		device_id = pickle.load(f)		
	with open('../cache/sparse_train2_1_xlgt.p', 'rb') as f:
		train_xlgt = pickle.load(f)
	with open('../cache/sparse_test2_1_xlgt.p', 'rb') as f:
		test_xlgt = pickle.load(f)
	with open('../cache/sparse_train2_1_event_availability.p', 'rb') as f:
		train_ea = pickle.load(f)
	with open('../cache/sparse_test2_1_event_availability.p', 'rb') as f:
		test_ea = pickle.load(f)
	
	train_ea = train_ea.values.reshape((train_ea.values.shape[0], 1))
	test_ea = test_ea.values.reshape((test_ea.values.shape[0], 1))
	with open('../cache/kfold_5.p', 'rb') as f:
		kf = pickle.load(f)
	
	train_ea_adj = None
	hour_adj = None
	train_hour_adj = None
	
	for train_index, test_index in kf:
		if train_ea_adj is not None:
			train_ea_adj = np.vstack((train_ea_adj, train_ea[test_index]))
		else:
			train_ea_adj = train_ea[test_index]
			
	ea_rows = np.where(train_ea_adj == 1)[0]	
	non_ea_rows = np.where(train_ea_adj == 0)[0]
	test_set_rows_ea = np.where(test_ea == 1)[0]
	
	#print(train1.shape)
	train = np.hstack((train1, train_xlgt, train2, train4, train17, train18, train21, train27, train22, train19, train20,train23, train24, train25, train26, train28, train29, train30, train34, train35, train36, train37, train38))
	test = np.hstack((test1, test_xlgt, test2, test4, test17, test18, test21, test27, test22, test19, test20, test23, test24, test25, test26, test28, test29, test30, test34, test35, test36, test37, test38))
	
	score = 0.0
	X_test_prob_sum = None
	
	for train_index, test_index in kf:
		train_index = np.intersect1d(train_index, ea_rows)
		test_index = np.intersect1d(test_index, ea_rows)
		X_train, X_val, y_train, y_val = train[train_index], train[test_index], y[train_index], y[test_index]
		score_, predictions = run_keras(X_train, y_train, X_val, y_val, epochs=500, ensemble=True,
								predict=[test[test_set_rows_ea]])
		score = score + score_
		if X_test_prob_sum is not None:
			X_test_prob_sum = np.add(X_test_prob_sum, predictions[0])
		else:
			X_test_prob_sum = predictions[0]
	
	X_test_prob_ea = np.multiply(X_test_prob_sum,0.2)
	score1 = score/5.0
	print('ENSEMBLE (EA): {}'.format(round(score1, 5)))
	
	for i in [train35, train33, train23, train27, train37]:
		print(log_loss(y[non_ea_rows].tolist(), i[non_ea_rows]))
	X_val_prob = train35*0.4 + train33*0.4 + train37*0.1 + train23*0.07 + train27*0.03
	score2 = log_loss(y[non_ea_rows].tolist(), X_val_prob[non_ea_rows])
	print('ENSEMBLE (NON-EA): {}'.format(round(score2, 5)))
	test_set_rows_nea = np.where(test_ea == 0)[0]
	X_test_prob_nea = (test35*0.4 + test33*0.4 + test37*0.1 + test23*0.07 + test27*0.03)[test_set_rows_nea]
	'''
	train = np.hstack((train35, train33, train21, train23))
	test = np.hstack((test35, test33, test21, test23))
	X_test_prob_sum = None
	score = 0.0		
	test_set_rows_nea = np.where(test_ea == 0)[0]
	for train_index, test_index in kf:
		train_index = np.intersect1d(train_index, non_ea_rows)
		test_index = np.intersect1d(test_index, non_ea_rows)
		X_train, X_val, y_train, y_val = train[train_index], train[test_index], y[train_index], y[test_index]
		score_, predictions = run_keras(X_train, y_train, X_val, y_val, epochs=500, ensemble=True, nea=True,
								predict=[test[test_set_rows_nea]])
		model = LogisticRegression().fit(X_train, y_train)
		score_ = log_loss(y_val.tolist(), model.predict_proba(X_val))
		
		predictions = [model.predict_proba(test[test_set_rows_nea])]
		score = score + score_
		if X_test_prob_sum is not None:
			X_test_prob_sum = np.add(X_test_prob_sum, predictions[0])
		else:
			X_test_prob_sum = predictions[0]

	X_test_prob_nea = np.multiply(X_test_prob_sum,0.2)
	score2 = score/5.0
	print('ENSEMBLE (NON-EA): {}'.format(round(score2, 5)))
	'''
	X_test_prob = np.vstack((X_test_prob_ea, X_test_prob_nea))
	test_rows = np.hstack((test_set_rows_ea, test_set_rows_nea))
	X_test_prob = X_test_prob[np.argsort(test_rows)]
	#X_test_prob, X_train_prob, X_valid_prob, best_iter = run_mlp(X_train, X_val, y_train, y_val, test, learning_rate=0.01, n_epochs=1000, n_hidden=[30], activation='tanh', early_stop=True)
	
	#mlp = ensemble_mlp(X_train, np.array(y_train), X_val, np.array(y_val))
	#X_test_prob = mlp.predict_proba(sparse.csr_matrix(test))
	#X_valid_prob = mlp.predict_proba(sparse.csr_matrix(X_val))
	#score = log_loss(y_val, X_valid_prob)
	#score = log_loss(y_val, X_train_prob)
	score = ((score1*test_set_rows_ea.shape[0]) + (score2*test_set_rows_nea.shape[0]))/(test_set_rows_nea.shape[0]+test_set_rows_ea.shape[0])

	'''
	print('...XGB')
	dtrain = xgb.DMatrix(X_train, y_train)
	dvalid = xgb.DMatrix(X_val, y_val)
	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	gbm = xgb.train(xgb_params, dtrain, 1000, evals=watchlist, early_stopping_rounds=10, verbose_eval=True, learning_rates = lr)
	gbm = xgb.train(xgb_params, dtrain, gbm.best_ntree_limit, verbose_eval=True, learning_rates = lr)
	X_valid_prob = gbm.predict(xgb.DMatrix(X_val))
	X_test_prob = gbm.predict(xgb.DMatrix(test))	
	'''
	# Write results
	print('# Ensemble - Submit')
	result = pd.DataFrame(X_test_prob, columns=classes)
	result["device_id"] = device_id
	result = result.set_index("device_id")
	sub_file = 'submission_ensemble_' + str(score) + '_' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')) + '.csv'
	result.to_csv(sub_file, index=True, index_label='device_id')
	
	
xgb_params = {
	"objective": "multi:softprob",
	"num_class": 12,
	"booster": 'gbtree',
	"eval_metric": "mlogloss",
	"eta": 0.0055,
	"silent": 1,
	'max_depth': 15,
	'min_child_weight': 11, 	
	'subsample': 0.8,
	'colsample_bytree': 0.3,
	'reg_alpha': 1,
	'gamma': 0.0
}

def lr(i, n):
	if(i < 40): 
		return 0.2
	elif(i < 60):
		return 0.1
	else:
		return 0.0055
		

xgb_params = {
		'booster': "gblinear",
		'num_class': 12,
		'objective': "multi:softprob",
		'eval_metric': "mlogloss",
		'eta': 0.01,
		'max_depth': 8,
		'lambda': 0.55,
		'lambda_bias': 0.5,
		'alpha': 0.5,
		'subsample': 0.75,
		'colsample_bytree': 0.72
		}

xgb_params = {
		'booster': "gblinear",
		'num_class': 12,
		'objective': "multi:softprob",
		'eval_metric': "mlogloss",
		'eta': 0.07,
		'max_depth': 8,
		'alpha': 3
		}

	
def lr(i, n):
	if(i < 35): 
		return 0.07
	else:
		return 0.0055

classes = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
			
def get_events_only_device_features():
	with open('../cache/sparse_train_1.p', 'rb') as f:
		train = pickle.load(f)
	with open('../cache/y.p', 'rb') as f:
		y = pickle.load(f)
	with open('../cache/sparse_train2_1_event_availability.p', 'rb') as f:
		train_ea = pickle.load(f)
		#Sum over column..

	rows = np.where(train_ea.values == 1)[0]
	print(rows.shape)
	train = train[rows,:]
	print(y.shape)
	y = y.values
	y = y[rows]
	with open('../cache/sparse_train_1_event_subset.p', 'wb') as f:
		pickle.dump(train, f)
	with open('../cache/y_event_subset.p', 'wb') as f:
		pickle.dump(y, f)
	#test = test[:,rows]
	
def get_events_available_feature_for_ensemble():
	events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
	events.drop_duplicates(subset='device_id', inplace=True)
	events['bool'] = 1
	print(len(events))
	events = events[['device_id', 'bool']]
	train = pd.read_csv("../input/gender_age_train.csv",
						dtype={'device_id': np.str})
	test = pd.read_csv("../input/gender_age_test.csv",
					   dtype={'device_id': np.str})
	train = pd.merge(train, events, how='left', on='device_id', left_index=True)
	test = pd.merge(test, events, how='left', on='device_id', left_index=True)
	train.fillna(0, inplace=True)
	test.fillna(0, inplace=True)
	
	print(len(train['bool']))
	print(len(test['bool']))
	print(len(train[train['bool'] == 1]))
	print(len(test[test['bool'] == 1]))
	
	with open('../cache/sparse_train2_1_event_availability.p', 'wb') as f:
		pickle.dump(train['bool'], f)
	with open('../cache/sparse_test2_1_event_availability.p', 'wb') as f:
		pickle.dump(test['bool'], f)	
	
if(__name__ == "__main__"):

	#feature_generator()
	#save_pca_data()
	#xgboost_tuning()
	#xgboost(xgb_params)
	#xgboost('gblinear',0.055)
	
	#mlp()
	#kNN()
	#extratrees()
	#randomforest()
	#get_events_available_feature_for_ensemble()
	
	#models_only_on_pbdm()
	first_level_probs(False)
	#ensemble()	
	


		
