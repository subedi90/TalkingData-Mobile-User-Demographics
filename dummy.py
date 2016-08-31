import numpy as np
import pandas as pd

labels = pd.read_csv('../features/labels_feature.csv', dtype={'device_id': np.str, 'label_id': np.str}, index_col=False)

train = pd.read_csv('../features/processed_train_xgb.csv', dtype={'device_id': np.str})
test = pd.read_csv('../features/processed_test_xgb.csv', dtype={'device_id': np.str})

labels_train = pd.merge(train[['device_id']], labels, how='left', on='device_id', left_index=True)
labels_test = pd.merge(test[['device_id']], labels, how='left', on='device_id', left_index=True)

labels_train.fillna('-1',inplace=True)
labels_test.fillna('-1',inplace=True)

count_vect = CountVectorizer()
count_vect.fit(labels['label_id'])
labels_train = count_vect.transform(labels_train['label_id'])
print(labels_train.shape)

labels_test = count_vect.transform(labels_test['label_id'])
print(labels_test.shape)

train = hstack([train[features], labels_train])
test = hstack([test[features], labels_test])