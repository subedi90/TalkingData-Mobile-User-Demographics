import pandas as pd
import numpy as np

print("# Read Train")
train = pd.read_csv("../input/gender_age_train.csv", usecols=['device_id'])
print('Generate Row_ID')
train.reset_index(level=0, inplace=True)
train = train['index'] / (len(train))
print(train.min())
print(train.max())
np.save('../cache/train_row_id', train.values)
del train

print("# Read Test")
train = pd.read_csv("../input/gender_age_test.csv", usecols=['device_id'])
print('Generate Row_ID')
train.reset_index(level=0, inplace=True)
train = train['index'] / (len(train))
print(train.min())
print(train.max())
np.save('../cache/test_row_id', train.values)
del train
