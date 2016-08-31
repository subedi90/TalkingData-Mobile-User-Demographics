import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app_labels = pd.read_csv('../input/app_labels.csv', dtype={'label_id': np.str},index_col=False)
app_labels['label_id'] = app_labels.groupby(['app_id'])['label_id'].transform(lambda x: ' '.join(x))
app_labels.drop_duplicates('app_id',keep='first',inplace=True)
#print(app_labels)

app_events = pd.read_csv('../input/app_events.csv', index_col=False)
app_events  = app_events.loc[app_events['is_active']==1]
event_labels = pd.merge(app_events, app_labels, how='left', on=['app_id'], left_index=True)[['event_id','label_id']]
event_labels['label_id'] = event_labels.groupby(['event_id'])['label_id'].transform(lambda x: ' '.join(x))
event_labels.drop_duplicates('event_id',keep='first',inplace=True)
print(event_labels)

events = pd.read_csv('../input/events.csv', dtype={'device_id': np.str, 'label_id': np.str}, index_col=False)
device_labels = pd.merge(events, event_labels, how='left', on=['event_id'], left_index=True)[['device_id','label_id']].dropna()
print(device_labels)
device_labels['label_id'] = device_labels.groupby(['device_id'])['label_id'].transform(lambda x: ' '.join(x.dropna()))
device_labels.drop_duplicates('device_id', keep='first',inplace=True)
print(device_labels)

for c in adj_col:
	hour[c] = (hour[c] - hour[c].mean())/hour[c].std(ddof=0)

device_labels.to_csv('../features/labels_feature.csv')
