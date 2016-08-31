import datetime
import time
import pandas as pd
import numpy as np
import pickle as pk

import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

##############################################################################
#Load Dataset...

print('Read events...')
events = pd.read_csv('../input/events.csv', usecols = ['longitude','latitude'], dtype={'device_id': np.str})
events.drop_duplicates(inplace=True)
events = events[(events.longitude != 0) & (events.latitude != 0)]
X = events.as_matrix()
print(X[:3])

##############################################################################
# Compute clustering with KMeans

mbk = KMeans(init='k-means++', n_clusters=10,
                      n_init=10, verbose=0)
t0 = time.time()
X_ = mbk.fit_predict(X)
t_mini_batch = time.time() - t0
print(t_mini_batch)
#mbk_means_labels = mbk.labels_
#mbk_means_cluster_centers = mbk.cluster_centers_
#mbk_means_labels_unique = np.unique(mbk_means_labels)

##############################################################################
#Save Cluster Model...
with open("../models/geolocation_mbk", "wb") as f:
	pk.dump(mbk, f)
	
##############################################################################
#Plot result...
plt.scatter(X[:, 0], X[:, 1], c=X_)
plt.show()

##############################################################################
#Save Location Cluster Info...
loc = pd.DataFrame(X, columns=['longitude','latitude'])
loc['Cluster'] = X_
loc.to_csv('../features/geolocation.csv')
