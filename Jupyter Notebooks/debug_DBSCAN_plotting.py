from sklearn.datasets import load_iris
from sklearn.preprocessing import scale

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import Toolbox.plotting_tools as plt_tools
import Toolbox.clustering_tools as clstr_tools

from sklearn import metrics


import importlib
importlib.reload(plt_tools)


X, y = load_iris(return_X_y=True)
X_scaled = scale(X)
n_samples, n_features = X.shape

min_pts = 4
eps = 0.6
np.random.seed(42)
db = DBSCAN(eps=eps, min_samples=min_pts).fit(X_scaled)
DBSCAN_estimated_labels_1 = db.labels_
est_labels = DBSCAN_estimated_labels_1
n_clusters_ = len(set(est_labels)) - (1 if -1 in est_labels  else 0)
n_noise_ = list(est_labels ).count(-1)

print("min_pts: %d, \t eps: %f" % (min_pts, eps))
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

plt_tools.visualize_clusters(X, DBSCAN_estimated_labels_1 , dims=2)