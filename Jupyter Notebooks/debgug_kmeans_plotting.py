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

k_list = list(range(2,21))

k_with_highest_score, k_labels_dict, scores = clstr_tools.benchmark_kmeans(X_scaled, k_list, print_rslt=False)

k=2
plt_tools.visualize_clusters(X, k_labels_dict[k], dims=2)
plt.title("K-Means Result with K=%d" % k, fontsize = 20)
plt.show()