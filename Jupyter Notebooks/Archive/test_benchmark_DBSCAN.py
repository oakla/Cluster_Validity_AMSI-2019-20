from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN

import Toolbox.clustering_tools as clstr_tools



from sklearn.preprocessing import scale
X, y = load_iris(return_X_y=True)
X_scaled = scale(X)
n_samples, n_features = X.shape

min_pts = 8

a = clstr_tools.benchmark_dbscan(X, min_pts=min_pts, eps_list=[0.5], print_rslt=True)



