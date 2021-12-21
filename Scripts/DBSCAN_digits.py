import numpy as np
from sklearn.preprocessing import scale
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits

import NearestNeighbourDistances

# Meanings
NOISE_CLUSTER_LABEL = -1
SEED_VALUE = 42

# Options
PRINT = True

# init
np.random.seed(SEED_VALUE)

# load data
X_digits, y_digits = load_digits(return_X_y=True)

# Data Transform
inputs = scale(X_digits)

# get metadata
n_samples, n_features = inputs.shape
n_digits = len(np.unique(y_digits))

# Fit using default parameters of DBCSCAN
# fit model
db = DBSCAN().fit(inputs)

# create a 1D boolean array representing which samples are 'core' samples
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# creating a new name as a reminder of what db.labels is
estimated_labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
#   if there is a 'noise' cluster, do not count it in the number of clusters
n_clusters_ = (len(set(estimated_labels)) 
                - (1 if NOISE_CLUSTER_LABEL in estimated_labels else 0))

# how many datapoints were counted as noise
n_noise_ = list(estimated_labels).count(-1)

# output
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# Run DBSCAN with custom eps value

# Plot k-distance
print('plotting k-distance...')
eps_est = NearestNeighbourDistances.get_k_distance_plot(inputs, 20)


eps = eps_est[1][np.argmax(eps_est[0])]

# TODO: rewrite to automate use of k_distance for eps
#   eps (short for epsilon) defines maximum distance between points in a cluster
#   one recommended way to determine a 'good' value for epsilon is to use the elbow
#   of a k-distance graph
np.random.seed(SEED_VALUE)
db = DBSCAN(eps=4.25).fit(inputs)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True


estimated_labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
#   if there is a 'noise' cluster, do not count it in the number of clusters
n_clusters_ = len(set(estimated_labels)) - (1 if -1 in estimated_labels else 0)


n_noise_ = list(estimated_labels).count(-1)
print("New cluster estimate after using kdist for eps")
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Yay. We now have clusters...
# What if we did feature extraction instead of just flattening the pixel array?


# if PRINT:
#     print("n_digits: %d, \t n_samples %d, \t n_features %d"
#           % (n_digits, n_samples, n_features))
#     print(78 * '_')
#     heading_string = 'init/k-values\t\t'
#     for k in K_VALUES:
#         heading_string += '%-8i' % k
#     print(heading_string)

# init_by_k_scores = {}
# for init in INITS:
#     multi_estimated_labels = multi_k_means(inputs, K_VALUES, init=init)
#     init_by_k_scores[init] = multi_silhouette(inputs, multi_estimated_labels)
#     if print_:
#         row_string = '%-9s\t\t' % init
#         for k_scores in init_by_k_scores[init].values():
#             row_string += '\t%.3f' % k_scores
#         print(row_string)
