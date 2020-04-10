'''
It seems that the best settings for DBSCAN that I found where eps=2.75 and min_pts
'''
import numpy as np
from skimage.feature import hog
from sklearn import preprocessing
from collections import Counter

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')
# X, y = mnist['data'], mnist['target']
# data = np.array(X, 'int16')
# target = np.array(y, 'int')
import NearestNeighbourDistances

X, y_digits = load_digits(return_X_y=True)
data = X
known_labels = y_digits

digits = load_digits()
plt.figure(6, figsize=(3,3))
plt.imshow(digits.images[960],cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

list_hog = []
for feature in data:
    fd, image = hog(feature.reshape((8, 8)), orientations=9, pixels_per_cell=(2,2), cells_per_block=(1,1), visualize=True)
    list_hog.append(fd)

hog_features = np.array(list_hog,'float64')

# now we run DBSCAN
inputs = hog_features

n_samples, n_features = inputs.shape
n_digits = len(np.unique(y_digits))

db = DBSCAN().fit(inputs)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
estimated_labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(estimated_labels)) - (1 if -1 in estimated_labels else 0)
n_noise_ = list(estimated_labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

print('plotting...')
eps_est = NearestNeighbourDistances.get_k_distance_plot(inputs, 19)

eps = eps_est[1][np.argmax(eps_est[0])]

np.random.seed(42)
db = DBSCAN(eps=2.75, min_samples=5).fit(inputs)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
estimated_labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(estimated_labels)) - (1 if -1 in estimated_labels else 0)
n_noise_ = list(estimated_labels).count(-1)
print("New cluster estimate after using kdist for eps")
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
