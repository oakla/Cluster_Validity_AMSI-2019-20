from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from NearestNeighbourDistances import get_k_distance_plot
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

X_iris, y_iris = load_iris(return_X_y=True)
inputs = scale(X_iris)
n_samples, n_features = inputs.shape

# Settings
min_pts = n_features*2
eps = 0.75
run_k_distance = False

if run_k_distance:
    frequencies, bin_boundaries = get_k_distance_plot(inputs, min_pts - 1)
    left_most_boundary_of_most_frequent_bin = bin_boundaries[np.argmax(frequencies)] #This is broken. Frequencies does not include empty bins

db = DBSCAN(eps=eps, min_samples=min_pts).fit(inputs)
estimated_labels = db.labels_
n_clusters = len(set(estimated_labels)) - (1 if -1 in estimated_labels else 0)
n_noise_pts = list(estimated_labels).count(-1)

fig = plt.figure(1, figsize=(4,3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(inputs)
inputs_pca = pca.transform(inputs)

ax.scatter(inputs_pca[:, 0], inputs_pca[:, 1], inputs_pca[:, 2], c=estimated_labels,
           cmap=plt.cm.nipy_spectral,
           edgecolor='k')

plt.show()


