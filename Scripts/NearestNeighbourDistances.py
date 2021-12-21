import numpy as np
from sklearn.neighbors import kneighbors_graph

from matplotlib import pyplot as plt




def get_k_distance_plot(X: np.ndarray, k: int) -> (int, np.ndarray):

    # X (data), k (the number of nearest neighbours that define a datapoints label-identity)
    dist_matrix = kneighbors_graph(X, k, mode='distance').toarray()
    max_dist_to_kth_neighbor = dist_matrix.max(axis=1)

    # plot a histogram of distance to furtherest neighbor of each data point
    n, bins, patches = plt.hist(max_dist_to_kth_neighbor, bins=np.linspace(0,10,20))
    plt.show()

    
    plt.plot(np.sort(max_dist_to_kth_neighbor))
    plt.show()

    return n, bins