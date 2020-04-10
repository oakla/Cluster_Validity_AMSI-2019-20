from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
import numpy as np
from sklearn.neighbors import kneighbors_graph

import matplotlib._color_data as mcd

# A list of markers to be used for plotting cluster assignments
from sklearn.preprocessing import scale

markers = ['o', 'v', '^', 's', 'p', 'd']


def visualize_iris_ground_truth_2d():
    data = load_iris()
    visualize_labeling_ground_truth_2d(data)


def visualize_wine_ground_truth_2d():
    data = load_wine()
    visualize_labeling_ground_truth_2d(data)


def visualize_labeling_ground_truth_2d(data):
    # adapted from:
    # - https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Data_Visualization_Iris_Dataset_Blog.ipynb

    X = scale(data.data)
    y = data.target

    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA of Ground Truth', fontsize=20)

    # Reorder the labels to have colors matching the cluster results
    #y = np.choose(y, [1, 2, 0]).astype(np.float)
    targets = data.target_names
    colors = ['r', 'g', 'b']
    ax.scatter(X[:, 0], X[:, 1], c=y,
               # cmap=plt.get_cmap('Dark2'),
               edgecolors='k',
               s=70)

    # ax.legend(targets)
    ax.grid(b=True)


def visualize_iris_ground_truth_3d():
    data = load_iris()
    visualize_labeling_ground_truth_3d(data)


def visualize_wine_ground_truth_3d():
    data = load_wine()
    visualize_labeling_ground_truth_3d(data)


def visualize_labeling_ground_truth_3d(data):
    # adapted from:
    # - https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html and
    # - https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html

    X = data.data
    y = data.target

    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])


def visualize_clusters(X, est_y, dims=2, labels=None):
    # adapted from:
    # - https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html and
    # - https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
    clusters = np.unique(est_y)
    n_clusters = len(clusters) - (1 if -1 in est_y else 0)
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()
    colors = np.linspace(0, 1, n_clusters)
    pca = decomposition.PCA(n_components=dims)
    pca.fit(X)
    x_pca = pca.transform(X)

    if dims == 3:
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.cla()
        ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=est_y.astype(float),
                   # cmap="Dark2",
                   edgecolor='k')

        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_zlabel('Principal Component 3', fontsize=15)
        ax.set_title('3 Component PCA', fontsize=20)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
    elif dims == 2:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 Component PCA of {0} clusters'.format(n_clusters), fontsize=20)
        for c, i in enumerate(clusters):
            if labels is not None:
                for t in np.unique(labels):
                    if i == -1:
                        edgecolor = 'k'
                        marker = 'x'
                    else:
                        edgecolor = plt.cm.Set3.__call__([c % 9])
                        marker = markers[c % len(markers)]
                    color = plt.cm.Dark2.__call__([t % 9])
                    ax.scatter(x_pca[(est_y == i) & (labels == t), 0], x_pca[(est_y == i) & (labels == t), 1],
                               c=color,
                               s=100,
                               marker=marker,
                               edgecolors=edgecolor,
                               linewidths=2)
            else:
                if i == -1:
                    color = 'k'
                    marker = 'x'
                else:
                    color = plt.cm.Set1.__call__([c % 9])
                    marker = markers[c % len(markers)]
                ax.scatter(x_pca[est_y == i, 0], x_pca[est_y == i, 1], c=color,
                           marker=marker,
                           s=80)

        ax.grid(b=True)

    # plt.show()
    return ax


def get_k_distance_plot(X: np.ndarray, k: int) -> (int, np.ndarray):
    dist_matrix = kneighbors_graph(X, k, mode='distance').toarray()
    max_dist_to_kth_neighbor = dist_matrix.max(axis=1)

    plt.plot(np.sort(max_dist_to_kth_neighbor))

    plt.grid(b=True)


def radboud_cluster_plot(X, clusterid, centroids=None, y=None):
    '''
    CLUSTERPLOT Plots a clustering of a data set as well as the true class
    labels. If data is more than 2-dimensional it should be first projected
    onto the first two principal components. Data objects are plotted as a dot
    with a circle around. The color of the dot indicates the true class,
    and the cicle indicates the cluster index. Optionally, the centroids are
    plotted as filled-star markers, and ellipsoids corresponding to covariance
    matrices (e.g. for gaussian mixture models).

    Usage:
    clusterplot(X, clusterid)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix, covars=c_tensor)

    Input:
    X           N-by-M data matrix (N data objects with M attributes)
    clusterid   N-by-1 vector of cluster indices
    centroids   K-by-M matrix of cluster centroids (optional)
    y           N-by-1 vector of true class labels (optional)
    '''

    X = np.asarray(X)
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    cls = np.asarray(clusterid)
    if y is None:
        y = np.zeros((X.shape[0], 1))
    else:
        y = np.asarray(y)
    if centroids is not None:
        centroids = np.asarray(centroids)
    K = np.size(np.unique(cls))
    C = np.size(np.unique(y))
    ncolors = np.max([C, K])

    # plot data points color-coded by class, cluster markers and centroids
    # plt.hold(True)
    colors = [0] * ncolors
    for color in range(ncolors):
        colors[color] = plt.cm.jet.__call__((color * 255) // (ncolors - 1))[:3]
    for i, cs in enumerate(np.unique(y)):
        plt.plot(X[(y == cs).ravel(), 0], X[(y == cs).ravel(), 1], 'o',
                 markeredgecolor='k', markerfacecolor=colors[i], markersize=6,
                 zorder=2)
    for i, cr in enumerate(np.unique(cls)):
        plt.plot(X[(cls == cr).ravel(), 0], X[(cls == cr).ravel(), 1], 'o',
                 markersize=12, markeredgecolor=colors[i],
                 markerfacecolor='None', markeredgewidth=3, zorder=1)
    if centroids is not None:
        for cd in range(centroids.shape[0]):
            plt.plot(centroids[cd, 0], centroids[cd, 1], '*', markersize=22,
                     markeredgecolor='k', markerfacecolor=colors[cd],
                     markeredgewidth=2, zorder=3)
    # plt.hold(False)

    # create legend
    legend_items = (np.unique(y).tolist() + np.unique(cls).tolist() +
                    np.unique(cls).tolist())
    for i in range(len(legend_items)):
        if i < C:
            legend_items[i] = 'Class: {0}'.format(legend_items[i])
        elif i < C + K:
            legend_items[i] = 'Cluster: {0}'.format(legend_items[i])
        else:
            legend_items[i] = 'Centroid: {0}'.format(legend_items[i])
    plt.legend(legend_items, numpoints=1, markerscale=.75, prop={'size': 9})
