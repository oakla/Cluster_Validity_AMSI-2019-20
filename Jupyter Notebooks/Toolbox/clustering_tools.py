from typing import Dict, Any

from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN

from s_dbw import S_Dbw

import numpy as np

METRIC_SAMPLE_SIZE = None
n_init = 10  # parameter for KMeans: how many times to run the algorithm with various init centroids


def benchmark_kmeans(X: np.ndarray, k_list: [int], print_rslt=True, init_methods=["k-means++"]):
    n_samples, n_features = X.shape

    if print_rslt:
        print("n_samples %d, \t n_features %d"
              % (n_samples, n_features))
        print("Using the following methods for intial centroid placement: {0}".format(init_methods))
        print(82 * '_')
        heading_string = 'init/k-values\t'
        for k in k_list:
            heading_string += '%-8i' % k
        print(heading_string)

    for init in init_methods:
        k_labels_dict = {}
        scores = {}
        np.random.seed(42)
        # get cluster labels for various k values
        for k in k_list:
            kmeans = KMeans(n_clusters=k, init=init, n_init=n_init).fit(X)
            centroids = kmeans.cluster_centers_
            estimated_labels = kmeans.labels_
            inertia = kmeans.inertia_
            k_labels_dict[k] = estimated_labels
        # send clusters to validity checker
        for k, labels in k_labels_dict.items():
            score = metrics.silhouette_score(X, labels,
                                             metric='euclidean',
                                             sample_size=METRIC_SAMPLE_SIZE)
            scores[k] = score
        if print_rslt:
            row_string = '%-9s' % init
            for k_scores in scores.values():
                row_string += '\t%.3f' % k_scores
            print(row_string, '\n')

        best_cluster = max(scores, key=scores.get)
        return best_cluster, k_labels_dict, scores


def benchmark_dbscan(X: np.ndarray, min_pts: int, eps_list: [float], print_rslt=False,
                     validity_metrics=['silhouette', 's_dbw']):
    n_samples, n_features = X.shape

    if print_rslt:
        print("Running DBSCAN %d times..." % len(eps_list))
        print("n_samples %d, \t n_features %d"
              % (n_samples, n_features))
        print("Using the following parameters: \n\tmin_pts = {0}\n\teps = {1}".format(min_pts, eps_list))

    eps_label_dict = {}
    dbs = []
    np.random.seed(42)
    # get cluster labels for various k values
    for eps in eps_list:
        db = DBSCAN(eps=eps, min_samples=min_pts).fit(X)
        est_labels = db.labels_
        n_clusters_ = len(set(est_labels)) - (1 if -1 in est_labels else 0)
        n_noise_ = list(est_labels).count(-1)
        eps_label_dict[eps] = db.labels_
        dbs.append(db)
        if print_rslt:
            print("min_pts: %d, \t eps: %f" % (min_pts, eps))
            print('Estimated number of clusters: %d' % n_clusters_)
            print('Estimated number of noise points: %d' % n_noise_)

    # send clusters to validity checkers

    metric_scores_dict = multi_validity(X, eps_label_dict, validity_metrics, print_rslt)

    if print_rslt:

        for key_1, scores in metric_scores_dict.items():
            row_string = '%-9s' % key_1
            for score in scores:
                row_string += '\t%.3f' % score
            print(row_string, '\n')

    return metric_scores_dict, dbs


def multi_validity(X, labels_dict, validity_metrics, print_rslt=False):
    metric_scores_dict = {}
    for metric in validity_metrics:

        scores = []
        for parameter, labels in labels_dict.items():
            if metric == 's_dbw':
                score = S_Dbw(X, labels)

            elif metric == 'silhouette':
                score = metrics.silhouette_score(X, labels,
                                                 metric='euclidean',
                                                 sample_size=METRIC_SAMPLE_SIZE)
            else:
                print("error: no valid metric supplied")
            scores.append(score)
            if print_rslt:
                print("parameter = %f.2 yields a %s score of %f" % (parameter, metric, score))
        metric_scores_dict[metric] = scores

    return metric_scores_dict
