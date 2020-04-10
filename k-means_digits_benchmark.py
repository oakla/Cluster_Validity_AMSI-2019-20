'''this code is adapted from code found at https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html'''

import numpy as np
from sklearn.preprocessing import scale

from Toolbox.multi_validity_checker import multi_silhouette
from Toolbox.multi_Cluster import multi_k_means

from sklearn.datasets import load_digits

# Macro Values
K_VALUES = [6,7,8,9,10,11,12]
print_ = True
INITS = ['k-means++', 'random']



#load data
X_digits, y_digits = load_digits(return_X_y=True)
inputs = scale(X_digits)

n_samples, n_features = inputs.shape
n_digits  = len(np.unique(y_digits))
labels = y_digits

if print_:
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))
    print(78 * '_')
    heading_string = 'init/k-values\t\t'
    for k in K_VALUES:
        heading_string += '%-8i' % k
    print(heading_string)

init_by_k_scores = {}
for init in INITS:
    np.random.seed(42)
    multi_estimated_labels = multi_k_means(inputs, K_VALUES, init=init)
    init_by_k_scores[init] = multi_silhouette(inputs, multi_estimated_labels)
    if print_:
        row_string = '%-9s\t\t' % init
        for k_scores in init_by_k_scores[init].values():
            row_string += '\t%.3f' % k_scores
        print(row_string)
