import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale

from Toolbox.multi_Cluster import multi_k_means
from Toolbox.multi_validity_checker import multi_silhouette

# Macro Values
K_VALUES = [2,3,4,5,6,7]
print_ = True
INITS = ['k-means++', 'random',"PCA-based"]
data = 'data/iris.csv'

np.random.seed(42)

#load data
col_indexes = pd.read_csv(data, nrows=1).columns
inputs = pd.read_csv(data, header = None, usecols=range(4)).to_numpy()
n_samples, n_features = inputs.shape
#get labels
labels = pd.read_csv(data, header = None, usecols=[len(col_indexes)-1]).to_numpy().flatten()
n_classes = len(np.unique(labels))
labels_integer = np.zeros(len(labels))
for i, name in enumerate(np.unique(labels)):
    labels_integer[name == labels] = i

#pre-process data
scaled_inputs = scale(inputs)

if print_:
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_classes, n_samples, n_features))
    print(78 * '_')
    heading_string = 'init/k-values\t\t'
    for k in K_VALUES:
        heading_string += '%-8i' % k
    print(heading_string)


init_by_k_scores = {}
for init in INITS:
    # get cluster labels for various k values
    multi_estimated_labels = multi_k_means(inputs, K_VALUES, init=init)
    # send clusters to validity checker
    init_by_k_scores[init] = multi_silhouette(inputs, multi_estimated_labels)
    if print_:
        row_string = '%-9s\t\t' % init
        for k_scores in init_by_k_scores[init].values():
            row_string += '\t%.3f' % k_scores
        print(row_string)

