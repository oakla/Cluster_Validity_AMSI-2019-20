from sklearn.cluster import k_means
from sklearn.decomposition import PCA

def multi_k_means(inputs, k_list, init='k-means++'):
    ret_dict = {}
    init_name_str = init
    n_init = 10
    for k in k_list:
        centroids, estimated_labels, intertia = k_means(X=inputs, n_clusters=k, init=init,n_init=n_init)
        ret_dict[k] = estimated_labels
    return ret_dict
