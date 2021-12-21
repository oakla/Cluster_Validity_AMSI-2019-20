from sklearn import metrics

METRIC_SAMPLE_SIZE = None
# for each label set (i.e. for each value of k)
#   get score
#   add to validity list

#   Here we get the silhouette scores. These scores range from -1 to 1.
#       1 is the best, -1 is the worst
def multi_silhouette(inputs, multi_estimated_labels):
    scores = {}
    for k, labels in multi_estimated_labels.items():
        score = metrics.silhouette_score(inputs, labels,
                                         metric='euclidean',
                                         sample_size=METRIC_SAMPLE_SIZE)
        scores[k] = score
    return scores
