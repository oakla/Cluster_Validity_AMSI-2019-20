

# Event clustering with machine learning


import pandas as pd

attributes_df = pd.read_csv('Mawson_attributes.txt', sep=',')
attributes_df['start_time'] = pd.to_datetime(attributes_df['start_time'])
attributes_df['stop_time'] = pd.to_datetime(attributes_df['stop_time'])

attributes_df.columns



attributes_df = attributes_df.drop(columns=['attribute_11', 'attribute_12', 'attribute_34', 'attribute_35', \
                                            'attribute_36', 'attribute_37'])





### Attribute/feature analysis


attributes_df.dtypes


import seaborn as sn
import matplotlib.pyplot as plt


def boxplot(attributes_df, ymin=None, ymax=None):
    # create new figure
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    # set axis labels and scale
    ax.tick_params(axis='x', labelrotation=90)
    ax.set(ylabel='Attribute value')
    if not ymin == None and not ymax == None:
        plt.ylim([ymin, ymax])

    # plot boxplot using seaborn
    sn.boxplot(data=attributes_df.drop(columns=['start_time', 'stop_time']), orient='v', ax=ax)

    plt.show()


boxplot(attributes_df, ymin=0, ymax=1e6)


import numpy as np

# create empty dataframe with headings for each attribute
attribute_scaling_df = pd.DataFrame(columns=attributes_df.columns[2:len(attributes_df.columns)])

# set scaling for each attribute to 'linear' by default
a = list()
for i in range(0, len(attribute_scaling_df.columns)):
    a.append('linear')
attribute_scaling_df.loc[0] = a

# manually change scaling for attributes
attribute_scaling_df['attribute_1'][0] = 'log'
attribute_scaling_df['attribute_24'][0] = 'log'
attribute_scaling_df['attribute_25'][0] = 'log'
attribute_scaling_df['attribute_26'][0] = 'log'
# attribute_scaling_df['attribute_34'][0] = 'log'
# attribute_scaling_df['attribute_35'][0] = 'log'
# attribute_scaling_df['attribute_36'][0] = 'log'
# attribute_scaling_df['attribute_37'][0] = 'log'

print(attribute_scaling_df)

# create a copy of the database to applying the scaling corrections
attributes_scaled_df = attributes_df.copy()

# rescale each attribute to log or linear scale centred on zero
for i in range(0, len(attribute_scaling_df.columns)):
    if attribute_scaling_df[attribute_scaling_df.columns[i]][0] == 'linear':
        attributes_scaled_df[attribute_scaling_df.columns[i]] \
            = attributes_scaled_df[attribute_scaling_df.columns[i]] / \
              np.median(attributes_df[attribute_scaling_df.columns[i]]) - 1
    else:
        attributes_scaled_df[attribute_scaling_df.columns[i]] \
            = np.log10(attributes_scaled_df[attribute_scaling_df.columns[i]] / \
                       np.median(attributes_df[attribute_scaling_df.columns[i]]))

# rescale the distribution of each attribute to have standard deviation of unity
for i in range(0, len(attribute_scaling_df.columns)):
    attributes_scaled_df[attribute_scaling_df.columns[i]] \
        = attributes_scaled_df[attribute_scaling_df.columns[i]] / \
          np.std(attributes_scaled_df[attribute_scaling_df.columns[i]])


boxplot(attributes_scaled_df, ymin=-5, ymax=5)


### Correlation analysis


import seaborn as sn
import matplotlib.pyplot as plt


def correlation_plot(attributes_df):
    # create correlation matrix and mask for upper triangular matrix
    corrMatt = attributes_df.drop(columns=['start_time', 'stop_time']).corr()
    mask = np.array(corrMatt)
    mask[np.tril_indices_from(mask)] = False

    # create correlation plot
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 10)
    sn.heatmap(corrMatt, mask=mask, vmin=-1, vmax=1, square=True, annot=True, annot_kws={'size': 9}, fmt='.2f')



correlation_plot(attributes_scaled_df)


### Self-Organising Maps with TensorFlow






import numpy as np
import pandas as pd
from minisom import MiniSom


def self_organising_maps(attributes_df, size=32):
    # define matrix of attributes and values
    X = attributes_df.iloc[:, 2:len(attributes_df.columns)]

    # run self-organising maps algorithm
    # create an object
    som = MiniSom(x=size, y=size, input_len=len(attributes_df.columns) - 2, sigma=1.5 * np.sqrt(size),
                  learning_rate=0.5)
    ### maybe sigma=5*np.sqrt(size) and learning_rate=1
    # initialize the weights and train
    som.random_weights_init(X.values)
    som.train_random(X.values, num_iteration=2 * len(attributes_df))
    # som.train_batch(X.values, num_iteration=len(attributes_df)) # this gives more predictable results

    print('There are ' + str(len(attributes_df)) + ' events assigned to ' + str(size * size) + ' neurons.')

    return som






def get_locations(som):
    # create empty array
    locations = np.zeros_like(som.distance_map()).tolist()

    # add locations to array
    for i in range(0, len(locations[:][0])):
        for j in range(0, len(locations[0][:])):
            locations[i][j] = (i, j)

    return np.asarray(locations)



import matplotlib.pyplot as plt
import matplotlib.colors as colors


def distance_plotter(attributes_df, som, labels=None):
    # setup figure
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    ax.axis('off')
    ax.set_aspect('equal')

    # correct weightings of edge points (for lack of neighbours)
    mid = som.distance_map()
    mid[:, 0] = np.minimum(1, mid[:, 0] * 1.5)
    mid[:, len(mid[0, :]) - 1] = np.minimum(1, mid[:, len(mid[0, :]) - 1] * 1.5)
    mid[0, :] = np.minimum(1, mid[0, :] * 1.5)
    mid[len(mid[:, 0]) - 1, :] = np.minimum(1, mid[len(mid[:, 0]) - 1, :] * 1.5)

    # create heatmap of mean inter-neuron distance
    plt.pcolor(mid.T, norm=colors.PowerNorm(gamma=1. / 2), cmap='RdPu', vmin=np.quantile(mid, 0.01), vmax=1)
    cb = plt.colorbar()
    cb.set_label('Mean inter-neuron distance')

    # calculate number of events in each neuron
    if labels is None:
        count = np.zeros_like(mid)
        X = attributes_df.iloc[:, 2:len(attributes_df.columns)]
        for x in X.values:
            w = som.winner(x)
            count[w[0], w[1]] += 1

        # plot circles over heatmap to represent number of events per neuron
        plt.plot(1, 1)
        for i in range(0, len(count[:, 0])):
            for j in range(0, len(count[0, :])):
                plt.plot(i + 0.5, j + 0.5, marker='o', markeredgecolor='black', markerfacecolor='None', \
                         markersize=400 * np.sqrt(count[i, j] / np.max(count)) / len(count[0, :]))
    else:
        # plot contour map of mean inter-neuron distance
        for i in range(0, int(np.max(labels)) + 1):
            plt.contour(np.minimum(np.absolute(labels.T - i), 0.5), colors='k', \
                        levels=[0.15], \
                        linewidths=1, origin='lower')

        # add labels to each cluster
        locations = get_locations(som)
        for i in range(0, int(np.max(labels)) + 1):
            # find median x and y location for cluster
            result = np.where(labels == i)
            xs = [x[0] for x in locations[result]]
            ys = [x[1] for x in locations[result]]
            plt.annotate('cluster ' + str(i + 1), xy=(np.median(xs) + 0.5, np.median(ys) + 0.5), \
                         horizontalalignment='center', verticalalignment='center', size=9.5, \
                         bbox=dict(boxstyle="round", alpha=0.5, fc=(1.0, 1.0, 1.0), ec="none"))

    plt.show()




som = self_organising_maps(attributes_scaled_df, size=64)


distance_plotter(attributes_scaled_df, som)


### Identifying clusters of neurons



from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import davies_bouldin_score


def clustering(som, min_clusters=5, max_clusters=20):
    # create array of the centroid locations of each neuron
    array3D = som.get_weights()
    array2D = array3D.reshape((array3D.shape[0] * array3D.shape[1]), array3D.shape[2])

    # find optimal number of clusters using Davies-Bouldin score
    prev_score = 99.99
    for i in range(min_clusters, max_clusters + 1):
        # find clusters for current number of clusters
        cluster = SpectralClustering(n_clusters=i, affinity='rbf')  # can use AgglomerativeClustering instead
        new_labels = cluster.fit_predict(array2D)
        new_score = davies_bouldin_score(array2D, new_labels)

        # test if new cluster is better than previous cluster
        if new_score < prev_score:
            prev_score = new_score
            labels = new_labels.copy()

    # convert labels to a 3D array
    labels = labels.reshape(array3D.shape[0], array3D.shape[1])

    return labels



labels = clustering(som, min_clusters=3, max_clusters=20)


distance_plotter(attributes_scaled_df, som, labels=labels)




def cluster_locations(cluster_number, labels, som):
    # get location of each cluster
    locations = get_locations(som)
    # find locations in correct cluster (numbered as shown in plot; i.e. n + 1)
    result = np.where(labels == cluster_number - 1)

    return np.asarray(locations[result])


print(cluster_locations(1, labels, som))

### Finding events in neurons of interest


def neuron_inversion(attributes_df, som, locations):
    # catch a single location not defined using a list
    if not isinstance(locations, (list, np.ndarray)):
        locations = [locations]

    # define matrix of attributes and values
    X = attributes_df.iloc[:, 2:len(attributes_df.columns)]
    # define empty pandas dataframe to store event times
    event_times_df = pd.DataFrame(columns=['start_time', 'stop_time'])

    # use a dictionary to obtain the mappings
    mappings = som.win_map(X.values)

    # get the events assigned to the neuron at location (x,y)
    events_df = pd.DataFrame(columns=X.columns)
    i = 0
    for location in locations:
        events = mappings[(location[0], location[1])]
        for event in events:
            events_df.loc[i] = event
            i = i + 1

    # find start time of event
    event_times_df = pd.DataFrame(columns=['start_time', 'stop_time'])
    for i in range(0, len(events_df)):
        event_times_df.loc[i] = list([attributes_df['start_time'][finder(X, events_df.iloc[i, :])].item(), \
                                      attributes_df['stop_time'][finder(X, events_df.iloc[i, :])].item()])

    # convert pandas database to datatime objects
    event_times_df['start_time'] = pd.to_datetime(event_times_df['start_time'], unit='ns')
    event_times_df['stop_time'] = pd.to_datetime(event_times_df['stop_time'], unit='ns')

    return event_times_df.sort_values(['start_time'], ascending=True).reset_index(drop=True)



def finder(X, row):
    df = X.copy()
    # find the row with a perfect match for all attributes
    for col in df:
        df = df.loc[df[col] == row[col]]

    # return the row number for the matched event
    return df.index


event_times_df = neuron_inversion(attributes_scaled_df, som, locations=cluster_locations(1, labels, som))

print(event_times_df.to_string())




def cluster_events(attributes_scaled_df, labels, som):
    # find events associated with each cluster and add to pandas dataframe
    for i in range(0, int(np.max(labels)) + 1):
        df = neuron_inversion(attributes_scaled_df, som, \
                              locations=cluster_locations(i + 1, labels, som))
        df['cluster'] = i + 1
        # add dataframe to existing dataframe if one exists
        if i == 0:
            event_times_df = df.copy()
        else:
            event_times_df = event_times_df.append(df, ignore_index=True)

    return event_times_df


event_times_df = cluster_events(attributes_scaled_df, labels, som)

event_times_df


event_times_df.to_csv('Mawson_clusters.txt', index=False)


### Agglomerative and spectral clustering methods




from sklearn.cluster import AgglomerativeClustering, SpectralClustering


def raw_clustering(attributes_df, min_clusters=5, max_clusters=20):
    # create array of the attributes for each event
    array2D = attributes_df.iloc[:, 2:len(attributes_df.columns)].values[0:1000, :]

    # find optimal number of clusters using Davies-Bouldin score
    prev_score = 99.99
    for i in range(min_clusters, max_clusters + 1):
        # find clusters for current number of clusters
        cluster = SpectralClustering(n_clusters=i, affinity='rbf')  # can use AgglomerativeClustering instead
        new_labels = cluster.fit_predict(array2D)
        new_score = davies_bouldin_score(array2D, new_labels)

        # test if new cluster is better than previous cluster
        if new_score < prev_score:
            prev_score = new_score
            labels = new_labels.copy()

    return labels



raw_labels = raw_clustering(attributes_scaled_df, min_clusters=8, max_clusters=30)


print(raw_labels[raw_labels > 0])



