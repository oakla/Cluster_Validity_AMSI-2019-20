import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import k_means

target = pd.read_csv("iris.csv", delimiter=',', usecols=[4], dtype=str, header=None).to_numpy().flatten()

target_as_ints = np.zeros(len(target),dtype=int)

target_names = np.unique(target)

for i,l in enumerate(target_names):
    target_as_ints[target == l] = i

cols = pd.read_csv('iris.csv',nrows=1,header=None).columns
inputs = np.genfromtxt(fname='iris.csv', delimiter=',', dtype=float, usecols=cols[:-1])

scaler = StandardScaler()
scaled_inputs = scaler.fit_transform(inputs)

centroids, labels, inertia = k_means(inputs,n_clusters=4, init='k-means++')











