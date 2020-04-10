from sklearn import decomposition
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()



X, y = load_iris(return_X_y=True)

X = scale(X)

pca = decomposition.PCA(n_components=2).fit(X)
X_reduced = pca.transform(X)

plt.scatter(X_reduced[:,0], X_reduced[:,1])
plt.show()

k_test_list = np.arange(1, 21)

models = [GaussianMixture(k,'full',
                          random_state=42).fit(X) for k in k_test_list]

plt.plot(k_test_list, [m.bic(X) for m in models], label='BIC')
plt.plot(k_test_list, [m.aic(X) for m in models], label="AIC")
plt.legend(loc='best')
plt.xlabel('k value')
plt.xticks(np.arange(1,22,2))
plt.show()

best_model = models[np.where(k_test_list == 3)[0][0]]

best_model.fit(X)

p_labels = best_model.predict(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=p_labels.astype(float), cmap='viridis');
plt.show()