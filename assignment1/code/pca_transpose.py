from sklearn.decomposition import PCA
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
X -= np.mean(X, axis=0)
Xt = np.matrix.transpose(X)

pca = PCA()
pca.fit(X)

# now our goal is to recreate these principal components using SVD of transposed data
V, Sigma_t, U_t = np.linalg.svd(Xt)
#change the params since we want to get transposed SVD
V_t = np.matrix.transpose(V)

for i in range(pca.singular_values_.size):
    print(pca.singular_values_[i], pca.components_[i])

for i in range(len(Sigma_t)):
    print(Sigma_t[i], V_t[i])
