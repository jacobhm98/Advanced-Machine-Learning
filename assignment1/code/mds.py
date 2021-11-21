import numpy as np
import get_distance_matrix as d
import matplotlib.pyplot as plt
import sklearn.manifold as sk


def main():
    cities, D = d.get_distance_matrix()
    classical_embedding = classical_mds(D)
    metric_embedding = metric_mds(D)
    create_plot(cities, metric_embedding)


def create_plot(labels, data_points):
    x = list(data_points[:, 0])
    y = list(data_points[:, 1])
    fig, ax = plt.subplots()
    ax.scatter(data_points[:, 0], data_points[:, 1])
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]))
    plt.show()


def metric_mds(D):
    model = sk.MDS(n_components=2, dissimilarity='precomputed')
    embedding = model.fit_transform(D)
    return embedding


def get_gram_matrix(D):
    N = len(D)
    D_squared = np.square(D)
    # Double centering
    J = np.identity(N) - 1 / N * np.ones((N, N))
    S = -0.5 * np.matmul(np.matmul(J, D_squared), J)
    return S


def classical_mds(D):
    S = get_gram_matrix(D)
    eigenval, eigenvec = np.linalg.eigh(S)
    # flip values since eigh returns in ascending order
    eigenval = eigenval[::-1]
    eigenvec = np.fliplr(eigenvec)
    # choose two dimensions
    eigenval = eigenval[:2]
    eigenval = np.sqrt(eigenval)
    eigenvec = eigenvec[:, :2]
    # do matrix multiplication to get our points
    eigenval = np.diag(eigenval)
    X = np.matmul(eigenvec, eigenval)
    return X


main()
