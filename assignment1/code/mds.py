import numpy as np
import get_distance_matrix as d


def main():
    D = np.array([[0, 93, 82, 133], [93, 0, 52, 60], [82, 52, 0, 111], [133, 60, 111, 0]])
    # D = d.get_distance_matrix()
    embedding = classical_mds(D)
    # metric_mds()


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
    #flip values since eigh returns in ascending order
    eigenval = eigenval[::-1]
    eigenvec = np.fliplr(eigenvec)
    eigenval = np.sqrt(eigenval)
    #choose two dimensions
    eigenval = eigenval[:2]
    eigenvec = eigenvec[:, :2]
    #do matrix multiplication to get our points
    eigenval = np.diag(eigenval)
    X = np.matmul(eigenvec, eigenval)
    return X


main()
