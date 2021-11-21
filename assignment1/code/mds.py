import numpy as np
import get_distance_matrix


def main():
    D = np.array([[0, 93, 82, 133], [93, 0, 52, 60], [82, 52, 0, 111], [133, 60, 111, 0]])
    D = get_distance_matrix.get_distance_matrix()
    classical_mds(D)
    metric_mds()

def get_gram_matrix(D):
    N = len(D)
    D_squared = np.square(D)
    # Double centering
    J = np.identity(N) - 1/N * np.matmul(np.ones(N), np.transpose(np.ones(N)))
    S = -0.5 * np.matmul(np.matmul(J, D_squared), J)
   # S = -0.5 * (D -
   #     1 / N * np.matmul(np.matmul(D_squared, np.ones(N)), np.transpose(np.ones(N)))
   #     - 1 / N * np.matmul(np.matmul(np.ones(N), np.transpose(np.ones(N))), D_squared)
   #     + 1 / (N ** 2) * np.matmul(np.matmul(np.matmul(np.ones(N), np.transpose(np.ones(N))), D_squared), np.matmul(np.ones(N), np.transpose(np.ones(N)))))
    return S


def classical_mds(D):
    S = get_gram_matrix(D)
    eigenvec, eigenval = np.linalg.eigh(S)
    eigenvec = eigenvec[:1]
    eigenval = eigenval[:1, :1]
    eigenval = np.sqrt(eigenval)
    X = np.matmul(eigenvec, eigenval)
    D_hat = np.empty(len(D), 2)
    for i in range(len(D)):
        D_hat[i] = np.matmul(X, D[i])

main()