import numpy as np
import random
from scipy.stats import multivariate_normal as normal_distribution
from scipy.stats import poisson as poisson_distribution
import matplotlib.pyplot as plt

K = 3


def main():
    X, S = read_in_data('X.txt', 'S.txt')
    estimate_parameters(X, S)


def estimate_parameters(X_data, S_data):
    pi, normals, poissons = initialize_models()
    visualize_data_and_models(X_data, S_data, normals, poissons)
    log_likelihood = calculate_log_likelihood(X_data, S_data, pi, normals, poissons)
    next_likelihood = 0
    while next_likelihood > log_likelihood:
        if next_likelihood != 0:
            log_likelihood = next_likelihood
        gammas = calculate_gammas(X_data, S_data, pi, normals, poissons)
        pi, normals, poissons = re_estimate_parameters(X_data, S_data, gammas)
        next_likelihood = calculate_log_likelihood(X_data, S_data, pi, normals, poissons)
        visualize_data_and_models(X_data, S_data, normals, poissons)
    return pi, normals, poissons

def visualize_data_and_models(X_data, S_data, normals, poissons):
    x, y = np.mgrid[-20:20, -20:20]
    pos = np.dstack((x, y))
    Z = normals[0].pdf(pos)
    plt.contour(x, y, Z, col='red', linewidths=poissons[0].mean(), alpha=0.1)
    plt.scatter(X_data[:, 0], X_data[:, 1], s=S_data)
    plt.show()





def calculate_log_likelihood(X_data, S_data, pi, normals, poissons):
    N = len(X_data)
    log_likelihood = 0
    for n in range(N):
        nth_addition = 0
        for k in range(K):
            nth_addition += pi[k] * normals[k].pdf(X_data[n]) * poissons[k].pmf(S_data[n])
        nth_addition = np.log(nth_addition)
        log_likelihood += nth_addition
    return log_likelihood


def re_estimate_parameters(X_data, S_data, gammas):
    N = len(X_data)
    mus = np.zeros((K, 2))
    sigmas = np.zeros((K, 2, 2))
    poisson_params = np.zeros(K)
    pi = np.zeros(K)
    normals = []
    poissons = []
    for k in range(K):
        for n in range(N):
            # store values of n_k into pi_k
            pi[k] += gammas[k][n]
            mus[k] += gammas[k][n] * X_data[n]
            poisson_params[k] += gammas[k][n] * S_data[n]
        mus[k] /= pi[k]
        poisson_params /= pi[k]
        for n in range(N):
            #numpy fuckry needed in order to transpose
            temp = np.zeros((1, 2))
            temp[0] = X_data[n] - mus[k]
            sigmas[k] += gammas[k][n] * np.matmul(np.transpose(temp), temp)
        sigmas[k] /= pi[k]
        normal = normal_distribution(mean=mus[k], cov=sigmas[k])
        normals.append(normal)
        poisson = poisson_distribution(poisson_params[k])
        poissons.append(poisson)
        pi[k] /= N
    return pi, normals, poissons


def calculate_gammas(X_data, S_data, pi, normals, poissons):
    N = len(X_data)
    gammas = np.zeros((K, N))
    for k in range(K):
        for n in range(N):
            nominator = pi[k] * normals[k].pdf(X_data[n]) * poissons[k].pmf(S_data[n])
            denominator = 0
            for j in range(K):
                denominator += pi[j] * normals[j].pdf(X_data[n]) * poissons[j].pmf(S_data[n])
            gammas[k][n] = nominator / denominator
    return gammas


def initialize_models():
    pi = np.zeros(K)
    normals = []
    poissons = []
    for k in range(K):
        pi[k] = random.uniform(0, 1)
        # initialize poisson distributions
        poisson_param = random.uniform(0, 10)
        poisson = poisson_distribution(poisson_param)
        poissons.append(poisson)
        # initialize normal distributions
        mu = np.zeros(2)
        sigma = np.zeros((2, 2))
        for i in range(2):
            mu[i] = random.uniform(-10, 10)
            sigma[i][i] = random.uniform(1, 10)
        normal = normal_distribution(mean=mu, cov=sigma)
        normals.append(normal)
        # enforce probabilities adding up to 1
        sum = np.sum(pi)
        pi /= sum
    return pi, normals, poissons


def read_in_data(X_file, S_file):
    X = []
    S = []
    with open(X_file) as f:
        for data_point in f:
            data_point = data_point.rstrip()
            coord1, coord2 = data_point.split(sep=' ')
            X.append([float(coord1), float(coord2)])
    with open(S_file) as f:
        for data_point in f:
            data_point = data_point.rstrip()
            S.append(float(data_point))
    assert len(X) == len(S)
    return np.array(X), np.array(S)


main()
