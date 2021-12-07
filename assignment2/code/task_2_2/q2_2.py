""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates.
    However this is not a "must" and you can code however you like.
    You can write helper functions etc however you want.

    If you want, you can use the class structures provided to you (Node and Tree classes in Tree.py
    file), and modify them as needed. In addition to the data files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want.

    For this task, we gave you three different trees (q2_2_small_tree, q2_2_medium_tree, q2_2_large_tree).
    Each tree has 5 samples (the inner nodes' values are masked with np.nan).
    We want you to calculate the likelihoods of each given sample and report it.

    Note:   The alphabet "K" is K={0,1,2,3,4}.

    Note:   A VERY COMMON MISTAKE is to use incorrect order of nodes' values in CPDs.
            theta is a list of lists, whose shape is approximately (num_nodes, K, K).
            For instance, if node "v" has a parent "u", then p(v=Zv | u=Zu) = theta[v][Zu][Zv].

            If you ever doubt your useage of theta, you can double check this marginalization:
            \sum_{k=1}^K p(v = k | u=Zu) = 1
"""

import numpy as np
from Tree import Tree
from Tree import Node
import copy

K = 2

def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: list of numpy arrays. Dimensions (approximately): (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """
    latent_vertices, leaves = get_leave_nodes(beta)
    for element in latent_vertices:
        factors_containing_element = [element]
        child1, child2 = get_children(element, tree_topology)
        factors_containing_element.append(child1)
        factors_containing_element.append(child2)
        factor_product(element, child1, theta)
        factor_marginalize(factors_containing_element, theta)
    marginal_jpd = np.zeros((K, K, K))
    for i in range(K):
        for j in range(K):
            for k in range(K):
                marginal_jpd[i][j][k] = theta[2][i] * theta[3][j] * theta[4][k]
    print(marginal_jpd.sum())
    likelihood = 1.0
    for element in leaves:
        likelihood *= theta[element][int(beta[element])]
    return likelihood

def factor_marginalize(factors_to_be_marginalized, theta):
    factor1 = factors_to_be_marginalized[1]
    factor2 = factors_to_be_marginalized[2]
    f1 = np.zeros(K)
    f2 = np.zeros(K)
    for i in range(K):
        for j in range(K):
            f1[j] += theta[factor1][i][j]
            f2[j] += theta[factor2][i][j]
    theta[factor1] = f1
    theta[factor2] = f2



def factor_product(parent, child, theta):
    for i in range(K):
        for j in range(K):
            theta[child][i][j] = theta[child][i][j] * theta[parent][i]

def get_children(node, tree_topology):
    children = []
    for index, element in enumerate(tree_topology[node::]):
        if element == int(node):
            children.append(index + node)
    return children

def get_leave_nodes(beta):
    leaves = []
    latent_vertices = []
    for index, node in enumerate(beta):
        if np.isnan(node):
            latent_vertices.append(index)
        else:
            leaves.append(index)
    return latent_vertices, leaves

def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    filename = "data/q2_2_small_tree.pkl"  # "data/q2_2_medium_tree.pkl", "data/q2_2_large_tree.pkl"
    print("filename: ", filename)

    t = Tree()
    t.create_random_binary_tree(seed_val=0, k=2, num_nodes=4)
    #t.load_tree(filename)
    t.print()
    t.sample_tree(num_samples=100000)
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        theta = copy.deepcopy(t.get_theta_array())
        calculated_likelihood = calculate_likelihood(t.get_topology_array(), theta, beta)
        print("\tLikelihood: ", calculated_likelihood)
        sample_likelihood = estimate_likelihood_from_samples(t, beta)
        print("\tSample Likelihood: ", sample_likelihood)
        #brute_marginalization(t.get_topology_array(), t.get_theta_array(), beta)

def brute_marginalization(tree_topology, theta, beta):
    likelihood = 1
    latent_variables, leaves = get_leave_nodes(beta)
    for node in leaves:
        continue
    for node in latent_variables:
        continue



def estimate_likelihood_from_samples(tree, beta):
    count = 0
    filtered_samples = tree.filtered_samples
    for sample in filtered_samples:
        if np.array_equal(sample, beta, equal_nan=True):
            count += 1
    return count / tree.num_samples

if __name__ == "__main__":
    main()
