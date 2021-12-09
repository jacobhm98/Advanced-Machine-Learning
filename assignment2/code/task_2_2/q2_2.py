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
import itertools
from Tree import Tree
from Tree import Node
import copy

K = 5

DYNAMICU_PROGRAMMINGGU = {}


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
    _, leaves = partition_leaves(beta)
    likelihood = 0
    DYNAMICU_PROGRAMMINGGU.clear()
    for i in range(K):
        likelihood += s(0, i, theta, tree_topology, leaves, beta) * theta[0][i]
    return likelihood

# returns the probability of all observations underneath node, given that node takes on value
def s(node, i, theta, tree_topology, leaves, beta):
    key = (node, i)

    if key in DYNAMICU_PROGRAMMINGGU:
        return DYNAMICU_PROGRAMMINGGU[key]

    left_child, right_child = get_children(node, tree_topology)
    left_recursion = 0
    right_recursion = 0
    #base case
    if left_child in leaves:
        left_recursion = theta[left_child][i][int(beta[left_child])]
    else:
        for j in range(K):
            left_recursion += s(left_child, j, theta, tree_topology, leaves, beta) * theta[left_child][i][j]

    if right_child in leaves:
        right_recursion = theta[right_child][i][int(beta[right_child])]
    else:
        for k in range(K):
            right_recursion += s(right_child, k, theta, tree_topology, leaves, beta) * theta[right_child][i][k]
    return_val = left_recursion * right_recursion
    DYNAMICU_PROGRAMMINGGU[key] = return_val
    return left_recursion * right_recursion


def get_children(node, tree_topology):
    children = []
    for index, element in enumerate(tree_topology[node::]):
        if element == int(node):
            children.append(index + node)
    return children


def partition_leaves(beta):
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
    #t.create_random_binary_tree(seed_val=0, k=2, num_nodes=4)
    t.load_tree(filename)
    t.print()
    t.sample_tree(num_samples=10000, seed_val=0)
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




def estimate_likelihood_from_samples(tree, beta):
    count = 0
    filtered_samples = tree.filtered_samples
    for sample in filtered_samples:
        if np.array_equal(sample, beta, equal_nan=True):
            count += 1
    return count / tree.num_samples


if __name__ == "__main__":
    main()
