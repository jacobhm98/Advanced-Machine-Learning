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
    for i in range(K):
        likelihood += s(0, i, theta, tree_topology, leaves, beta) * theta[0][i]
    return likelihood

# returns the probability of all observations underneath node, given that node takes on value
def s(node, i, theta, tree_topology, leaves, beta):
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
    return left_recursion * right_recursion





def partition_factors(element, factor_list):
    factors_containing_element = []
    factors_not_containing_element = []
    for factor in factor_list:
        if element in factor[0]:
            factors_containing_element.append(factor)
        else:
            factors_not_containing_element.append(factor)
    return factors_containing_element, factors_not_containing_element


def eliminate(element, factors):
    # factors_containing_element = get_factors_containing_element(element, factors)
    combined_factor = combine_factors(factors, element)
    return marginalize(combined_factor, element)

def marginalize(factor, variable):
    dimension = factor[0].index(variable)
    factor[1] = np.sum(factor[1], axis=dimension)
    del factor[0][dimension]
    return factor

def combine_factors(factors, common_variable):
    factor = factors.pop()
    move_variable_to_first_dimension(common_variable, factor)
    for factor_to_combine in factors:
        move_variable_to_first_dimension(common_variable, factor_to_combine)
        factor = factor_product(factor, factor_to_combine)
    return factor


def factor_product(f1, f2):
    assert f1[0][0] == f2[0][0]
    combined_CPT = []
    CPT1 = f1[1]
    CPT2 = f2[1]
    variable_dimension = f1[0]
    for variable in f2[0][1::]:
        variable_dimension.append(variable)
    desired_shape = [K] * (len(variable_dimension) - 1)
    for i in range(len(CPT1)):
        permutations = [x * y for x, y in itertools.product(CPT1[i].flatten(), CPT2[i].flatten())]
        permutations = np.reshape(permutations, desired_shape)
        combined_CPT.append(permutations)
    factor = [variable_dimension, np.array(combined_CPT)]
    return factor


def move_variable_to_first_dimension(variable, factor):
    dimension_of_variable = factor[0].index(variable)
    factor = swap_dimensions(factor, 0, dimension_of_variable)


def swap_dimensions(factor, dim1, dim2):
    indices_of_variables = factor[0]
    CPT = factor[1]
    temp = indices_of_variables[dim1]
    indices_of_variables[dim1] = indices_of_variables[dim2]
    indices_of_variables[dim2] = temp
    CPT = np.swapaxes(CPT, dim1, dim2)
    return [indices_of_variables, CPT]


def create_elimination_ordering(tree_topology):
    return None


def create_factor_list(theta, tree_topology):
    factor_list = []
    theta[1] = multiply_in_singular_factor(theta[0], theta[1])
    for element, parent in enumerate(tree_topology[1::]):
        factor = [[int(parent), element + 1]]
        factor.append(np.array(theta[element + 1]))
        factor_list.append(factor)
    return factor_list

def multiply_in_singular_factor(singular, second_factor):
    for i in range(K):
        for j in range(K):
            second_factor[i][j] = second_factor[i][j] * singular[i]
    return second_factor


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
        # brute_marginalization(t.get_topology_array(), t.get_theta_array(), beta)




def estimate_likelihood_from_samples(tree, beta):
    count = 0
    filtered_samples = tree.filtered_samples
    for sample in filtered_samples:
        if np.array_equal(sample, beta, equal_nan=True):
            count += 1
    return count / tree.num_samples


if __name__ == "__main__":
    main()
