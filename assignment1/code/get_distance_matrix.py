import requests
import numpy as np
import csv


base_url = "https://se.avstand.org/route.json?stops="
CREATE_DISTANCE_MATRIX = True

def get_distance_matrix(calculate_distance_matrix=False):
    cities = read_in_cities("treated_cities.csv")
    if calculate_distance_matrix is True:
        D = create_distance_matrix(cities)
    else:
        D = np.load("distance_matrix.npy")
    return cities, D


def get_distance(city1, city2):
    response = requests.request("GET", base_url + str(city1) + "|" + str(city2))
    data = response.json()
    distance = data["distances"][0]
    return distance

def create_distance_matrix(cities):
    N = len(cities)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            if i == j:
                continue
            else:
                distance = get_distance(cities[i], cities[j])
                D[i][j] = distance
                D[j][i] = distance
    np.save("distance_matrix", D)
    return D

def read_in_cities(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data[0]
