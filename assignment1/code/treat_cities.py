import requests
import pandas
import numpy as np
import csv

base_url = "https://se.avstand.org/route.json?stops="

def main():
    cities = read_in_cities("cities.csv")
    with open('treated_cities.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(cities)
    print(cities)

def call_api(city1, city2):
    response = requests.request("GET", base_url + str(city1) + "|" + str(city2))
    data = response.json()
    try:
        if len(data["distances"]) is not 0:
            return 0
        if data["stops"][0]["type"] == "Invalid":
            return -1
    except:
        print(city1, city2)
    return 0

def read_in_cities(file):
    cities = pandas.read_csv(file)
    cities = cities['capital'].tolist()
    treated_cities = []
    for i in range(len(cities)):
        if i == len(cities) - 1:
            result = call_api(cities[i], cities[i - 1])
        else:
            result = call_api(cities[i], cities[i + 1])
        if result != -1:
            treated_cities.append(cities[i])
    return treated_cities


main()