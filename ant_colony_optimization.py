import numpy as np
import random

# Number of cities and ants
n_cities = 10
n_ants = 5

# Generate random coordinates for 10 cities
np.random.seed(0)
cities = np.random.rand(n_cities, 2) * 100

# Calculate distance matrix
def calculate_distance_matrix(cities):
    n = len(cities)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
    return distance_matrix

distance_matrix = calculate_distance_matrix(cities)

def initialize_pheromone_levels(n_cities, initial_pheromone):
    return np.full((n_cities, n_cities), initial_pheromone)

def update_pheromone_levels(pheromones, paths, evaporation_rate, Q):
    pheromones *= evaporation_rate  # Exponential decrease of pheromone due to evaporation

    # Adding pheromone based on the quality of the path
    for path in paths:
        for i in range(len(path) - 1):
            pheromones[path[i], path[i + 1]] += Q / path_length(path, distance_matrix)
    return pheromones

def path_length(path, distance_matrix):
    return sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))

def choose_next_city(pheromones, distance_matrix, current_city, unvisited):
    pheromones_to_unvisited = pheromones[current_city, unvisited]
    heuristic = 1 / (distance_matrix[current_city, unvisited] + 1e-10)  # Adding a small constant to avoid division by zero
    probabilities = pheromones_to_unvisited * heuristic
    probabilities /= probabilities.sum()
    return np.random.choice(unvisited, p=probabilities)

def construct_solution(pheromones, distance_matrix, n_cities):
    path = [random.randint(0, n_cities - 1)]
    unvisited = set(range(n_cities)) - {path[0]}
    while unvisited:
        next_city = choose_next_city(pheromones, distance_matrix, path[-1], np.array(list(unvisited)))
        path.append(next_city)
        unvisited.remove(next_city)
    return path

def ant_colony_optimization(distance_matrix, n_cities, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    initial_pheromone = 1 / (n_cities * n_cities)
    pheromones = initialize_pheromone_levels(n_cities, initial_pheromone)

    best_path = None
    best_path_length = np.inf

    for iteration in range(n_iterations):
        paths = [construct_solution(pheromones ** alpha * (1 / distance_matrix) ** beta, distance_matrix, n_cities) for _ in range(n_ants)]
        pheromones = update_pheromone_levels(pheromones, paths, evaporation_rate, Q)

        # Find the best path in this iteration
        for path in paths:
            length = path_length(path, distance_matrix)
            if length < best_path_length:
                best_path = path
                best_path_length = length

    return best_path, best_path_length

# Parameters for ACO
n_iterations = 100
alpha = 0.1
beta = 1
evaporation_rate = 0.5
Q = 100

# Run ACO
best_path, best_path_length = ant_colony_optimization(distance_matrix, n_cities, n_ants, n_iterations, alpha, beta, evaporation_rate, Q)
best_path, best_path_length, cities[best_path]

# Formatting the results
print(f"Best Path: {best_path}")
print(f"Total Distance: {best_path_length}")
print("Cities Visited in Order:")
for city in best_path:
    print(f"City {city}: Coordinates {cities[city]}")