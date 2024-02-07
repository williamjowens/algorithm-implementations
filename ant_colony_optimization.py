import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the distance function
def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Define the Ant Colony Optimization function
def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    # Iterating over the number of iterations
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        # Looping over each ant
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0

            # Loop until all points are visited
            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                # Calculate probabilities for moving to next point
                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = (pheromone[current_point, unvisited_point]**alpha *
                                        (1 / distance(points[current_point], points[unvisited_point]))**beta)

                probabilities /= np.sum(probabilities)

                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point

            paths.append(path)
            path_lengths.append(path_length)

            # Update the best path if a shorter path is found
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        # Evaporation of pheromones
        pheromone *= evaporation_rate

        # Pheromone update for the paths taken by all ants
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points-1):
                pheromone[path[i], path[i+1]] += Q/path_length
            pheromone[path[-1], path[0]] += Q/path_length

    return best_path, best_path_length

# Example usage: Solve TSP for 10 random cities in a 3D space
np.random.seed(42)
points = np.random.rand(10, 3) # Generate 10 random 3D points

# Parameters for the ACO algorithm
n_ants = 10
n_iterations = 100
alpha = 1.0
beta = 1.0
evaporation_rate = 0.5
Q = 1.0

# Run the ACO algorithm
best_path, best_path_length = ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q)

# Output the results
print(f"Best Path: {best_path}")
print(f"Best Path Length: {best_path_length}")

# Visualize the best path in a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:,0], points[:,1], points[:,2], c="r", marker="o")

# Plot the best path
for i in range(len(best_path)-1):
    ax.plot([points[best_path[i],0], points[best_path[i+1],0]],
            [points[best_path[i],1], points[best_path[i+1],1]],
            [points[best_path[i],2], points[best_path[i+1],2]],
            c="g", linestyle="-", linewidth=2, marker="o")

# Plot the path from the last city to the first city
ax.plot([points[best_path[0],0], points[best_path[-1],0]],
        [points[best_path[0],1], points[best_path[-1],1]],
        [points[best_path[0],2], points[best_path[-1],2]],
        c="g", linestyle="-", linewidth=2, marker="o")

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

plt.show()