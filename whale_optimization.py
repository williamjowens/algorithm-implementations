import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class WhaleOptimizationAlgorithm:
    def __init__(self, objective_func, dim, search_space, max_iter, population_size):
        self.objective_func = objective_func
        self.dim = dim
        self.search_space = search_space
        self.max_iter = max_iter
        self.population_size = population_size

    def _initialize_population(self):
        """
        Initialize the population of whales randomly within the search space.
        """
        population = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.dim):
                population[i, j] = random.uniform(self.search_space[j][0], self.search_space[j][1])
        return population

    def _calculate_fitness(self, population):
        """
        Calculate the fitness values for each whale in the population.
        """
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness[i] = self.objective_func(population[i, :])
        return fitness

    def _update_position(self, position, best_position, a, c, l, p):
        """
        Update the position of a whale based on the WOA equations.
        """
        r1 = random.random()
        r2 = random.random()
        A = 2 * a * r1 - a
        C = 2 * r2
        b = 1
        if p < 0.5:
            if abs(A) < 1:
                D = abs(C * best_position - position)
                position_new = best_position - A * D
            else:
                rand_leader_index = math.floor(self.population_size * random.random())
                X_rand = self.population[rand_leader_index, :]
                D = abs(C * X_rand - position)
                position_new = X_rand - A * D
        else:
            D = abs(best_position - position)
            position_new = D * math.exp(b * l) * math.cos(2 * math.pi * l) + best_position
        return position_new

    def _check_bounds(self, position):
        """
        Check if the position is within the search space boundaries and adjust it if necessary.
        """
        for j in range(self.dim):
            if position[j] < self.search_space[j][0]:
                position[j] = self.search_space[j][0]
            if position[j] > self.search_space[j][1]:
                position[j] = self.search_space[j][1]
        return position

    def optimize(self):
        """
        Perform the optimization using the Whale Optimization Algorithm.
        """
        self.population = self._initialize_population()
        fitness = self._calculate_fitness(self.population)
        best_index = np.argmin(fitness)
        best_position = self.population[best_index, :].copy()
        best_fitness = fitness[best_index]

        convergence_curve = np.zeros(self.max_iter)
        trajectory = []

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)
            for i in range(self.population_size):
                p = random.random()
                l = (a - 1) * random.random() + 1
                self.population[i, :] = self._update_position(self.population[i, :], best_position, a, 2, l, p)
                self.population[i, :] = self._check_bounds(self.population[i, :])

            fitness = self._calculate_fitness(self.population)
            best_index = np.argmin(fitness)
            if fitness[best_index] < best_fitness:
                best_position = self.population[best_index, :].copy()
                best_fitness = fitness[best_index]

            convergence_curve[t] = best_fitness
            trajectory.append(best_position)

        return best_position, best_fitness, convergence_curve, trajectory

def sphere(x):
    """
    Sphere function.
    """
    return np.sum(x**2)

def rastrigin(x):
    """
    Rastrigin function.
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * math.pi * x), axis=0)

def create_surface(func, search_space):
    """
    Create the surface plot data for the given objective function.
    """
    x = np.linspace(search_space[0][0], search_space[0][1], 100)
    y = np.linspace(search_space[1][0], search_space[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    return X, Y, Z

def plot_convergence(convergence_curve):
    """
    Plot the convergence curve of the best fitness value over iterations.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(convergence_curve, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Convergence Curve')
    plt.grid(True)
    plt.show()

def plot_solution_space(func, search_space, best_position, trajectory):
    """
    Plot the solution space with the best position and the solution trajectory.
    """
    X, Y, Z = create_surface(func, search_space)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
    ax.scatter(best_position[0], best_position[1], func(best_position), color='red', marker='*', s=100)
    
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], func(trajectory.T), color='black', linewidth=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Solution Space')
    plt.show()

def main():
    """
    Run the optimization and plot the results.
    """
    dim = 2
    search_space = [(-5.12, 5.12) for _ in range(dim)]
    max_iter = 100
    population_size = 50

    objective_funcs = [sphere, rastrigin]
    func_names = ['Sphere', 'Rastrigin']

    for i, func in enumerate(objective_funcs):
        print(f"Optimizing {func_names[i]} function...")
        woa = WhaleOptimizationAlgorithm(func, dim, search_space, max_iter, population_size)
        best_position, best_fitness, convergence_curve, trajectory = woa.optimize()

        print(f"Best position: {best_position}")
        print(f"Best fitness: {best_fitness}")

        plot_convergence(convergence_curve)
        plot_solution_space(func, search_space, best_position, trajectory)

if __name__ == "__main__":
    main()