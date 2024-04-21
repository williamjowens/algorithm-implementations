import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MBO:
    def __init__(self, objective_func, dim, pop_size, max_iter, lb, ub, share_factor=0.1, swap_interval=10):
        """
        Initialize the Migrating Birds Optimization (MBO) algorithm.
        
        Args:
            objective_func (function): The objective function to be minimized.
            dim (int): The dimensionality of the problem.
            pop_size (int): The population size of the birds.
            max_iter (int): The maximum number of iterations.
            lb (array-like): The lower bounds of the search space.
            ub (array-like): The upper bounds of the search space.
            share_factor (float): The sharing factor for information sharing.
            swap_interval (int): The interval for swapping the leader bird.
        """
        self.objective_func = objective_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.share_factor = share_factor
        self.swap_interval = swap_interval
        self.population = None
        self.fitness = None
        self.leader_idx = None
        self.best_solution = None
        self.best_fitness = None
        self.convergence_curve = None

    def _initialize_population(self):
        """Initialize the population of birds randomly within the search space."""
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_func(self.population[i])
        self.leader_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.leader_idx].copy()
        self.best_fitness = self.fitness[self.leader_idx]

    def _update_positions(self):
        """Update the positions of the birds based on the leader and random birds."""
        for i in range(self.pop_size):
            if i != self.leader_idx:
                r1 = np.random.rand()
                r2 = np.random.rand()
                random_bird_idx = np.random.randint(0, self.pop_size)
                while random_bird_idx == i:
                    random_bird_idx = np.random.randint(0, self.pop_size)
                new_position = self.population[i] + r1 * (self.population[self.leader_idx] - self.population[i]) + \
                               r2 * (self.population[random_bird_idx] - self.population[i])
                new_position = np.clip(new_position, self.lb, self.ub)
                new_fitness = self.objective_func(new_position)
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_position
                    self.fitness[i] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_position.copy()
                        self.best_fitness = new_fitness

    def _swap_leader(self):
        """Swap the leader bird with a randomly selected follower bird."""
        follower_idx = np.random.randint(0, self.pop_size)
        while follower_idx == self.leader_idx:
            follower_idx = np.random.randint(0, self.pop_size)
        self.population[[self.leader_idx, follower_idx]] = self.population[[follower_idx, self.leader_idx]]
        self.fitness[[self.leader_idx, follower_idx]] = self.fitness[[follower_idx, self.leader_idx]]
        self.leader_idx = np.argmin(self.fitness)

    def _share_information(self):
        """Share information among a subset of birds to enhance convergence."""
        shared_birds = int(self.pop_size * self.share_factor)
        shared_indices = np.random.choice(self.pop_size, shared_birds, replace=False)
        for i in range(shared_birds):
            r = np.random.rand()
            new_position = self.population[shared_indices[i]] + r * (self.best_solution - self.population[shared_indices[i]])
            new_position = np.clip(new_position, self.lb, self.ub)
            new_fitness = self.objective_func(new_position)
            if new_fitness < self.fitness[shared_indices[i]]:
                self.population[shared_indices[i]] = new_position
                self.fitness[shared_indices[i]] = new_fitness
                if new_fitness < self.best_fitness:
                    self.best_solution = new_position.copy()
                    self.best_fitness = new_fitness

    def optimize(self):
        """Run the MBO optimization algorithm."""
        self._initialize_population()
        self.convergence_curve = np.zeros(self.max_iter)

        for i in range(self.max_iter):
            self._update_positions()
            if (i + 1) % self.swap_interval == 0:
                self._swap_leader()
            self._share_information()
            self.convergence_curve[i] = self.best_fitness

        return self.best_solution, self.best_fitness, self.convergence_curve

def rastrigin(x):
    """Rastrigin benchmark function."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    """Ackley benchmark function."""
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

def plot_search_space(func, lb, ub, best_solution):
    """Plot the search space and the best solution found by MBO."""
    if len(lb) == 2:
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

        plt.figure(figsize=(8, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Fitness')
        ax.set_title("Search Space")

        # Plot the best solution found by MBO
        ax.scatter(best_solution[0], best_solution[1], func(best_solution), color='red', marker='*', s=100)

        plt.show()

def plot_convergence_curve(convergence_curve):
    """Plot the convergence curve of the MBO algorithm."""
    plt.figure(figsize=(8, 6))
    plt.plot(convergence_curve)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Convergence Curve")
    plt.grid(True)
    plt.show()

def main():
    """Run the MBO algorithm on benchmark functions."""
    dim = 2
    pop_size = 50
    max_iter = 100
    lb = -5.12 * np.ones(dim)
    ub = 5.12 * np.ones(dim)

    objective_funcs = [rastrigin, ackley]
    func_names = ["Rastrigin", "Ackley"]

    for i, func in enumerate(objective_funcs):
        print(f"Optimizing {func_names[i]} function...")
        
        mbo = MBO(func, dim, pop_size, max_iter, lb, ub)
        best_solution, best_fitness, convergence_curve = mbo.optimize()

        print(f"Best solution: {best_solution}")
        print(f"Best fitness: {best_fitness}\n")

        plot_convergence_curve(convergence_curve)
        plot_search_space(func, lb, ub, best_solution)

if __name__ == "__main__":
    main()