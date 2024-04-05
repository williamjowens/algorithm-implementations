import numpy as np
from typing import Callable, Tuple

class AntLionOptimizer:
    def __init__(self, obj_func: Callable, lb: list, ub: list, ant_count: int, max_iter: int,
                 elite_ratio: float = 0.1, C_a: float = 1.0, C_r: float = 0.5, seed: int = None):
        self.obj_func = obj_func  # Objective function
        self.lb = np.array(lb)  # Lower bounds
        self.ub = np.array(ub)  # Upper bounds
        self.ant_count = ant_count  # Number of ants
        self.max_iter = max_iter  # Maximum iterations
        self.elite_ratio = elite_ratio  # Ratio of elite ants
        self.C_a = C_a  # Constant for antlion random walk
        self.C_r = C_r  # Constant for ant random walk
        self.dim = len(lb)  # Dimension of the problem
        self.seed = seed  # Random seed for reproducibility
        self.best_solution = None  # Best solution found
        self.best_fitness = np.inf  # Best fitness value

    def initialize_ants(self) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)
        ants = np.random.uniform(self.lb, self.ub, (self.ant_count, self.dim))
        print(f"Initialized ants: shape = {ants.shape}")
        return ants

    def initialize_antlions(self) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)
        antlions = np.random.uniform(self.lb, self.ub, (self.ant_count, self.dim))
        print(f"Initialized antlions: shape = {antlions.shape}")
        return antlions

    def ant_random_walk(self, ant: np.ndarray) -> np.ndarray:
        r_a = np.random.rand(self.dim)
        r_b = np.random.rand(self.dim)
        new_ant = ant + (r_a - 0.5) * (self.ub - self.lb) * self.C_r
        return new_ant

    def antlion_random_walk(self, antlion: np.ndarray, ant: np.ndarray, elite: np.ndarray) -> np.ndarray:
        r_a = np.random.rand(self.dim)
        r_b = np.random.rand(self.dim)
        new_antlion = antlion + r_a * (elite - antlion) + r_b * (ant - antlion)
        return new_antlion

    def boundary_check(self, solutions: np.ndarray) -> np.ndarray:
        # Clamp the solutions within the specified bounds
        print(f"Boundary check: input shape = {solutions.shape}")
        clipped_solutions = np.clip(solutions, self.lb, self.ub)
        print(f"Boundary check: output shape = {clipped_solutions.shape}")
        return clipped_solutions

    def optimize(self) -> Tuple[np.ndarray, float]:
        ants = self.initialize_ants()
        antlions = self.initialize_antlions()

        ant_fitness = np.array([self.obj_func(ant) for ant in ants])
        print(f"Ant fitness: shape = {ant_fitness.shape}")

        antlion_fitness = np.array([self.obj_func(antlion) for antlion in antlions])
        print(f"Antlion fitness: shape = {antlion_fitness.shape}")

        elite_count = max(1, int(self.elite_ratio * self.ant_count))
        elite_indices = np.argsort(ant_fitness)[:elite_count]
        elite_ant = ants[elite_indices[0]]
        print(f"Elite ant: shape = {elite_ant.shape}")

        self.best_solution = elite_ant
        self.best_fitness = ant_fitness[elite_indices[0]]

        iteration = 0
        while iteration < self.max_iter:
            print(f"\nIteration: {iteration}")

            # Ant random walk
            ants = np.apply_along_axis(self.ant_random_walk, 1, ants)
            print(f"Ants after random walk: shape = {ants.shape}")

            # Antlion random walk
            for i in range(self.ant_count):
                antlions[i] = self.antlion_random_walk(antlions[i], ants[i], elite_ant)
            print(f"Antlions after random walk: shape = {antlions.shape}")

            # Boundary check
            ants = self.boundary_check(ants)
            print(f"Ants after boundary check: shape = {ants.shape}")

            antlions = self.boundary_check(antlions)
            print(f"Antlions after boundary check: shape = {antlions.shape}")

            # Update ant fitness
            ant_fitness = np.array([self.obj_func(ant) for ant in ants])
            print(f"Updated ant fitness: shape = {ant_fitness.shape}")

            # Update antlion fitness
            antlion_fitness = np.array([self.obj_func(antlion) for antlion in antlions])
            print(f"Updated antlion fitness: shape = {antlion_fitness.shape}")

            # Antlion trapping and update
            for i in range(self.ant_count):
                if antlion_fitness[i] > ant_fitness[i]:
                    antlions[i] = ants[i]
                    antlion_fitness[i] = ant_fitness[i]

            print(f"Antlions after update: shape = {antlions.shape}")
            print(f"Antlion fitness after update: shape = {antlion_fitness.shape}")

            # Update elite ant
            elite_indices = np.argsort(ant_fitness)[:elite_count]
            elite_ant = ants[elite_indices[0]]
            print(f"Updated elite ant: shape = {elite_ant.shape}")

            # Update best solution
            if ant_fitness[elite_indices[0]] < self.best_fitness:
                self.best_solution = elite_ant
                self.best_fitness = ant_fitness[elite_indices[0]]

            # Update iteration
            iteration += 1

        return self.best_solution, self.best_fitness

# Example objective function (Sphere function)
def sphere_function(x: np.ndarray) -> float:
    return np.sum(x**2)

# Main function
def main():
    # Define problem parameters
    obj_func = sphere_function
    lb = [-10] * 10  # Lower bounds
    ub = [10] * 10  # Upper bounds
    ant_count = 50  # Number of ants
    max_iter = 100  # Maximum iterations
    seed = 42  # Random seed for reproducibility

    # Create an instance of the AntLionOptimizer
    optimizer = AntLionOptimizer(obj_func, lb, ub, ant_count, max_iter, seed=seed)

    # Optimize the objective function
    best_solution, best_fitness = optimizer.optimize()

    # Print the results
    print(f"\nBest solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

if __name__ == "__main__":
    main()