import random
import math
import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, problem_dim, bounds, population_size, mutation_rate, crossover_rate,
                 tournament_size, extinction_interval, extinction_ratio, extinction_mutation_rate,
                 extinction_tournament_size):
        """
        Initialize the GeneticAlgorithm class with the given parameters.
        
        Args:
            problem_dim (int): The dimension of the problem (number of variables).
            bounds (list): The lower and upper bounds for each variable.
            population_size (int): The size of the population.
            mutation_rate (float): The probability of mutation.
            crossover_rate (float): The probability of crossover.
            tournament_size (int): The size of the tournament for selection.
            extinction_interval (int): The number of generations between mass extinction events.
            extinction_ratio (float): The ratio of the population to be eliminated during mass extinction.
            extinction_mutation_rate (float): The mutation rate after mass extinction.
            extinction_tournament_size (int): The tournament size after mass extinction.
        """
        self.problem_dim = problem_dim
        self.bounds = bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.extinction_interval = extinction_interval
        self.extinction_ratio = extinction_ratio
        self.extinction_mutation_rate = extinction_mutation_rate
        self.extinction_tournament_size = extinction_tournament_size
        
    def _initialize_population(self):
        """ 
        Initialize the population with random individuals within the specified bounds.
        
        Returns:
            list: The initialized population.
        """
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(low, high) for low, high in self.bounds]
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual):
        """ 
        Evaluate the fitness of an individual.
        
        Args:
            individual (list): The individual to evaluate.
            
        Returns:
            float: The fitness value of the individual.
        """
        x, y = individual
        distance = math.sqrt(x**2 + y**2)
        return 1 - distance
    
    def _tournament_selection(self, population, tournament_size):
        """ 
        Perform tournament selection to select a parent from the population.
        
        Args:
            population (list): The population to select from.
            tournament_size (int): The size of the tournament.
            
        Returns:
            list: The selected parent.
        """
        tournament = random.sample(population, tournament_size)
        best_individual = max(tournament, key=self._evaluate_fitness)
        return best_individual

    def _crossover(self, parent1, parent2):
        """ 
        Perform crossover between two parents to create a child.
        
        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.
            
        Returns:
            list: The child created by crossover.
        """
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.problem_dim - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
        else:
            child = parent1[:]
        return child
    
    def _mutate(self, individual, mutation_rate):
        """ 
        Perform mutation on an individual.
        
        Args:
            individual (list): The individual to mutate.
            mutation_rate (float): The probability of mutation.
            
        Returns:
            list: The mutated individual.
        """
        for i in range(self.problem_dim):
            if random.random() < mutation_rate:
                individual[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
        return individual
        
    def _mass_extinction(self, population):
        """ 
        Perform mass extinction on the population.
        
        Args:
            population (list): The population to perform mass extinction on.
            
        Returns:
            list: The survivors of the mass extinction.
        """
        sorted_population = sorted(population, key=self._evaluate_fitness, reverse=True)
        extinction_size = int(len(population) * self.extinction_ratio)
        survivors = sorted_population[:extinction_size]
        return survivors
    
    def run(self, max_generations, fitness_threshold):
        """ 
        Run the genetic algorithm.
        
        Args:
            max_generations (int): The maximum number of generations to run.
            fitness_threshold (float): The fitness threshold to stop the algorithm.
            
        Returns:
            tuple: A tuple containing the best individual, its fitness value, the list of best fitnesses,
                   and the list of best solutions over generations.
        """
        population = self._initialize_population()
        generation = 0
        best_fitnesses = []
        best_solutions = []
        
        while generation < max_generations:
            generation += 1
            
            # Evaluate fitness of individuals
            fitness_values = [self._evaluate_fitness(individual) for individual in population]
            best_fitness = max(fitness_values)
            best_individual = population[fitness_values.index(best_fitness)]
            best_fitnesses.append(best_fitness)
            best_solutions.append(best_individual)
            
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            
            # Check termination condition
            if best_fitness >= fitness_threshold:
                break
            
            # Perform mass extinction
            if generation % self.extinction_interval == 0:
                population = self._mass_extinction(population)
                mutation_rate = self.extinction_mutation_rate
                tournament_size = self.extinction_tournament_size
            else:
                mutation_rate = self.mutation_rate
                tournament_size = self.tournament_size
                
            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, tournament_size)
                parent2 = self._tournament_selection(population, tournament_size)
                child = self._crossover(parent1, parent2)
                mutated_child = self._mutate(child, mutation_rate)
                new_population.append(mutated_child)
                
            population = new_population
            
        return best_individual, best_fitness, best_fitnesses, best_solutions
    
def visualize_convergence(best_fitnesses):
    """ 
    Visualize the convergence of the genetic algorithm.
    
    Args:
        best_fitnesses (list): The list of best fitness values over generations.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(best_fitnesses)), best_fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Convergence Plot")
    plt.show()
    
def visualize_solution_space(bounds, best_solutions):
    """ 
    Visualize the solution space and the trajectory of the best solutions found by the genetic algorithm.
    
    Args:
        bounds (list): The lower and upper bounds for each variable.
        best_solutions (list): The list of best solutions found over generations.
    """
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Distance from Origin')
    
    # Plot the trajectory of the best solutions
    best_solutions_x = [solution[0] for solution in best_solutions]
    best_solutions_y = [solution[1] for solution in best_solutions]
    plt.plot(best_solutions_x, best_solutions_y, 'r-', linewidth=1, label='Best Solution Trajectory')
    plt.plot(best_solutions_x[-1], best_solutions_y[-1], 'ro', markersize=10, label='Final Best Solution')
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Solution Space")
    plt.legend()
    plt.show()
    
def main():
    """ 
    Run the genetic algorithm and visualize the results.
    """
    problem_dim = 2
    bounds = [(-5, 5), (-5, 5)]
    population_size = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    tournament_size = 5
    extinction_interval = 50
    extinction_ratio = 0.8
    extinction_mutation_rate = 0.2
    extinction_tournament_size = 10
    max_generations = 100
    fitness_threshold = 0.99
    
    ga = GeneticAlgorithm(problem_dim, bounds, population_size, mutation_rate, crossover_rate,
                          tournament_size, extinction_interval, extinction_ratio,
                          extinction_mutation_rate, extinction_tournament_size)
    
    best_solution, best_fitness, best_fitnesses, best_solutions = ga.run(max_generations, fitness_threshold)
    
    print(f"\nBest Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")
    
    visualize_convergence(best_fitnesses)
    visualize_solution_space(bounds, best_solutions)
    
if __name__ == "__main__":
    main()