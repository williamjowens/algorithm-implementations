import numpy as np 

# Objective function
def objective_function(x):
    return np.sum(np.abs(x) ** (np.arange(len(x)) + 2))

# Create a random population within the bounds [-1, 1]
def create_population(pop_size, dimensions):
    return np.random.uniform(-1, 1, (pop_size, dimensions))

# Evaluate the fitness of the population
def evaluate_population(population):
    return np.array([objective_function(ind) for ind in population])

# Select parents based on their fitness (lower = better)
def select_parents(fitness, num_parents):
    # Roulette wheel selection
    max_fitness = np.max(fitness) + 1
    inverted_fitness = max_fitness - fitness 
    total_fit = np.sum(inverted_fitness)
    probs = inverted_fitness / total_fit
    parent_indices = np.random.choice(np.arange(len(fitness)),
                                      size=num_parents,
                                      p=probs)
    return parent_indices

# Perform crossover between pairs of parents
def crossover(parents, pop_size):
    offspring = np.empty((pop_size, parents.shape[1]))
    crossover_point = np.uint8(parents.shape[1] / 2)
    for k in range(pop_size):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

# Mutate the population
def mutate(offspring_crossover, mutation_rate):
    mutations_counter = np.uint8(mutation_rate * offspring_crossover.shape[1])
    for idx in range(offspring_crossover.shape[0]):
        gene_indices = np.random.choice(np.arange(offspring_crossover.shape[1]),
                                        mutations_counter)
        # Randomly change the values of the selected genes
        offspring_crossover[idx, gene_indices] = np.random.uniform(-1, 1, mutations_counter)
    return offspring_crossover

# Genetic algorithm parameters
dimensions = 2
pop_size = 10
num_parents = 4
mutation_rate = 0.2
num_generations = 50

# Initialize the population
population = create_population(pop_size, dimensions)

# Run the genetic algorithm
for generation in range(num_generations):
    # Evaluate the current population
    fitness = evaluate_population(population)
    
    # Select the best parents for mating
    parent_indices = select_parents(fitness, num_parents)
    parents = population[parent_indices]
    
    # Generate offspring through crossover
    offspring_crossover = crossover(parents, pop_size - parents.shape[0])
    
    # Add mutations to the offspring
    offspring_mutation = mutate(offspring_crossover, mutation_rate)
    
    # Create new population based on the parents and offspring
    population[0:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = offspring_mutation
    
    # Print the best result in current iteration
    print(f"Best fitness at generation {generation + 1}: {np.min(fitness)}")
    
# Best solution after all generations
best_fitness = np.min(evaluate_population(population))
best_index = np.argmin(evaluate_population(population))
best_solution = population[best_index]

best_fitness, best_solution