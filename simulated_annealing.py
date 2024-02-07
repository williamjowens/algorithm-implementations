import numpy as np 
import math 

# Define the function
def objective_function(x):
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))

# Simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # Generate an initial point
    best = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    
    # Evaluate initial point
    best_eval = objective(best)
    
    # Current working solution
    curr, curr_eval = best, best_eval
    
    # Run the algorithm
    for i in range(n_iterations):
        # Take a step
        candidate = curr + np.random.randn(len(bounds)) * step_size
        
        # Evaluate candidate point
        candidate_eval = objective(candidate)
        
        # Check for new best solution
        if candidate_eval < best_eval:
            # Store new best point
            best, best_eval = candidate, candidate_eval
            
            # Report progress
            print(">%d f(%s) = %.5f" % (i, best, best_eval))
            
        # Difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        
        # Calculate temperature for current epoch
        t = temp / float(i + 1)
        
        # Calculate metropolis acceptance criterion
        metropolis = math.exp(-diff / t)
        
        # Check whether to keep new point
        if diff < 0 or np.random.rand() < metropolis:
            # Store the new current point
            curr, curr_eval = candidate, candidate_eval
            
    return [best, best_eval]


# Define bounds and parameters for SA
bounds = np.asarray([[-2 * np.pi, 2 * np.pi]] * 2)
n_iterations = 1000
step_size = 0.1
temp = 10

# Run SA optimization
best, score = simulated_annealing(objective_function, bounds,
                                  n_iterations, step_size, temp)

print("Done!")
print("Best Solution: f(%s) = %.5f" % (best, score))