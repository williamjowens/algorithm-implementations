import numpy as np 

# Define the objective function
def objective_function(x, beta):
    n = x.size
    total_sum = 0
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            # Check for division by zero or other issues
            if i == 0 or (x[j - 1] < 0 and (j / i) % 1 != 0):
                continue
            total_sum += (j**2 + beta) * (x[j - 1]**(j / i) - 1)
    return total_sum

# Particle Swarm Optimization
class Particle:
    def __init__(self, bounds, beta):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_value = float("inf")
        self.beta = beta
        
    def evaluate(self, objective_function):
        value = objective_function(self.position, self.beta)
        if value < self.best_value:
            self.best_value = value
            self.best_position = self.position 
            
    def update_velocity(self, global_best_position, w=0.5, c1=1, c2=2):
        r1, r2 = np.random.rand(2)
        self.velocity = w * self.velocity \
                        + c1 * r1 * (self.best_position - self.position) \
                        + c2 * r2 * (global_best_position - self.position)
                        
    def update_position(self, bounds):
        self.position += self.velocity
        for i, (low, high) in enumerate(bounds):
            self.position[i] = np.clip(self.position[i], low, high)
            
class ParticleSwarmOptimizer:
    def __init__(self, objective_function, bounds, num_particles, beta, iterations):
        self.objective_function = objective_function
        self.bounds = bounds 
        self.num_particles = num_particles 
        self.beta = beta 
        self.iterations = iterations 
        self.global_best_position = None 
        self.global_best_value = float("inf")
        self.swarm = [Particle(bounds, beta) for _ in range(num_particles)]
        
    def optimize(self):
        for _ in range(self.iterations):
            for particle in self.swarm:
                particle.evaluate(self.objective_function)
                
                if particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value 
                    self.global_best_position = particle.best_position 
                    
            for particle in self.swarm:
                particle.update_velocity(self.global_best_position)
                particle.update_position(self.bounds)
                
        return self.global_best_position, self.global_best_value 
    
# Parameters for PSO
n = 5
bounds = [(-n, n)] * n
num_particles = 30
iterations = 100
betas = [0.1, 0.5, 1, 2, 5]

# Run PSO for different values of beta and records the results
results = {}

for beta in betas:
    pso = ParticleSwarmOptimizer(objective_function, bounds, num_particles, beta, iterations)
    best_position, best_value = pso.optimize()
    results[beta] = (best_position, best_value)
    
# Create a more readable format for results dictionary
readable_results = {beta: {"Best Position": list(position), "Objective Value": value} for beta, (position, value) in results.items()}

# Print results
for beta, result in readable_results.items():
    print(f"Beta = {beta}")
    print(f"   Best Position: {result['Best Position']}")
    print(f"   Objective Value: {result['Objective Value']}\n")