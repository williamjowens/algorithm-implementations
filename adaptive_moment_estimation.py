import numpy as np
import matplotlib.pyplot as plt

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, params):
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)

    def update(self, params, grads):
        self.t += 1

        if self.m is None:
            self.initialize(params)

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params

def objective_function(x):
    return np.square(x - 2)

def gradient_function(x):
    return 2 * (x - 2)

def optimize(optimizer, params, num_iterations, callback=None):
    objective_values = []
    for i in range(num_iterations):
        # Compute gradients
        grads = gradient_function(params)

        # Update parameters using the optimizer
        params = optimizer.update(params, grads)

        # Record objective function value
        objective_values.append(objective_function(params))

        # Invoke callback function, if provided
        if callback is not None:
            callback(params, i, objective_values)

    return params, objective_values

def main():
    # Initialize parameters
    params = np.array([10.0])

    # Create Adam optimizer with learning rate decay
    def learning_rate_schedule(iteration):
        return 0.1 * (0.1 ** (iteration // 50))

    optimizer = AdamOptimizer(learning_rate=learning_rate_schedule(0))

    # Define callback function
    def print_progress(params, iteration, objective_values):
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}: Parameters = {params}, Objective = {objective_values[-1]}")
            optimizer.learning_rate = learning_rate_schedule(iteration)

    # Perform optimization
    num_iterations = 100
    optimized_params, objective_values = optimize(optimizer, params, num_iterations, callback=print_progress)

    print("\nOptimized Parameters:")
    print(optimized_params)
    print("Objective Function Value:")
    print(objective_function(optimized_params))

    # Plot objective function values
    plt.plot(range(1, num_iterations + 1), objective_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Optimization Progress")
    plt.show()

if __name__ == "__main__":
    main()