# Objective function
def objective_function(x):
    function = x**4 - 3*x**3 + 2
    return function

# Derivative of the function
def derivative(x):
    derivative_x = 4*x**3 - 9*x**2
    return derivative_x

# Gradient descent
def gradient_descent(starting_point, learning_rate, iterations):
    x = starting_point
    for i in range(iterations):
        grad = derivative(x)
        x = x - learning_rate * grad
        print(f"Iteration {i + 1}: x = {x}, f(x) = {objective_function(x)}")
    return x


# Main block
if __name__ == "__main__":
    # Parameters
    starting_point = 0.5
    learning_rate = 0.01
    iterations = 100
    
    # Run the gradient descent
    minimum = gradient_descent(starting_point, learning_rate, iterations)
    print(f"Minimum occurs at x = {minimum}")