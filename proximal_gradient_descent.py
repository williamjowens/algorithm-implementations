import numpy as np

def soft_threshold(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def prox_l1(x, alpha):
    return soft_threshold(x, alpha)

def prox_l2(x, alpha):
    return x / (1 + alpha)

def proximal_gradient_descent(x0, f, grad_f, g, prox_g, alpha, max_iter=100, tol=1e-6, verbose=False):
    x = x0.copy()
    f_values = []
    
    for iter in range(max_iter):
        x_prev = x.copy()
        
        # Gradient descent step
        x_intermediate = x - alpha * grad_f(x)
        
        # Proximal operator step
        x = prox_g(x_intermediate, alpha)
        
        # Compute objective function value
        f_value = f(x)
        f_values.append(f_value)
        
        # Check for convergence
        if np.linalg.norm(x - x_prev) < tol:
            if verbose:
                print(f"Converged after {iter + 1} iterations.")
            break
    
    if verbose and iter == max_iter - 1:
        print(f"Max iterations ({max_iter}) reached.")
    
    return x, f_values

def lasso_objective(X, y, beta, lambda_):
    n = len(y)
    return 1 / (2 * n) * np.sum((y - X @ beta) ** 2) + lambda_ * np.sum(np.abs(beta))

def lasso_grad(X, y, beta):
    n = len(y)
    return -1 / n * X.T @ (y - X @ beta)

def lasso_prox(beta, alpha, lambda_):
    return prox_l1(beta, alpha * lambda_)

def main():
    # Generate synthetic data
    np.random.seed(42)
    n, p = 100, 50
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    beta_true[beta_true < 1] = 0
    y = X @ beta_true + np.random.randn(n)

    # Hyperparameters
    lambda_ = 0.1
    alpha = 0.01
    max_iter = 1000
    tol = 1e-6

    # Initial guess
    beta0 = np.zeros(p)

    # Define objective function, gradient, and proximal operator
    f = lambda beta: lasso_objective(X, y, beta, lambda_)
    grad_f = lambda beta: lasso_grad(X, y, beta)
    prox_g = lambda beta, alpha: lasso_prox(beta, alpha, lambda_)

    # Run proximal gradient descent
    beta_hat, f_values = proximal_gradient_descent(beta0, f, grad_f, None, prox_g, alpha, max_iter, tol, verbose=True)

    print("\nEstimated coefficients:")
    print(beta_hat)

    # Plot objective function values
    import matplotlib.pyplot as plt
    plt.plot(f_values)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Convergence Plot")
    plt.show()

if __name__ == "__main__":
    main()