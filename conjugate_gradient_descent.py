import numpy as np

# Conjugate gradient descent
def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    r_norm_sq = r.T @ r
    
    num_iter = 0
    for i in range(max_iter):
        Ap = A @ p
        alpha = r_norm_sq / (p.T @ Ap)
        x += alpha * p
        r -= alpha * Ap
        r_norm_sq_new = r.T @ r
        
        num_iter += 1
        if np.sqrt(r_norm_sq_new) < tol:
            break
        
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new
    
    return x, num_iter

def create_spd_matrix(n):
    """
    Create a random symmetric, positive definite matrix of size n x n
    """
    A = np.random.rand(n, n)
    A = (A + A.T) / 2
    A += n * np.eye(n)
    return A

def create_rhs_vector(n):
    """
    Create a random right-hand side vector of size n x 1
    """
    b = np.random.rand(n, 1)
    return b

# Implementation
if __name__ == "__main__":
    # Set the size of the problem
    n = 1000
    
    # Create a random symmetric, positive definite matrix A and right-hand side vector b
    A = create_spd_matrix(n)
    b = create_rhs_vector(n)
    
    # Set the initial guess for the solution
    x0 = np.zeros((n, 1))
    
    # Call the conjugate gradient descent function
    x, num_iter = conjugate_gradient(A, b, x0)
    
    # Compute the relative residual
    residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    
    print(f"Number of iterations: {num_iter}")
    print(f"Relative residual: {residual:.6e}")