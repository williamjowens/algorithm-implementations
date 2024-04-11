import numpy as np

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

def rosen_der(x):
    """Derivative of the Rosenbrock function"""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der

def line_search(func, x, d, a=0.3, c=0.8, maxiter=100):
    """Backtracking line search"""
    f0 = func(x)
    g0 = np.dot(rosen_der(x), d)
    for i in range(maxiter):
        fi = func(x + a * d)
        if np.all(fi <= f0 + c * a * g0):
            break
        a *= 0.5
    return a

def two_loop_recursion(s, y, rho, alpha, q, H0):
    """Two-loop recursion for L-BFGS"""
    k = len(s)
    for i in range(k - 1, -1, -1):
        alpha[i] = rho[i] * np.sum(s[i] * q)
        q = q - alpha[i] * y[i]
    r = H0.dot(q)
    for i in range(k):
        beta = rho[i] * np.sum(y[i] * r)
        r = r + s[i] * (alpha[i] - beta)
    return r

def l_bfgs(func, x0, m=10, maxiter=1000, tol=1e-6):
    """L-BFGS optimization algorithm"""
    x = np.asarray(x0).flatten()
    n = len(x)
    s = []
    y = []
    rho = []
    H0 = np.eye(n)
    
    for i in range(maxiter):
        g = rosen_der(x)
        if np.linalg.norm(g) < tol:
            break
        
        if i == 0:
            d = -g
        else:
            q = g
            alpha = np.zeros(len(s))
            d = -two_loop_recursion(s, y, rho, alpha, q, H0)
        
        a = line_search(func, x, d)
        s_new = a * d
        x_new = x + s_new
        y_new = rosen_der(x_new) - g
        
        if len(s) == m:
            s.pop(0)
            y.pop(0)
            rho.pop(0)
        
        s.append(s_new)
        y.append(y_new)
        
        s_new = s_new.flatten()
        y_new = y_new.flatten()
        
        ys = np.dot(y_new, s_new)
        if ys != 0:
            rho.append(1.0 / ys)
        else:
            rho.append(1.0)  # Handle division by zero
        
        yy = np.dot(y_new, y_new)
        if yy != 0:
            H0 = np.dot(s_new, y_new) / yy * np.eye(n)
        else:
            H0 = np.eye(n)  # Handle division by zero
        
        x = x_new
    
    return x, i + 1

def main():
    x0 = np.array([-1.2, 1.0])
    x_opt, num_iter = l_bfgs(rosen, x0)
    print(f"Optimum point: {x_opt}")
    print(f"Number of iterations: {num_iter}")
    print(f"Minimum value: {rosen(x_opt)}")

if __name__ == "__main__":
    main()