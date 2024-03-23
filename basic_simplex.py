import numpy as np 

def find_pivot_column(tableau):
    """ 
    Find the pivot column using the most negative coefficient in the objective function row.
    """
    last_row = tableau[-1, :-1]
    pivot_col = np.argmin(last_row)
    return pivot_col if last_row[pivot_col] < 0 else None 

def find_pivot_row(tableau, pivot_col):
    """ 
    Find the pivot row using the minimum ratio test, exluding negative ratios. 
    """
    ratios = np.array([row[-1] / row[pivot_col] if row[pivot_col] > 0 else np.inf for row in tableau[:-1]])
    pivot_row = np.argmin(ratios)
    return pivot_row if ratios[pivot_row] != np.inf else None 

def pivot_operation(tableau, pivot_row, pivot_col):
    """ 
    Perform the pivot operation to form a new basic feasible solution. 
    """
    pivot_element = tableau[pivot_row, pivot_col]
    tableau[pivot_row, :] /= pivot_element 
    for i in range(len(tableau)):
        if i != pivot_row:
            tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
            
def simplex_method(c, A, b):
    """ 
    Simplex algorithm implementation 
    """
    
    # Assuming the starting point is already feasible
    num_constraints, num_variables = A.shape
    tableau = np.hstack((A, np.eye(num_constraints), b.reshape(-1, 1)))
    c_extended = np.hstack((c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, -c_extended.reshape(1, -1)))
    
    while True:
        pivot_col = find_pivot_column(tableau)
        if pivot_col is None:
            break
        pivot_row = find_pivot_row(tableau, pivot_col)
        if pivot_row is None:
            return None 
        pivot_operation(tableau, pivot_row, pivot_col) 
        
    # Extract solution
    solution = np.zeros(num_variables)
    for i in range(num_variables):
        col = tableau[:, i]
        if np.sum(col == 1) == 1 and np.sum(col) == 1:
            solution[i] = tableau[np.where(col == 1)[0], -1]
    return solution

# Simple example
c = np.array([5, 4])  # Objective function coefficients
A = np.array([[1, 1], [2, 1]])  # Constraint coefficients
b = np.array([5, 8]) # Constraint bounds

solution = simplex_method(c, A, b)
print(f"Optimal Solution {solution}")