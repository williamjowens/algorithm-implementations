import numpy as np

# Define the fuzzy membership functions for demand
def low_demand_membership(x):
    return np.clip(1 - x / 500, 0, 1)

def medium_demand_membership(x):
    return np.clip(np.minimum(x / 500, 1 - (x - 500) / 500), 0, 1)

def high_demand_membership(x):
    return np.clip((x - 500) / 500, 0, 1)

# Define the objective function (profit maximization)
def objective(product_A, product_B, inventory_A, inventory_B):
    return 50 * product_A + 80 * product_B - 10 * inventory_A - 15 * inventory_B

# Define the constraints
def production_capacity(product_A, product_B):
    return 2 * product_A + 3 * product_B <= 3000

def inventory_balance_A(product_A, inventory_A):
    return inventory_A == 100 + product_A - 600

def inventory_balance_B(product_B, inventory_B):
    return inventory_B == 150 + product_B - 800

def fuzzy_demand_A(product_A):
    return medium_demand_membership(product_A) >= 0.8

def fuzzy_demand_B(product_B):
    return high_demand_membership(product_B) >= 0.7

def branch_and_bound(product_A_range, product_B_range, inventory_A_range, inventory_B_range, best_solution, best_objective):
    if len(product_A_range) == 1 and len(product_B_range) == 1 and len(inventory_A_range) == 1 and len(inventory_B_range) == 1:
        pA, pB, iA, iB = product_A_range[0], product_B_range[0], inventory_A_range[0], inventory_B_range[0]
        if (
            production_capacity(pA, pB) and
            inventory_balance_A(pA, iA) and
            inventory_balance_B(pB, iB) and
            fuzzy_demand_A(pA) and
            fuzzy_demand_B(pB)
        ):
            obj_value = objective(pA, pB, iA, iB)
            if obj_value > best_objective:
                best_solution = (pA, pB, iA, iB)
                best_objective = obj_value
    else:
        if len(product_A_range) > 1:
            mid = len(product_A_range) // 2
            left_range, right_range = product_A_range[:mid], product_A_range[mid:]
            best_solution, best_objective = branch_and_bound(left_range, product_B_range, inventory_A_range, inventory_B_range, best_solution, best_objective)
            best_solution, best_objective = branch_and_bound(right_range, product_B_range, inventory_A_range, inventory_B_range, best_solution, best_objective)
        elif len(product_B_range) > 1:
            mid = len(product_B_range) // 2
            left_range, right_range = product_B_range[:mid], product_B_range[mid:]
            best_solution, best_objective = branch_and_bound(product_A_range, left_range, inventory_A_range, inventory_B_range, best_solution, best_objective)
            best_solution, best_objective = branch_and_bound(product_A_range, right_range, inventory_A_range, inventory_B_range, best_solution, best_objective)
        elif len(inventory_A_range) > 1:
            mid = len(inventory_A_range) // 2
            left_range, right_range = inventory_A_range[:mid], inventory_A_range[mid:]
            best_solution, best_objective = branch_and_bound(product_A_range, product_B_range, left_range, inventory_B_range, best_solution, best_objective)
            best_solution, best_objective = branch_and_bound(product_A_range, product_B_range, right_range, inventory_B_range, best_solution, best_objective)
        elif len(inventory_B_range) > 1:
            mid = len(inventory_B_range) // 2
            left_range, right_range = inventory_B_range[:mid], inventory_B_range[mid:]
            best_solution, best_objective = branch_and_bound(product_A_range, product_B_range, inventory_A_range, left_range, best_solution, best_objective)
            best_solution, best_objective = branch_and_bound(product_A_range, product_B_range, inventory_A_range, right_range, best_solution, best_objective)

    return best_solution, best_objective

def main():
    product_A_range = list(range(0, 1001))
    product_B_range = list(range(0, 1001))
    inventory_A_range = list(range(0, 501))
    inventory_B_range = list(range(0, 501))

    best_solution = None
    best_objective = float('-inf')

    best_solution, best_objective = branch_and_bound(product_A_range, product_B_range, inventory_A_range, inventory_B_range, best_solution, best_objective)

    if best_solution is not None:
        print("Objective value (Profit): $", best_objective)
        print("Product A production:", best_solution[0])
        print("Product B production:", best_solution[1])
        print("Inventory A:", best_solution[2])
        print("Inventory B:", best_solution[3])
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    main()