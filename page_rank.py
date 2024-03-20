#######################
# Page Rank Algorithm #
#######################
def page_rank(graph, d=0.85, max_iterations=100, tolerance=1e-6):
    """
    Compute the PageRank of each node in a graph.

    :param graph: Adjacency matrix representing the graph, where graph[i][j] is
                  True if there is an edge from i to j.
    :param d: Damping factor, typically set to 0.85.
    :param max_iterations: Maximum number of iterations to perform.
    :param tolerance: Tolerance for convergence.
    :return: Dictionary containing the PageRank of each node.
    """
    n = len(graph)
    if n == 0 or any(len(row) != n for row in graph):
        raise ValueError("Graph must be non-empty and square")

    rank = {i: 1 / n for i in range(n)}  # Initial rank for each node

    for iteration in range(max_iterations):
        new_rank = {}
        total_change = 0  # Track total change in rank for convergence check

        for i in range(n):
            incoming_rank_sum = sum(rank[j] / sum(graph[j]) for j in range(n) if graph[j][i])
            new_rank[i] = (1 - d) / n + d * incoming_rank_sum

            total_change += abs(new_rank[i] - rank[i])

        if total_change < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            return new_rank

        rank = new_rank

    print("Reached maximum iterations without convergence")
    return rank

# Graph
graph = [
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
]

rank = page_rank(graph)
print(rank)

############################################
# Page Rank Algorithm with Personalization #
############################################
def page_rank(graph, d=0.85, max_iterations=100, tolerance=1e-6, personalization=None):
    """
    Compute the PageRank of each node in a graph.

    :param graph: Adjacency matrix representing the graph.
    :param d: Damping factor, 0.85.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Convergence tolerance.
    :param personalization: Personalization vector indicating the probability to jump to each node.
    :return: Dictionary with node indices as keys and PageRank as values.
    """
    n = len(graph)
    if n == 0 or any(len(row) != n for row in graph):
        raise ValueError("Graph must be non-empty and square")

    if personalization is None:
        personalization = {i: 1 / n for i in range(n)}
    else:
        if sum(personalization.values()) != 1:
            raise ValueError("Personalization vector must sum to 1")

    rank = {i: 1 / n for i in range(n)}
    
    for iteration in range(max_iterations):
        new_rank = {}
        dangling_sum = sum(rank[i] for i in range(n) if sum(graph[i]) == 0)

        for i in range(n):
            incoming_rank_sum = sum(rank[j] / sum(graph[j]) for j in range(n) if graph[j][i] and sum(graph[j]) > 0)
            new_rank[i] = (1 - d) * personalization[i] + d * (incoming_rank_sum + dangling_sum / n)

        total_change = sum(abs(new_rank[i] - rank[i]) for i in range(n))

        if total_change < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            return new_rank

        rank = new_rank

    print("Reached maximum iterations without convergence")
    return rank

# Graph
graph = [
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
]

# Personalization vector
personalization = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}

rank = page_rank(graph, personalization=personalization)
print(rank)