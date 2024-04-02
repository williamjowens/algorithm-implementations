import heapq
from typing import Callable, TypeVar, List, Tuple

T = TypeVar('T')

def beam_search(start_state: T, generate_successors: Callable[[T], List[T]], 
                is_goal: Callable[[T], bool], heuristic: Callable[[T], float], 
                beam_width: int, max_iterations: int = 1000) -> Tuple[T, List[T]]:
    
    # Initialize the beam with the start state and its heuristic score
    beam = [(heuristic(start_state), 0, start_state, [start_state])]
    iterations = 0

    while beam and iterations < max_iterations:
        # Get the top states from the beam
        scores, _, states, paths = zip(*beam)
        
        # Check if any of the states is a goal state
        for state, path in zip(states, paths):
            if is_goal(state):
                return state, path
        
        # Generate successor states for each state in the beam
        successors = []
        for state, path in zip(states, paths):
            for successor in generate_successors(state):
                if successor not in path:  # Avoid cycles
                    successors.append((heuristic(successor), len(path), successor, path + [successor]))
        
        # Select the top beam_width states based on their heuristic scores
        beam = heapq.nsmallest(beam_width, successors)
        iterations += 1
    
    # If no goal state is found, return None and an empty path
    return None, []

def main():
    # Example problem: Find a path from the start city to the goal city
    # Each state is represented as a tuple (city, cost)
    # The cost represents the total cost of reaching that city from the start city

    def generate_successors(state):
        city, cost = state
        successors = []
        if city == 'A':
            successors.append(('B', cost + 5))
            successors.append(('C', cost + 2))
        elif city == 'B':
            successors.append(('D', cost + 3))
            successors.append(('E', cost + 1))
        elif city == 'C':
            successors.append(('F', cost + 6))
        elif city == 'D':
            successors.append(('G', cost + 4))
        elif city == 'E':
            successors.append(('G', cost + 2))
            successors.append(('H', cost + 7))
        elif city == 'F':
            successors.append(('I', cost + 3))
        elif city == 'H':
            successors.append(('J', cost + 1))
        elif city == 'I':
            successors.append(('J', cost + 5))
        return successors

    def is_goal(state):
        city, _ = state
        return city == 'J'

    def heuristic(state):
        city, cost = state
        
        # Estimated cost from each city to the goal city 'J'
        heuristic_costs = {
            'A': 10,
            'B': 6,
            'C': 8,
            'D': 4,
            'E': 7,
            'F': 6,
            'G': 3,
            'H': 2,
            'I': 4
        }
        return cost + heuristic_costs.get(city, 0)

    start_state = ('A', 0)
    beam_width = 3
    max_iterations = 1000

    goal_state, path = beam_search(start_state, generate_successors, is_goal, heuristic, beam_width, max_iterations)

    if goal_state:
        print("Goal state found:", goal_state)
        print("Path:", [city for city, _ in path])
        print("Total cost:", goal_state[1])
    else:
        print("No goal state found.")

if __name__ == "__main__":
    main()