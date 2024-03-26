import numpy as np
from itertools import product

class GameTheoryOptimization:
    def __init__(self, num_players, strategies, payoff_matrix):
        self.num_players = num_players
        self.strategies = strategies
        self.payoff_matrix = payoff_matrix

    def generate_strategy_profiles(self):
        return list(product(range(len(self.strategies)), repeat=self.num_players))

    def get_payoff(self, strategy_profile):
        payoffs = self.payoff_matrix
        for player, strategy in enumerate(strategy_profile):
            payoffs = payoffs[strategy]
        return payoffs

    def find_best_responses(self, strategy_profiles):
        best_responses = []

        for player in range(self.num_players):
            player_best_responses = []

            for profile in strategy_profiles:
                player_strategy = profile[player]
                payoffs = self.get_payoff(profile)
                player_payoff = payoffs[player]

                if not player_best_responses or player_payoff > player_best_responses[-1][1]:
                    player_best_responses = [(self.strategies[player_strategy], player_payoff)]
                elif player_payoff == player_best_responses[-1][1]:
                    player_best_responses.append((self.strategies[player_strategy], player_payoff))

            best_responses.append([response[0] for response in player_best_responses])

        return best_responses

    def find_nash_equilibria(self, strategy_profiles):
        nash_equilibria = []

        for profile in strategy_profiles:
            is_nash_equilibrium = True

            for player in range(self.num_players):
                payoffs = self.get_payoff(profile)
                player_payoff = payoffs[player]
                player_strategies = range(len(self.strategies))

                for alternative_strategy in player_strategies:
                    if alternative_strategy != profile[player]:
                        alternative_profile = tuple(profile[:player] + (alternative_strategy,) + profile[player+1:])
                        alternative_payoffs = self.get_payoff(alternative_profile)
                        alternative_payoff = alternative_payoffs[player]

                        if alternative_payoff > player_payoff:
                            is_nash_equilibrium = False
                            break

                if not is_nash_equilibrium:
                    break

            if is_nash_equilibrium:
                nash_equilibria.append(profile)

        return nash_equilibria

    def is_pareto_efficient(self, profile, strategy_profiles):
        for alternative_profile in strategy_profiles:
            if alternative_profile != profile:
                is_dominated = True
                for player in range(self.num_players):
                    if self.get_payoff(alternative_profile)[player] > self.get_payoff(profile)[player]:
                        is_dominated = False
                        break
                if is_dominated:
                    return False

        return True

def main():
    # Define the game parameters
    num_players = 3
    strategies = ['Cooperate', 'Defect']
    payoff_matrix = [
        [
            [(6, 6, 6), (0, 8, 0)],
            [(8, 0, 0), (2, 2, 2)]
        ],
        [
            [(6, 6, 6), (8, 0, 0)],
            [(0, 8, 0), (2, 2, 2)]
        ],
        [
            [(6, 6, 6), (0, 0, 8)],
            [(0, 0, 8), (2, 2, 2)]
        ]
    ]

    game = GameTheoryOptimization(num_players, strategies, payoff_matrix)

    print("Game Setup:")
    print("Number of players:", game.num_players)
    print("Strategies:", game.strategies)
    print("Payoff Matrix:")
    for player_payoff_matrix in game.payoff_matrix:
        print(np.array(player_payoff_matrix))
    print()

    strategy_profiles = game.generate_strategy_profiles()
    print("Strategy Profiles:")
    print(strategy_profiles)
    print()

    best_responses = game.find_best_responses(strategy_profiles)
    print("Best Responses:")
    for player, responses in enumerate(best_responses, start=1):
        print(f"Player {player}: {responses}")
    print()

    nash_equilibria = game.find_nash_equilibria(strategy_profiles)
    print("Nash Equilibria:")
    for equilibrium in nash_equilibria:
        for player, strategy in enumerate(equilibrium, start=1):
            print(f"Player {player} strategy: {game.strategies[strategy]}")

        if game.is_pareto_efficient(equilibrium, strategy_profiles):
            print("This equilibrium is Pareto efficient.")
        else:
            print("This equilibrium is not Pareto efficient.")

        print("Payoffs:")
        payoffs = game.get_payoff(equilibrium)
        for player, payoff in enumerate(payoffs, start=1):
            print(f"Player {player}: {payoff}")
        print()

if __name__ == "__main__":
    main()