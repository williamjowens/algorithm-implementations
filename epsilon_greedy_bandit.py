import random
import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, num_arms, epsilon, epsilon_decay, epsilon_min, action_probs):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min 
        self.action_probs = action_probs 
        self.q_values = [0] * num_arms 
        self.counts = [0] * num_arms 
        
    def select_arm(self):
        if random.random() < self.epsilon:
            # Explore: select an arm based on action probabilities
            arm = np.random.choice(range(self.num_arms), p=self.action_probs)
        else:
            # Exploit: select the arm with the highest estimated value
            arm = self.q_values.index(max(self.q_values))
        return arm
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.q_values[arm]
        new_value = value + (1 / n) * (reward - value)
        self.q_values[arm] = new_value
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

def get_reward(arm):
    # Simulated reward distribution for each arm
    reward_distributions = [
        lambda: random.gauss(1.0, 1.0),  # Arm 0
        lambda: random.gauss(2.0, 1.0),  # Arm 1
        lambda: random.gauss(1.5, 1.0),  # Arm 2
    ]
    return reward_distributions[arm]()
   
def simulate(bandit, num_trials):
    rewards = []
    best_arm_counts = []
    for trial in range(num_trials):
        arm = bandit.select_arm()
        reward = get_reward(arm)
        bandit.update(arm, reward)
        bandit.update_epsilon()
        rewards.append(reward)
        best_arm = bandit.q_values.index(max(bandit.q_values))
        best_arm_counts.append(int(arm == best_arm))
    return rewards, best_arm_counts 

def plot_results(rewards, best_arm_counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cumulative rewards
    ax1.plot(range(len(rewards)), np.cumsum(rewards))
    ax1.set_xlabel("Trials")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Cumulative Reward Over Trials")
    
    # Plot percentage of best arm selections
    best_arm_percentages = np.cumsum(best_arm_counts) / (np.arange(len(best_arm_counts)) + 1) * 100
    ax2.plot(range(len(best_arm_percentages)), best_arm_percentages)
    ax2.set_xlabel("Trials")
    ax2.set_ylabel("Percentage of Best Arm Selections")
    ax2.set_title("Percentage of Best Arm Selections Over Trials")
    
    plt.tight_layout()
    plt.show()
    
def main():
    num_arms = 3
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    action_probs = [0.3, 0.5, 0.2]
    num_trials = 1000
    
    bandit = EpsilonGreedyBandit(num_arms, epsilon, epsilon_decay, epsilon_min, action_probs)
    
    rewards, best_arm_counts = simulate(bandit, num_trials)
    
    print(f"Number of arms: {num_arms}")
    print(f"Initial epsilon: {epsilon}")
    print(f"Epsilon decay: {epsilon_decay}")
    print(f"Minimum epsilon: {epsilon_min}")
    print(f"Action probabilities: {action_probs}")
    print(f"Number of trials: {num_trials}")
    print(f"Final Q-values: {bandit.q_values}")
    print(f"Total reward: {sum(rewards)}")

    plot_results(rewards, best_arm_counts)

if __name__ == "__main__":
    main()