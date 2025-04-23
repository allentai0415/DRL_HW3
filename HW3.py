import numpy as np
import matplotlib.pyplot as plt

# Define the number of arms and the true mean rewards for each arm
num_arms = 10
mean_rewards = np.random.normal(0, 1, num_arms)

# Define the number of iterations and the epsilon parameter for the epsilon-greedy algorithm
num_iterations = 1000
epsilon = 0.1

# Define the initial estimated rewards for each arm
estimated_rewards = np.zeros(num_arms)

# Define the history of rewards and choices for each algorithm
epsilon_greedy_rewards = []
epsilon_greedy_choices = []
ucb_rewards = []
ucb_choices = []
thompson_rewards = []
thompson_choices = []
softmax_rewards = []
softmax_choices = []

# Define the main loop for the simulation
for i in range(num_iterations):
    # Epsilon-Greedy Algorithm
    if np.random.rand() < epsilon:
        # Select a random arm
        arm = np.random.randint(num_arms)
    else:
        # Select the arm with the highest estimated reward
        arm = np.argmax(estimated_rewards)
    # Pull the selected arm and observe the reward
    reward = np.random.normal(mean_rewards[arm], 1)
    # Update the estimated reward for the selected arm
    estimated_rewards[arm] += (1/(i+1)) * (reward - estimated_rewards[arm])
    # Append the reward and choice to the history lists
    epsilon_greedy_rewards.append(reward)
    epsilon_greedy_choices.append(arm)
    
    # UCB Algorithm
    ucb = estimated_rewards + np.sqrt(2*np.log(i+1)/(1+np.array(list(range(num_arms)))))  # Calculate the upper confidence bound for each arm
    arm = np.argmax(ucb)  # Select the arm with the highest upper confidence bound
    reward = np.random.normal(mean_rewards[arm], 1)
    estimated_rewards[arm] += (1/(i+1)) * (reward - estimated_rewards[arm])
    ucb_rewards.append(reward)
    ucb_choices.append(arm)
    
    # Thompson Sampling
    samples = [np.random.normal(estimated_rewards[j], 1) for j in range(num_arms)]  # Sample a reward distribution for each arm
    arm = np.argmax(samples)  # Select the arm with the highest expected reward based on the sampled distributions
    reward = np.random.normal(mean_rewards[arm], 1)
    estimated_rewards[arm] += (1/(i+1)) * (reward - estimated_rewards[arm])
    thompson_rewards.append(reward)
    thompson_choices.append(arm)
    
    # Softmax Action Selection
    softmax_probs = np.exp(estimated_rewards) / np.sum(np.exp(estimated_rewards))  # Compute the softmax probabilities for each arm
    arm = np.random.choice(range(num_arms), p=softmax_probs)  # Select an arm according to the computed probabilities
    reward = np.random.normal(mean_rewards[arm], 1)
    estimated_rewards[arm] += (1/(i+1)) * (reward - estimated_rewards[arm])
    softmax_rewards.append(reward)
    softmax_choices.append(arm)

# Plot the performance of each algorithm in terms of convergence to mean values
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(epsilon_greedy_rewards) / (1 + np.arange(num_iterations)), label='Epsilon-Greedy')
plt.plot(np.cumsum(ucb_rewards) / (1 + np.arange(num_iterations)), label='UCB')
plt.plot(np.cumsum(thompson_rewards) / (1 + np.arange(num_iterations)), label='Thompson Sampling')
plt.plot(np.cumsum(softmax_rewards) / (1 + np.arange(num_iterations)), label='Softmax Action Selection')
plt.legend()
plt.title('Performance of Multi-Armed Bandit Algorithms')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Average Reward')
plt.show()