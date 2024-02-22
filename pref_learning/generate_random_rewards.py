import numpy as np
from model_analysis.improve_trajectory import initialize_reward

np.random.seed(42)

true_rewards = [initialize_reward(5) for _ in range(4)]
print(true_rewards)
np.save('true_rewards.npy', true_rewards)