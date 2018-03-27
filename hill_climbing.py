import gym
import numpy as np
import matplotlib.pyplot as plt


# Reward yielded by an episode
def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0
    for timestep in range(500):
        env.render(close=True)  # hide the sim - necessary for mpl
        # possible actions are 0 (go left) or 1 (go right)
        # here the observations are mapped linearly to the actions
        action = 0 if np.dot(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        # update the total reward
        total_reward += reward
        if done:
            print("--- episode finished after %s timesteps ---" % (timestep))
            break

    return total_reward


# Hill climbing policy
def train(submit):
    env = gym.make('CartPole-v0')
    noise_scaling = 0.1  # equivalent of the learning rate for hillclimbing
    parameters = np.random.rand(4) * 2 - 1  # initialize between -1 and 1
    best_reward = 0
    counter = 0
    run_per_update = 5

    for episodes in range(1000):
        counter += 1
        new_parameters = parameters + ((np.random.rand(4) * 2 - 1) * noise_scaling)
        reward = 0
        for _ in range(run_per_update):
            run = run_episode(env, new_parameters)
            reward += run
        reward = reward / float(run_per_update)
        print("--- episode %d | reward %d | best %d ---" % (counter, reward, best_reward))
        if reward > best_reward:
            best_reward = reward
            parameters = new_parameters
            if reward == 200:  # upper threshold = in balance for 200 timesteps
                print("Stood up for 200 timesteps!")
                break

    return counter


# Graphs
results = []
for _ in range(100):
    results.append(train(submit=False))


plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel("Episodes required to reach 200")
plt.ylabel("Frequency")
plt.title("Histogram of Hillclimbing")
plt.savefig("cartpole-hillclimbing.png")
plt.show()

print("Average #episode required to reach target score (200): %s" % (np.sum(results) / 1000.0))
