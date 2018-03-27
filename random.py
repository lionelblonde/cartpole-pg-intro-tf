import gym
import numpy as np
import matplotlib.pyplot as plt


# Reward yielded by an episode
def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0
    for timestep in range(500):
        env.render()
        # env.render(close=True)  # hide the sim - necessary for matplotlib
        # equivalently, one could also remove the render() line
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
    best_reward = 0

    for episode in range(10000):
        parameters = np.random.rand(4) * 2 - 1  # between -1 and 1
        reward = run_episode(env, parameters)
        print("--- episode %d | reward %d | best %d ---" % (episode, reward, best_reward))
        if reward > best_reward:
            best_reward = reward
            if reward == 200:  # upper threshold = in balance for 200 timesteps
                print("Stood up for 200 timesteps!")
                break

    return episode


# Graphs
results = []
for _ in range(1000):
    results.append(train(submit=False))


plt.hist(results, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel("Episodes required to reach 200")
plt.ylabel("Frequency")
plt.title("Histogram of Random Search")
plt.savefig("cartpole-random.png")
plt.show()  # has to be after savefig call, otherwise blank image in file

print("Average #episode required to reach target score (200): %s" % (np.sum(results) / 1000.0))
