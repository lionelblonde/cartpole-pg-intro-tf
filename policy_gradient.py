import gym
# from gym import wrappers
# import math
import random
import numpy as np
import tensorflow as tf  # since we now use gradients
import matplotlib.pyplot as plt


# This function is useless here: tf provides tf.nn.softmax()
def softmax(x):  # x is a vector
    # substract by the max for num stability - mathematically equiv to stock softmax
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# The use of tf scopes enables us to use loss, optimizer, etc. names for both functions


# Update our policy to prefer certain actions
def policy_gradient():
    with tf.variable_scope("policy"):
        # state space dimension = 4, action space dimension = 2
        # policy = one linear combination of the state variables per action
        parameters = tf.get_variable("policy_parameters", [4, 2])
        state = tf.placeholder("float", [None, 4])
        action = tf.placeholder("float", [None, 2])
        advantage = tf.placeholder("float", [None, 1])
        linear = tf.matmul(state, parameters)  # no bias, outputs a vector ([1, 2])
        # Softmax activation: transforms the vector in probs of playing each action ([1, 2])
        # it is the usual choice as output activation for classification problems (sig too)
        computed_probs = tf.nn.softmax(linear)
        action_prob = tf.reduce_sum(tf.multiply(computed_probs, action), axis=1)
        # element-wise mul
        # action is a one-hot vector, so the element-wise mul outputs a one-hot vector
        # reduce_sum along the 1 axis transforms the one-hot vector into the scalar it contains
        # The two steps could be replaced by one dot product
        eligibility = tf.log(action_prob) * advantage  # no np.matmul since both are scalars
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    return computed_probs, state, action, advantage, optimizer


# How to measure the success of performing given actions in given states: values
# 1 hidden layer NN (10 neurons wide hidden layer) to determine the best action for a state
# input is the state ([1, 4])
# output is the value ([1, 1])
def value_gradient():
    with tf.variable_scope("value"):
        # Calculate the value of a state
        state = tf.placeholder("float", [None, 4])
        w1 = tf.get_variable("w1", [4, 10])  # weight matrix input (state) -> hidden
        b1 = tf.get_variable("b1", [1, 10])  # bias vector input (state) -> hidden
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)  # hidden layer, ReLU activation
        w2 = tf.get_variable("w2", [10, 1])  # weight matrix hidden -> output (value)
        b2 = tf.get_variable("b2", [1, 1])  # bias vector hidden -> output (value)
        computed_value = tf.matmul(h1, w2) + b2  # linear activation
        # it is the usual choice as output activation for regression problems

        # Update the value of a state
        new_value = tf.placeholder("float", [None, 1])
        loss = tf.nn.l2_loss(computed_value - new_value)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

    return computed_value, state, new_value, loss, optimizer


# Run episodes to gather data, similarly to random search and hillclimbing
# except that now we want to recoard the transitions and rewards gotten from them
def run_episode(env, policy_grad, value_grad, sess):
    pl_computed_probs, pl_state, pl_action, pl_advantage, pl_optimizer = policy_grad
    vl_computed_value, vl_state, vl_new_value, vl_loss, vl_optimizer = value_grad
    observation = env.reset()  # contains initial state information, wrong format though
    total_reward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_values = []

    # Run the episode
    for timestep in range(200):
        # env.render()  # uncomment to see the simulation as it runs
        # Step 1: compute the policy
        # Reshape observation from [4,] -> [1, 4] to coincide with state
        observed_state = np.expand_dims(observation, axis=0)
        # Compute the probabilities over actions in the observed states
        action_probs = sess.run(pl_computed_probs, feed_dict={pl_state: observed_state})
        # pl_computed_probs is a list -> sess.run returns a list
        # the returned list contains one element, which is a [1, 2] list containing the probs
        # [[action_0_prob, action_1_prob]] -> 2 square brackets
        # since we asked for one element (which happens to be a list), as opposed to several (if
        # we asked for several elements), for which we would have gotten a list of those elements
        action = 0 if random.uniform(0, 1) < action_probs[0][0] else 1
        # this ensures that the action is picked non-deterministically
        # otherwise we would just deterministically pick the action with highest prob all the time
        # instead of picking it according to its probability

        # Step 2: record the transition
        states.append(observation)  # observation before reshape
        action_one_hot = np.zeros(2)
        action_one_hot[action] = 1  # one-hot vector indicating which action to perform
        actions.append(action_one_hot)
        # Take action in the environment
        old_observation = observation  # already appened to states
        observation, reward, done, info = env.step(action)  # OpenAI Gym API
        transitions.append((old_observation, action, reward))
        # note that we ignore the s_{t+1}, the state we arrive at: observation
        total_reward += reward

        if done:
            print("--- episode finished after %s timesteps ---" % (timestep))
            break

    # Compute the return
    for index, transition in enumerate(transitions):
        observation, action, reward = transition  # reward useless: only future rewards

        # Step 1: calculate the discounted MC returned
        gamma = 0.97  # discount factor
        _return = 0  # only interested in the future reward, not the current or previous ones
        # _return is the empirical estimate of the Q-value
        episode_duration = len(transitions)  # reminder: there is one transition per timestep
        number_remaining_transitions = episode_duration - index
        for i in range(number_remaining_transitions):
            # add the immediate rewards of each remaining transitions in the current episode
            _return += transitions[index + i][2] * (gamma ** i)

        # Step 2: record the advantage
        observed_state = np.expand_dims(observation, axis=0)  # reshape to match state
        current_value = sess.run(vl_computed_value, feed_dict={vl_state: observed_state})[0][0]
        # [0][0] to go from [[value]] to value
        advantages.append(_return - current_value)

        # Step 3: record the return for value updating
        update_values.append(_return)

    # Update value function
    update_values_vector = np.expand_dims(update_values, axis=1)  # from [n,] to [n, 1] (vector)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_new_value: update_values_vector})

    # Update the policy
    advantages_vector = np.expand_dims(advantages, axis=1)  # from [m,] to [m, 1] (vector)
    sess.run(pl_optimizer,
             feed_dict={pl_state: states, pl_action: actions, pl_advantage: advantages_vector})

    return total_reward


def train(submit):
    env = gym.make('CartPole-v0')
    # env = wrappers.Monitor(env, "./cartpole-experiment")
    tf.reset_default_graph()  # necessary to clean up inbetween episodes

    policy_grad = policy_gradient()
    value_grad = value_gradient()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for episode in range(2000):
        reward = run_episode(env, policy_grad, value_grad, sess)
        print("--- episode %d | reward %d ---" % (episode, reward))
        if reward == 200:  # upper threshold = in balance for 200 timesteps
            print("Stood up for 200 timesteps!")
            break

    return episode


# Graphs
results = []
for _ in range(50):
    results.append(train(submit=False))


plt.hist(results, 50, normed=1, facecolor="g", alpha=0.75)
plt.xlabel("Episodes required to reach 200")
plt.ylabel("Frequency")
plt.title("Histogram of Policy Gradient")
plt.savefig("cartpole-policy-gradient.png")
plt.show()  # has to be after savefig call, otherwise blank image in file

print("Average #episode required to reach target score (200): %s" % (np.sum(results) / 50.0))
