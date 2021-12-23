"""Trains a model and saves it at the same time, for evaluation in visualise_model.py.
Also draws a graph with x = episode and y = average reward"""

import gym

from classes.policy import EpsilonGreedyPolicy
from classes.neural_network import *
from classes.approximator import Approximator
from classes.memory import Memory, SARSd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_learning(rewards):
    """Used to make a graph to visualize the learning process of the agent."""
    ax = sns.lineplot(rewards[0], rewards[1])

    ax.set(xlabel='Timesteps', ylabel='Rewards')
    plt.title('Average reward per timestep')

    plt.show()


def main():
    """Main loop to train an agent for the LunarLander-v2 environment."""
    # Create environment
    env = gym.make('LunarLander-v2')

    # Create and set some variables
    memory_size = 10000
    memory = Memory(memory_size)
    episodes = 5000
    batch_size = 10
    learning_rate = 0.001
    gamma = 0.9
    tau = 0.01
    copy_episodes = 5

    save_episodes = 100

    # Create the policy and target networks.
    policy_object = EpsilonGreedyPolicy()
    policy_network = Approximator(learning_rate)
    target_network = Approximator(learning_rate)

    # Used for the visualisation
    rewards = [[], []]

    for i_episode in range(episodes):  # For every episode
        observation = env.reset()  # Get the observation of the environment
        state_reward, done = 0, 0

        episode_reward = []
        for t in range(1000):  # We let an episode go on for 1000 timesteps at most
            last_observation = observation
            last_done = done

            # Select action
            action = policy_object.select_action(last_observation, env.action_space, policy_network, i_episode)

            # Get info of the environment
            observation, reward, done, info = env.step(action)

            episode_reward.append(reward)

            # Create SARSd object and add it to memory
            sarsd = SARSd(last_observation, action, reward, observation, last_done)
            Memory.append_to_memory(memory, sarsd)

            if done:
                episode_reward.append(reward)
                break

        rewards[0].append(i_episode)
        rewards[1].append(sum(episode_reward)/len(episode_reward))  # Average reward per timestep

        train(target_network, policy_network, memory, batch_size, gamma)  # Update stap deel 1: Train
        if i_episode % copy_episodes == 0 and i_episode > 0:  # Update stap deel 2: Targetnetwork kopieert (deels) policy
            copy_model(target_network, policy_network, tau)

        if i_episode % save_episodes == 0 and i_episode > 0:
            target_network.save_network("../savedmodels/v1_target.h5")
            policy_network.save_network("../savedmodels/v1_policy.h5")

    # Visualize the learning process of the agent
    visualize_learning(rewards)

    env.close()

main()
