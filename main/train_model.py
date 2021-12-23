import tensorflow as tensor
import gym

from classes.agent import Agent
from classes.policy import EpsilonGreedyPolicy, SARSd
from classes.neural_network import *
from classes.approximator import Approximator
from classes.memory import Memory
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_learning(rewards):
    ax = sns.lineplot(rewards[0], rewards[1])
    ax.set(xlabel='Days', ylabel='Amount_spend')

    # giving title to the plot
    plt.title('My first graph')

    # function to show plot
    plt.show()

def main():
    env = gym.make('LunarLander-v2')
    memory = Memory(10000)
    sample_size = 64
    learning_rate = 0.0005
    policy_object = EpsilonGreedyPolicy()
    gamma = 0.9
    policy_network = Approximator()
    target_network = Approximator()
    copy_steps = 4
    tau = 0.001
    rewards = [[], []]

    for i_episode in range(5000):
        observation = env.reset()
        state_reward, done = 0, 0
        episode_reward = 0

        for t in range(1000):
            last_observation = observation
            last_done = done

            action = policy_object.select_action(last_observation, env.action_space, policy_network, i_episode)
            observation, reward, done, info = env.step(action)
            episode_reward += reward

            sarsd = SARSd(last_observation, action, reward, observation, last_done)
            Memory.append_to_memory(memory, sarsd)

            if done:
                print("Episode: {}".format(i_episode))
                print("Episode finished after {} timesteps".format(t+1))
                print("Total episode reward: {}".format(episode_reward))
                break
        rewards[0].append(i_episode)
        rewards[1].append(episode_reward)

        train(target_network, policy_network, memory, sample_size, gamma)
        if i_episode % copy_steps == 0 and i_episode > 0:
            copy_model(target_network, policy_network, tau)

    visualize_learning(rewards)

    target_network.save_network("../savedmodels/target_network/v1")
    policy_network.save_network("../savedmodels/policy_network/v1")

    env.close()


main()
