import gym

from classes.policy import EpsilonGreedyPolicy, SARSd
from classes.neural_network import *
from classes.approximator import Approximator
from classes.memory import Memory
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_learning(rewards):
    ax = sns.lineplot(rewards[0], rewards[1])
    ax.set(xlabel='Timesteps', ylabel='Rewards')
    plt.title('Average reward per timestep')

    plt.show()


def main():
    env = gym.make('LunarLander-v2')
    memory = Memory(10000)
    batch_size = 64
    learning_rate = 0.0005
    policy_object = EpsilonGreedyPolicy()
    gamma = 0.9
    policy_network = Approximator()
    target_network = Approximator()
    copy_episodes = 4
    save_episodes = 100
    tau = 0.001
    rewards = [[], []]

    # LOGGING
    episode_reward_total = 0
    timestart = datetime.now()

    for i_episode in range(5):
        observation = env.reset()
        state_reward, done = 0, 0
        episode_reward = 0
        timeend = datetime.now()
        print("Iter took {}".format(timeend - timestart))
        timestart = datetime.now()
        for t in range(1000):

            last_observation = observation
            last_done = done

            action = policy_object.select_action(last_observation, env.action_space, policy_network, i_episode)
            observation, reward, done, info = env.step(action)

            episode_reward += reward

            sarsd = SARSd(last_observation, action, reward, observation, last_done)
            Memory.append_to_memory(memory, sarsd)

            if done:
                end = datetime.now()
                episode_reward_total += episode_reward
                print("Episode: {}".format(i_episode))
                print("Episode finished after {} timesteps".format(t+1))
                print("Total episode reward: {}".format(episode_reward))
                print("Average reward over all episodes: {}".format((episode_reward_total/(i_episode+1))))
                break

        rewards[0].append(i_episode)
        rewards[1].append(episode_reward)

        train(target_network, policy_network, memory, batch_size, gamma)
        if i_episode % copy_episodes == 0 and i_episode > 0:
            copy_model(target_network, policy_network, tau)

        if i_episode % save_episodes == 0 and i_episode > 0:
            target_network.save_network("../savedmodels/target_network/v1")
            policy_network.save_network("../savedmodels/policy_network/v1")

    visualize_learning(rewards)

    env.close()

main()
