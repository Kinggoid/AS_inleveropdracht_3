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

    memory = Memory(10000)
    episodes = 5000
    batch_size = 32
    learning_rate = 0.001
    gamma = 0.9
    tau = 0.01
    copy_episodes = 5

    save_episodes = 100

    policy_object = EpsilonGreedyPolicy()
    policy_network = Approximator(learning_rate)
    target_network = Approximator(learning_rate)

    rewards = [[], []]

    for i_episode in range(episodes):
        observation = env.reset()
        state_reward, done = 0, 0
        episode_reward = []
        for t in range(1000):
            last_observation = observation
            last_done = done

            action = policy_object.select_action(last_observation, env.action_space, policy_network, i_episode)
            observation, reward, done, info = env.step(action)

            episode_reward.append(reward)

            sarsd = SARSd(last_observation, action, reward, observation, last_done)
            Memory.append_to_memory(memory, sarsd)

            if done:
                episode_reward.append(reward)
                break

        rewards[0].append(i_episode)
        rewards[1].append(sum(episode_reward)/len(episode_reward))

        train(target_network, policy_network, memory, batch_size, gamma)  # Update stap deel 1: Train
        if i_episode % copy_episodes == 0 and i_episode > 0:  # Update stap deel 2: Targetnetwork kopieert (deels) policy
            copy_model(target_network, policy_network, tau)

        if i_episode % save_episodes == 0 and i_episode > 0:
            target_network.save_network("../savedmodels/target_network/v1.h5")
            policy_network.save_network("../savedmodels/policy_network/v1.h5")

    visualize_learning(rewards)

    env.close()

main()
