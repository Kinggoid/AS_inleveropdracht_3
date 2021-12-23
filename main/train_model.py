import tensorflow as tensor
import gym

from classes.agent import Agent
from classes.policy import EpsilonGreedyPolicy, SARSd
from classes.neural_network import *
from classes.approximator import Approximator
from classes.memory import Memory

env = gym.make('LunarLander-v2')
memory = Memory(1000)
sample_size = 64
learning_rate = 0.01
policy_object = EpsilonGreedyPolicy()
gamma = 0.9
policy_network = Approximator()
target_network = Approximator()
copy_steps = 10
tau = 0.3

for i_episode in range(5000):
    observation = env.reset()
    state_reward, done = 0, 0
    for t in range(1000):

        last_observation = observation
        last_done = done

        action = policy_object.select_action(last_observation, env.action_space, policy_network, t)
        observation, reward, done, info = env.step(action)

        sarsd = SARSd(last_observation, action, reward, observation, last_done)
        Memory.append_to_memory(memory, sarsd)

        if done:
            print("Episode: {}".format(i_episode))
            print("Episode finished after {} timesteps".format(t+1))
            print("Final reward: {}".format(reward))
            break

    train(target_network, policy_network, memory, sample_size, gamma)
    if i_episode % copy_steps == 0 and i_episode > 0:
        copy_model(target_network, policy_network, tau)

target_network.save_network("../savedmodels/target_network/v1")
policy_network.save_network("../savedmodels/policy_network/v1")

env.close()
