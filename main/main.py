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

for i_episode in range(5000):
    observation = env.reset()
    state_reward, done = 0, 0
    for t in range(1000):
        env.render()

        last_observation = observation
        last_done = done

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        sarsd = SARSd(last_observation, action, reward, observation, last_done)
        Memory.append_to_memory(memory, sarsd)





        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
env.close()
