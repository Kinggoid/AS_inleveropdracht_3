import tensorflow as tensor
import gym

from classes.agent import Agent
from classes.policy import EpsilonGreedyPolicy, SARSd
from classes.neural_network import *
from classes.approximator import BaseNetwork

env = gym.make('LunarLander-v2')
for i_episode in range(20):
    observation = env.reset()
    state_reward, done = 0, 0
    for t in range(100):
        env.render()

        last_observation = observation
        last_reward = state_reward
        last_done = done

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        sarsd = SARSd(last_observation, action, last_reward, observation, last_done)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
env.close()
