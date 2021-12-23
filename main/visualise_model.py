import gym

from classes.policy import EpsilonGreedyPolicy, SARSd
from classes.neural_network import *
from classes.approximator import Approximator
from classes.memory import Memory

from datetime import datetime

env = gym.make('LunarLander-v2')

policy_object = EpsilonGreedyPolicy()
policy_model = Approximator(0.0005)

policy_model.load_network("../savedmodels/policy_network/v1")


i_episode = 1100


while True:  # Termineer programma om loop te breken
    observation = env.reset()
    for t in range(500):
        env.render()
        action = policy_object.select_action(observation, env.action_space, policy_model, i_episode)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
