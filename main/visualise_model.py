"""Visualises the policy model from the saved models for evaluation of behavior."""

import gym

from classes.policy import EpsilonGreedyPolicy
from classes.neural_network import *
from classes.approximator import Approximator

def main():
    """Visualisation for saved models, automatically loads the policy model
    from the savedmodels folder"""
    env = gym.make('LunarLander-v2')

    policy_object = EpsilonGreedyPolicy()
    policy_model = Approximator(0.0005)  # Learning rate doesn't matter here, just needs a input.

    policy_model.load_network("../savedmodels/v1_policy.h5")

    i_episode = 1100  # Override epsilon to be minimal

    while True:  # To stop the program you need to kill the proces unfortunately.
        observation = env.reset()
        rewards = []
        for t in range(500):
            env.render()  # Show environment
            action = policy_object.select_action(observation, env.action_space, policy_model, i_episode)
            observation, reward, done, info = env.step(action)
            rewards.append(reward)  # Reward for print statement
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print("End reward: {}".format(np.average(rewards)))
                break

main()
