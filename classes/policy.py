import numpy as np
import random
from functions.helper import prob

from dataclasses import dataclass


class EpsilonGreedyPolicy:
    """Initialises a greedy policy, which always takes the action
    that results in the highest value according to the value_function
    of the linked agent that dictates the value for the future state and
    the reward of the future state"""
    def __init__(self):
        pass

    def select_action(self, state, actions, model, timestep):
        if prob(self.epsilon_decay(timestep)):
            possibleactions = [x for x in range(len(actions))]
            policyaction = random.choice(possibleactions)
        else:
            policyaction = np.argmax(model.get_output(state))
        return policyaction

    def epsilon_decay(self, x):
        # e *= disc**iters
        if x < 10:
            return 0.6
        elif 10 < x < 50:
            return 0.5
        elif 50 < x < 100:
            return 0.4
        elif x > 100:
            return 0.3


@dataclass
class SARSd:
    """Transition class with SARSd object."""
    def __init__(self, state, action, reward, next_state, done):
        self.sars = (state, action, reward, next_state, done)

    def get_state(self):
        return self.sars[0]

    def get_action(self):
        return self.sars[1]

    def get_reward(self):
        return self.sars[2]

    def get_next_state(self):
        return self.sars[3]

    def get_done(self):
        return self.sars[4]
