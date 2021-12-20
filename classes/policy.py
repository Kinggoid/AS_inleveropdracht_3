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

    def select_action(self, state, actions, model, epsilon):
        if prob(epsilon):
            possibleactions = [x for x in range(len(actions))]
            policyaction = random.choice(possibleactions)
        else:
            policyaction = np.argmax(model.predict(state))
        return policyaction

    def epsilon_decay(self, x):
        # e *= disc**iters
        pass  # TODO: Wiskunde formule opzoeken, misschien 1/x?

@dataclass
class SARSd:
    """Transition class with SARSd object."""
    def __init__(self, state, action, reward, next_state, done):
        self.sars = (state, action, reward, next_state, done)
