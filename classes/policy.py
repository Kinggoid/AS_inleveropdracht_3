import numpy as np
import random

from functions.helper import prob


class EpsilonGreedyPolicy:
    """Initialises a greedy policy, which always takes the action
    that results in the highest value according to the value_function
    of the linked agent that dictates the value for the future state and
    the reward of the future state"""
    def __init__(self):
        pass

    def select_action(self, state, actions, model, episode):
        """Selects an action based on current state (input for policy model),
        list of actions (env.action_space as input), model (Approximator class for policy)
        and the current episode (for epsilon_decay)"""
        if prob(self.epsilon_decay(episode)):  # Epsilon probability chance to pick a random action
            policyaction = actions.sample()
        else:
            policyaction = np.argmax(model.get_output(state))
        return policyaction

    def epsilon_decay(self, x):
        """Epsilon decay method which returns an epsilon for a episode.
        Designed to have relatively high epsilon at start to stimulate exploring
        and finding optimal state action sqeuences, later on reduces epsilon to
        converge to one of the optimal solutions."""
        if x <= 100:
            return 0.6
        elif 100 < x <= 200:
            return 0.5
        elif 200 < x <= 300:
            return 0.45
        elif 300 < x <= 400:
            return 0.4
        elif 400 < x <= 500:
            return 0.35
        elif 500 < x <= 600:
            return 0.3
        elif 600 < x <= 700:
            return 0.25
        elif 700 < x <= 800:
            return 0.2
        elif 800 < x <= 900:
            return 0.15
        elif 900 < x <= 1000:
            return 0.12
        elif 1000 < x:
            return 0.1
