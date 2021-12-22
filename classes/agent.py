import numpy as np
from classes.policy import EpsilonGreedyPolicy

class Agent:
    def __init__(self, env, startstate, policyobject):
        """Initialises an classes"""
        self.state = [startstate, None, 0, None, False]
        self.env = env
        self.policy = policyobject()  # Sommige algoritmes hebben geen policies nodig.

    def consult_policy(self):
        """Gets an action to use from the classes's policy."""
        return self.policy.select_action(self.state[0], self.env.action_space,)
