import numpy as np
from classes.policy import EpsilonGreedyPolicy

class Agent:
    def __init__(self, env, startstate, policyobject):
        """Initialises an classes"""
        self.state = [startstate, None, 0, None, False]
        self.policy = ()  # Sommige algoritmes hebben geen policies nodig.


    def consult_policy(self):
        """Gets an action to use from the classes's policy."""
        return self.policy.select_action(self.state, self.mazeinfo.terminals, self.mazeinfo.actionspace,
                                         self.mazeinfo.rewards, self.values, self.mazeinfo.discount)
