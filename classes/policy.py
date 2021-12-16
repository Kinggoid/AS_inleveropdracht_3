class EpsilonGreedyPolicy:
    """Initialises a greedy policy, which always takes the action
    that results in the highest value according to the value_function
    of the linked agent that dictates the value for the future state and
    the reward of the future state"""
    def __init__(self):
        pass

    def select_action(self, state, actions, model, epsilon):
        bestaction = model.predict(state)
        if

    def epsilon_decay(self, x):
        pass  # TODO: Wiskunde formule opzoeken, misschien 1/x?