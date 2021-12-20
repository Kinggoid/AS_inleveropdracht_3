from copy import deepcopy
import numpy as np
import tensorflow as tensor


def get_state_values(environment):
    # x = float
    # y = float
    # x_vol = float
    # y_vol = float
    # angle = float
    # angle_vol = float
    # right_leg = bool: int
    # left_leg = bool: int
    return deepcopy(environment.env)


# def get_values_neural_network(model, state):
#     q_values = model.predict(state)
#     return q_values
# DEPRECATED: Hier is al functionaliteit voor vanuit approximator.py


def train(targetmodel, policymodel, memory, batchsize, gamma, actions):
    """Train the approximator neural networks."""
    actions = {'fire_right_engine': '→', 'fire_left_engine': '←', 'fire_main_engine': '↓', 'nothing': '0'}

    batch = memory.sample(batchsize)
    batch_next_states = [sarsd.get_next_state() for sarsd in batch]
    rewards = [sarsd.get_reward() for sarsd in batch]
    for i in range(len(batch)):
        qvaluepolicy = policymodel.get_output(batch_next_states[i])
        bestaction = np.argmax(qvaluepolicy)
        qvaluetarget = policymodel.get_output(batch_next_states[i])
        bestactionqvalue = qvaluetarget[bestaction]
        # targets = get_values_neural_network(model, batch_next_states[i])  #
        target = batch[i].get_reward() + gamma * bestactionqvalue



