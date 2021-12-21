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
    batch_rewards = [sarsd.get_reward() for sarsd in batch]
    for i in range(len(batch)):
        next_state = batch_next_states[i]
        reward = batch_rewards[i]
        qvaluepolicy = policymodel.get_output(next_state)
        bestaction = np.argmax(qvaluepolicy)
        qvaluetarget = policymodel.get_output(next_state)
        bestactionqvalue = qvaluetarget[bestaction]
        target = batch[i].get_reward() + gamma * bestactionqvalue
        tensortarget = qvaluetarget.copy()
        tensortarget[bestaction] = target
        # Voer backpropagation uit
        # Tensorflow: Voorbeeld: Target = 0.5, A* = 2: output = [30,50,20,10], target = [30,50,0.5,10]
        policymodel.train_network(next_state, tensortarget)