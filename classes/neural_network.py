from copy import deepcopy
import numpy as np
import tensorflow as tensor


def fit(state, target):
    """"""


def train(targetmodel, policymodel, memory, batchsize, gamma, actions):
    """Train the approximator neural networks."""
    actions = {'fire_right_engine': '→', 'fire_left_engine': '←', 'fire_main_engine': '↓', 'nothing': '0'}

    batch = memory.sample(batchsize)
    # batch_current_states = [sarsd.get_state() for sarsd in batch]
    # batch_actions = [sarsd.get_action() for sarsd in batch]
    # batch_next_states = [sarsd.get_next_state() for sarsd in batch]
    # batch_rewards = [sarsd.get_reward() for sarsd in batch]
    # batch_done = [sarsd.get_done() for sarsd in batch]
    for i in range(len(batch)):
        state = batch[i]
        next_state = state.next_state
        if state.done:
            target = state.reward
        else:
            next_state_policies = policymodel.get_output(next_state)
            bestaction = np.argmax(next_state_policies)

            next_state_targets = targetmodel.get_output(next_state)
            bestactionqvalue = next_state_targets[bestaction]
            target = state.reward + gamma * bestactionqvalue

        tensortarget = policymodel.get_output(state)  # Tensorflow: Vervang de index beste actie met de target van de qvalues van target nn(?)
        tensortarget[state.action] = target
        # Voer backpropagation uit
        # Tensorflow: Voorbeeld: Target = 0.5, A* = 2: output = [30,50,20,10], target = [30,50,0.5,10]
        policymodel.train_network(next_state, tensortarget)
    return policymodel

def copy_model(targetmodel, policymodel, tau):
    weightstargetmodel = targetmodel.get_weights()
    weightspolicymodel = policymodel.get_weights()