from copy import deepcopy
import numpy as np
import tensorflow as tensor


def train(targetmodel, policymodel, memory, batchsize, gamma):
    """Train the approximator neural networks."""

    batch = memory.sample(batchsize)
    for i in range(len(batch)):
        state = batch[i]
        next_state = state.next_state
        if state.done:
            target = state.reward
        else:
            next_state_policies = policymodel.get_output(next_state.state)
            bestaction = np.argmax(next_state_policies)

            next_state_targets = targetmodel.get_output(next_state.state)
            bestactionqvalue = next_state_targets[bestaction]
            target = state.reward + gamma * bestactionqvalue

        tensortarget = policymodel.get_output(state.state)  # Tensorflow: Vervang de index beste actie met de target van de qvalues van target nn(?)
        tensortarget[state.action] = target
        # Voer backpropagation uit
        # Tensorflow: Voorbeeld: Target = 0.5, A* = 2: output = [30,50,20,10], target = [30,50,0.5,10]
        policymodel.train_network(next_state.state, tensortarget)
    return policymodel

def copy_model(targetmodel, policymodel, tau):
    weightstargetmodel = targetmodel.get_weights()
    weightspolicymodel = policymodel.get_weights()