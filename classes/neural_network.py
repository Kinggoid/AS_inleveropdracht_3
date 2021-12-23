from copy import deepcopy
import numpy as np
import tensorflow as tensor
from functions.helper import prob


def train(targetmodel, policymodel, memory, batchsize, gamma):
    """Train the approximator neural networks."""
    x = []
    y = []
    batch = memory.sample(batchsize)
    for i in range(len(batch)):
        state = batch[i]
        if state.done:
            target = state.reward
        else:
            next_state_policies = policymodel.get_output(state.next_state)
            bestaction = np.argmax(next_state_policies)

            next_state_targets = targetmodel.get_output(state.next_state)
            bestactionqvalue = next_state_targets[bestaction]
            target = state.reward + gamma * bestactionqvalue

        tensortarget = policymodel.get_output(state.state)  # Tensorflow: Vervang de index beste actie met de target van de qvalues van target nn(?)
        tensortarget[state.action] = target
        # Voer backpropagation uit
        # Tensorflow: Voorbeeld: Target = 0.5, A* = 2: output = [30,50,20,10], target = [30,50,0.5,10]
        x.append(state.next_state)
        y.append(tensortarget)
    x = np.array(x)
    y = np.array(y)
    policymodel.train_network(x, y)


def copy_model(targetmodel, policymodel, tau):
    """We will partly copy and past the weights of the policymodel to the targetmodel."""

    for layer in range(1, targetmodel.layers):
        policy_weights = policymodel.get_weights(layer)
        target_weights = targetmodel.get_weights(layer)
        policy_bias = policymodel.get_bias(layer)
        target_bias = targetmodel.get_bias(layer)
        for y, x in np.ndindex(policy_weights.shape):
            target_weights[y][x] = tau * policy_weights[y][x] + (1 - tau) * target_weights[y][x]
        for x in np.ndindex(policy_bias.shape):
            target_bias[x] = tau * policy_bias[x] + (1 - tau) * target_bias[x]
        targetmodel.set_weights_and_bias(target_weights, target_bias, layer)
    # return targetmodel  # Return is niet nodig, model wordt in-place geupdated.
