from copy import deepcopy
import numpy as np
import tensorflow as tensor
from functions.helper import prob

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def train_serial(targetmodel, policymodel, memory, batchsize, gamma):
    """Train the approximator neural networks."""
    x = []
    y = []
    size = len(memory.transitions)
    if size < batchsize:
        batchsize = size
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
        # Tensorflow: Voorbeeld: Target = 0.5, A* = 2: output policy model = [30,50,20,10], target = [30,50,0.5,10]
        x.append(state.state)
        y.append(tensortarget)
    x = np.array(x)
    y = np.array(y)
    policymodel.train_network(x, y)


def train(targetmodel, policymodel, memory, batchsize, gamma):
    """Train the approximator neural networks."""
    size = len(memory.transitions)
    if size < batchsize:
        batchsize = size

    batch = memory.sample(batchsize)

    batch_states = np.array([a.state for a in batch])
    batch_actions = np.array([b.action for b in batch])
    batch_rewards = np.array([c.reward for c in batch])
    batch_next_states = np.array([d.next_state for d in batch])
    batch_done = np.array([e.done for e in batch])

    next_state_policies = policymodel.get_output(batch_next_states)
    next_state_targets = targetmodel.get_output(batch_next_states)
    state_policies = policymodel.get_output(batch_states)
    test = state_policies.copy()

    for i in range(len(batch)):
        if batch_done[i]:
            target = batch_rewards[i]  # Q-values horen 0 te zijn, dus alleen reward telt hier.
        else:
            next_state_best_action = np.argmax(next_state_policies[i])
            next_state_best_action_qvalue = next_state_targets[i][next_state_best_action]
            target = batch_rewards[i] * gamma * next_state_best_action_qvalue  # Qp(S,A) = R + y* argmax a' Qt(S', a')

        state_policies[i][batch_actions[i]] = target

    print()
    policymodel.train_network(batch_states, state_policies)


def copy_model(targetmodel, policymodel, tau):
    """We will partly copy and past the weights of the policymodel to the targetmodel."""

    for layer in range(1, targetmodel.layers):
        policy_weights = policymodel.get_weights(layer)
        target_weights = targetmodel.get_weights(layer)
        policy_bias = policymodel.get_bias(layer)
        target_bias = targetmodel.get_bias(layer)
        for x, y in np.ndindex(policy_weights.shape):
            target_weights[x][y] = tau * policy_weights[x][y] + (1 - tau) * target_weights[x][y]
        for x in np.ndindex(policy_bias.shape):
            target_bias[x] = tau * policy_bias[x] + (1 - tau) * target_bias[x]
        targetmodel.set_weights_and_bias(target_weights, target_bias, layer)
