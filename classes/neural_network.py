from copy import deepcopy
import numpy as np
import tensorflow as tensor
from functions.helper import prob

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def train_serial(targetmodel, policymodel, memory, batchsize, gamma):
    """Train the approximator neural networks, using two approximators for satisfying
    the double constraint of double q-learning, a memory replay buffer for training,
    a batchsize to determine how many samples are evaluated"""
    x = []  # States go in this set
    y = []  # Targets go in this set (adjusted target from reward, gamma and qvalue from best action from policy in state S')
    size = len(memory.transitions)
    if size < batchsize:  # Fix for code crashing due to memory.sample(size) failing due to size being greater than len(memory)
        batchsize = size
    batch = memory.sample(batchsize)  # Get random SARSD samples
    for i in range(len(batch)):
        sarsd = batch[i]
        if sarsd.done:
            target = sarsd.reward
        else:
            next_state_policies = policymodel.get_output(sarsd.next_state)  # Get qvalues for actions for next state from policy model
            bestaction = np.argmax(next_state_policies)  # Get the best action (int) from previous step

            next_state_targets = targetmodel.get_output(sarsd.next_state)  # Get qvalues for actions for next state from target model
            bestactionqvalue = next_state_targets[bestaction]
            target = sarsd.reward + gamma * bestactionqvalue

        tensortarget = policymodel.get_output(sarsd.state)  # Tensorflow: Vervang de index beste actie met de target van de qvalues van target nn(?)
        tensortarget[sarsd.action] = target
        # Voer backpropagation uit
        # Tensorflow: Voorbeeld: Target = 0.5, A = 2: output policy model = [30,50,20,10], target = [30,50,0.5,10]
        x.append(sarsd.state)
        y.append(tensortarget)
    x = np.array(x)
    y = np.array(y)
    policymodel.train_network(x, y)


def train(targetmodel, policymodel, memory, batchsize, gamma):
    """Train the approximator neural networks."""
    size = len(memory.transitions)
    if size < batchsize:  # If batch size exceeds the memory size
        batchsize = size

    batch = memory.sample(batchsize)  # Sample of SARSd's in the memory

    batch_states = np.array([a.state for a in batch])
    batch_next_states = np.array([d.next_state for d in batch])

    # Policies and targets of the next state
    next_state_policies = policymodel.get_output(batch_next_states)
    next_state_targets = targetmodel.get_output(batch_next_states)

    # The policies of the next state
    state_policies = policymodel.get_output(batch_states)

    for i in range(len(batch)):
        sarsd = batch[i]  # (S, A, R, S', D)
        if sarsd.done:
            target = sarsd.reward  # Q-values horen 0 te zijn, dus alleen reward telt hier.
        else:
            next_state_best_action = np.argmax(next_state_policies[i])  # arg max a' Qp(S', a')
            next_state_best_action_qvalue = next_state_targets[i][next_state_best_action]  # <--- Qt(S', a') (a' in vorige stap)
            target = sarsd.reward * gamma * next_state_best_action_qvalue  # Qp(S,A) = R + y * argmax a' Qt(S', a')

        state_policies[i][sarsd.action] = target

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
