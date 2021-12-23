import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_serial(targetmodel, policymodel, memory, batchsize, gamma):
    """Train the approximator neural networks."""
    x = []  # Save the input variables of the SARSd's
    y = []  # Save the targets of the SARSd

    size = len(memory.transitions)
    if size < batchsize:  # Just in case the batchsize is greater than the memory size
        batchsize = size

    batch = memory.sample(batchsize)  # Take a sample from the memory
    for i in range(len(batch)):  # For every SARSd
        sarsd = batch[i]
        if sarsd.done:  # If SARSd is done, return the SARSd reward
            target = sarsd.reward

        else:
            next_state_policies = policymodel.get_output(sarsd.next_state)  # Get qvalues of the next state
            bestaction = np.argmax(next_state_policies)  # Find the best action from this next state

            # Return the same action of the targetmodel of this next state
            next_state_targets = targetmodel.get_output(sarsd.next_state)
            bestactionqvalue = next_state_targets[bestaction]

            # Calculate target
            target = sarsd.reward + gamma * bestactionqvalue

        # Get the outputs of the policymodel of the current state
        tensortarget = policymodel.get_output(sarsd.state)

        # Change the qvalue of the action of this policynetwork with the target
        tensortarget[sarsd.action] = target

        x.append(sarsd.state)
        y.append(tensortarget)

    x = np.array(x)
    y = np.array(y)

    # Train the policymodel
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
        sarsd = batch[i]
        if sarsd.done:
            target = sarsd.reward  # Q-values horen 0 te zijn, dus alleen reward telt hier.
        else:
            next_state_best_action = np.argmax(next_state_policies[i])  # arg max a' Qp(S', a')
            next_state_best_action_qvalue = next_state_targets[i][next_state_best_action]  # <--- Qt(S', a') (a' in vorige stap)
            target = sarsd.reward * gamma * next_state_best_action_qvalue  # Qp(S,A) = R + y * argmax a' Qt(S', a')

        # Change the qvalue of the action of this policynetwork with the target
        state_policies[i][sarsd.action] = target

    # Train the policymodel
    policymodel.train_network(batch_states, state_policies)


def copy_model(targetmodel, policymodel, tau):
    """We will partly copy and past the weights of the policymodel to the targetmodel."""
    for layer in range(1, targetmodel.layers):  # For every layer
        # Get the weights of the policy and target weights and biases
        policy_weights = policymodel.get_weights(layer)
        target_weights = targetmodel.get_weights(layer)
        policy_bias = policymodel.get_bias(layer)
        target_bias = targetmodel.get_bias(layer)

        for x, y in np.ndindex(policy_weights.shape):
            # Combine all the weigths (not bias). Tau dictates how much every weight counts in the calculation
            target_weights[x][y] = tau * policy_weights[x][y] + (1 - tau) * target_weights[x][y]

        for x in np.ndindex(policy_bias.shape):
            # Combine all the bias weigths. Tau dictates how much every weight counts in the calculation
            target_bias[x] = tau * policy_bias[x] + (1 - tau) * target_bias[x]

        # Change the weights of the targetmodel
        targetmodel.set_weights_and_bias(target_weights, target_bias, layer)
