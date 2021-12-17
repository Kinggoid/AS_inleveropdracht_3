from copy import deepcopy
import tensorflow as tensor


def get_state_values(environment):
    return deepcopy(environment.env)


def get_values_neural_network(model, state):
    # x =
    # y =
    # x_vol =
    # y_vol =
    # angle =
    # angle_vol =
    # right_leg =
    # left_leg =
    q_values = model.predict(state)
    return q_values


def train(model, memory, gamma):
    """Train the approximator neural networks."""
    batch = memory.sample()
    batch_next_states = [sars.next_state for sars in batch]
    for i in range(len(batch)):
        best_action = get_values_neural_network(model, batch_next_states[i])
        targets = get_values_neural_network(model, batch_next_states[i])
        target = batch[i].reward + gamma * targets[best_action]


