def policy_best_action_neural_network(state):
    pass


def target_neural_network(state):
    pass


def train(memory):
    """Train the neural networks."""
    gamma = 0.9
    batch = memory.sample()
    batch_next_states = [sars.next_state for sars in batch]
    for i in range(len(batch)):
        best_action = policy_best_action_neural_network(batch_next_states[i])
        targets = target_neural_network(batch_next_states[i])
        target = batch[i].reward + gamma * targets[best_action]


