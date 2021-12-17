

def policy_best_action_neural_network(state):
    pass


def target_neural_network(state):
    pass


def train(memory, gamma):
    """Train the approximator neural networks."""
    batch = memory.sample()
    batch_next_states = [sars.next_state for sars in batch]
    for i in range(len(batch)):
        best_action = policy_best_action_neural_network(batch_next_states[i])
        targets = target_neural_network(batch_next_states[i])
        target = batch[i].reward + gamma * targets[best_action]


