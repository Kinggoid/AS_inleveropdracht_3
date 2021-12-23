from collections import deque
import random
from dataclasses import dataclass


class Memory:
    def __init__(self, size):
        self.transitions = deque([], size)

    def sample(self, batch_size):
        return random.sample(self.transitions, batch_size)

    def append_to_memory(self, new_SARS):
        self.transitions.append(new_SARS)


@dataclass
class SARSd:
    """Transition class with SARSd object."""
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

