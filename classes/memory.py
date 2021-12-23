from collections import deque
import random
from dataclasses import dataclass


class Memory:
    def __init__(self, size):
        """Create a deque."""
        self.transitions = deque([], size)

    def sample(self, batch_size):
        """Take a sample of the deque."""
        return random.sample(self.transitions, batch_size)

    def append_to_memory(self, new_SARS):
        """Append a new SARSd to memory and remove the oldest memory."""
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

