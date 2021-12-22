from collections import deque
import random


class Memory:
    def __init__(self, size):
        self.transitions = deque([], size)

    def sample(self, batch_size):
        return random.sample(self.transitions, batch_size)

    def append_to_memory(self, new_SARS):
        self.transitions.append(new_SARS)

