from collections import deque
import random


class Memory:
    def __init__(self, size):
        self.size = size
        self.transitions = deque([])

    def sample(self, batch_size):
        return random.sample(self.transitions, batch_size)

    def remove_latests_memory(self):
        self.transitions.popleft()

    def append_to_memory(self, new_SARS):
        self.remove_latests_memory()
        self.transities.append(new_SARS)

