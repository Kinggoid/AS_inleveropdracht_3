import random


def prob(chance):
    """Gives back either True or False randomly, based on input;
    input (must be between 0 and 1, decimally) determines chance (0 = 0 %, 1 = 100 %)
    to return True, otherwise False."""
    randomnum = random.uniform(0,1)
    if randomnum < chance:
        return True
    else:
        return False