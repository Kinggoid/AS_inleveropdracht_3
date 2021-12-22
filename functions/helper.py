import random


def prob(chance):
    randomnum = random.uniform(0,1)
    if randomnum < chance:
        return True
    else:
        return False