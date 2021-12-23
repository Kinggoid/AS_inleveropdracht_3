# AS_inleveropdracht_3

## Agent analyse

The agent shows different types of unwanted behaviour

- Agent starts landing too soon, resulting in it taking off again (sometimes accompanied by
crashing into the ground)
- Agent levitates horizontally but doesn't land
- Agent flies away
- Agent performs a 180 with fatal consequences
- Potentially many more types of unwanted behaviour

Sometimes - very rarely - it manages to land without a fuzz
almost immediately, and sometimes - in these cases - land within
the given zone as well.

We would like our agent to perform better, but we are enthusiastic to see that he is actually learning! He can at least
find out that it is bad to touch the ground and good to stay horizontal. It's just a shame that it's hard for him to
combine all these good efforts. 

If anyone would want to improve our agent, we have the following suggestions:
- There are certain variables defined in main() from train_model.py. We believe that there are certain combinations of
those variables which can significantly improve the agents learning ability. We have just not found the right
combinations yet. 
- We are now training our agent by looking at his actions and judging how good these are. Then he can learn from our
judgements. We judge these actions by looking at the state the agent will enter when performed said action. Then we
look at how good this next state is. What you can do to maybe improve performance, is look even further at the state 
after the next state. This could be interesting to look at.

## Running

To train a model, run `main/train_model.py` 

Do keep in mind that this action will overwrite the models currently saved
in `savedmodels/`. 

The hyperparameters are also contained within this file and may be editted.

To visualise how a model/agent performs, run `visualise_model.py`.

This starts a instance of the environment which runs indefinitely and in which
the model does not learn. It is also visually represented and loops when
the environment is done.

## Classes

### `agent.py`

Unused. Originally intended to work in the environment, but
the OpenAI Gym environment is the agent essentially

### `approximator.py`

Static neural network classes that are used to predict action
qvalues based on state input.

### `neural_network`

Contains `train()` and `copy_model()`, which train a given policy approximator
and target approximator, and merges the policy approximator's weights 
(based on parameter tau) with the target approximator (partially), respectively.

### `memory`

Contains the `Memory` class, which is a deque replaybuffer which can be
used to save `SARSd` dataclass tuples


### `policy`

Contains the `EpsilonGreedyPolicy`, which accepts a `Approximator` object
and can be consulted using variables from the environment during application
(e.g. in `train_model.py` and `visualise_model.py`) to produce a suggested action.
Additionally - based on the current episode - acts as interference; it sometimes
produces another action than the optimal action would be to stimulate exploration.
As the episodes grow, the probability of this event occuring lowers.

### `functions`

Contains helper functions, namely `prob()`, which either returns a `True` or `False`
randomly, based on a given decimal percentage which influences the likelihood of the
function returning `True`.

## Tests

### `approximator_test.py`

Tests the various functionalities of the `Approximator` class, namely
the ability to manipulate weights and biases.