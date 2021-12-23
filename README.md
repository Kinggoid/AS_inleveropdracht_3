# AS_inleveropdracht_3

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