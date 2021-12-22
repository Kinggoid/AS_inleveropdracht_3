from classes.approximator import BaseNetwork
import unittest
import numpy as np
import tensorflow as tf

class WeightManipulation(unittest.TestCase):

    def setUp(self):
        """Sets up the environment for testing every time."""
        self.test = BaseNetwork()
        x = np.random.random((100, 8))
        y = np.random.random((100, 4))
        self.test.train_network(x, y)

    def testweightget(self):
        """Tests if weights are obtained properly."""
        layer = 1
        sample = np.array(self.test.network.layers[layer].weights[0])
        methodoutput = self.test.get_weights(layer)
        np.testing.assert_almost_equal(sample, methodoutput)

    def testweightandbiasset(self):
        layer = 1
        randomweights = np.random.random(self.test.get_weights(layer).shape)
        randombias = np.random.random(self.test.get_bias(layer).shape)
        self.test.set_weights(randomweights, randombias, layer)
        obtainedweights = self.test.get_weights(layer)
        np.testing.assert_almost_equal(randomweights, obtainedweights)
