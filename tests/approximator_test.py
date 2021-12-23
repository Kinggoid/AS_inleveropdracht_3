from classes.approximator import Approximator
import unittest
import numpy as np
import tensorflow as tf

class WeightBiasManipulation(unittest.TestCase):

    def setUp(self):
        """Sets up the environment for testing every time."""
        self.test = Approximator(0.0005)
        x = np.random.random((100, 8))
        y = np.random.random((100, 4))
        self.test.train_network(x, y)

    def testweightget(self):
        """Tests if weights can be obtained using class method
        WARNING: ONLY WORKS IF DISABLE_V2_BEHAVIOR() ISNT USED"""
        layer = 1
        sample = np.array(self.test.network.layers[layer].weights[0])
        methodoutput = self.test.get_weights(layer)
        np.testing.assert_almost_equal(sample, methodoutput)

    def testbiasget(self):
        """Tests if bias can be obtained using class method
        WARNING: ONLY WORKS IF DISABLE_V2_BEHAVIOR() ISNT USED"""
        layer = 1
        sample = np.array(self.test.network.layers[layer].weights[1])
        methodoutput = self.test.get_bias(layer)
        np.testing.assert_almost_equal(sample, methodoutput)

    def testweightsset(self):
        """Tests if weights can be set using class method"""
        layer = 1
        randomweights = np.random.random(self.test.get_weights(layer).shape)
        randombias = np.random.random(self.test.get_bias(layer).shape)
        self.test.set_weights(randomweights, layer)
        obtainedweights = self.test.get_weights(layer)
        np.testing.assert_almost_equal(randomweights, obtainedweights)

    def testweightandbiasset(self):
        """Tests if weights and bias can be set simultaneously using class method"""
        layer = 1
        randomweights = np.random.random(self.test.get_weights(layer).shape)
        randombias = np.random.random(self.test.get_bias(layer).shape)
        self.test.set_weights_and_bias(randomweights, randombias, layer)
        obtainedweights = self.test.get_weights(layer)
        obtainedbias = self.test.get_bias(layer)
        np.testing.assert_almost_equal(randomweights, obtainedweights)
        np.testing.assert_almost_equal(randombias, obtainedbias)
