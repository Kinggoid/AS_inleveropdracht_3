import tensorflow as tf
import numpy as np


class Approximator:
    """Base class which defines the network"""
    def __init__(self):
        # Base network
        inputlayer = tf.keras.layers.Input(shape=(8,))
        hidden1 = tf.keras.layers.Dense(32)(inputlayer)
        hidden2 = tf.keras.layers.Dense(32)(hidden1)
        outputlayer = tf.keras.layers.Dense(4)(hidden2)
        self.network = tf.keras.models.Model(inputs=inputlayer, outputs=outputlayer)
        self.network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    def get_output(self, inputs):
        """Gives the output of the Approximator neural network based
        on a single or a set of inputs."""
        if isinstance(inputs[0], list) or isinstance(inputs[0], np.ndarray):
            arrayinputs = np.array(inputs).reshape(len(inputs),8)  # https://stackoverflow.com/questions/70362733/input-to-the-neural-network-using-an-array
        else:
            arrayinputs = np.array(inputs).reshape(1,8)
        return self.network.predict(arrayinputs)

    def save_network(self, filepath):
        """Saves the model's assets in a given folder. Make sure the given folder is empty!"""
        return self.network.save(filepath)

    def load_network(self, filepath):
        """Loads a given model from an asset folder populated by save_network(); overwrites current with new model!
        No compiling necessary."""
        self.network = tf.keras.models.load_model(filepath)  # Geen compile nodig: https://www.tensorflow.org/tutorials/keras/save_and_load#savedmodel_format

    def train_network(self, x, y):
        """Trains the network on a given set of X-values (state) and Y-values (Yqt)"""
        self.network.fit(x, y)

    def set_weights_and_bias(self, weights: np.ndarray, bias: np.ndarray, layer: int):
        if weights.shape == self.network.layers[layer].get_weights()[0].shape and bias.shape == self.network.layers[layer].get_weights()[1].shape:
            wandb = [weights, bias]
            self.network.layers[layer].set_weights(wandb)

    def set_weights(self, weights: np.ndarray, layer: int):
        if weights.shape == self.network.layers[layer].get_weights()[0].shape:
            bias = self.get_bias(layer)
            wandb = [weights, bias]
            self.network.layers[layer].set_weights(wandb)

    def get_weights(self, layer):
        """Gets the inputs and the bias of a given layer."""
        if layer <= 0:
            raise Exception("Input layer does not have weights.")
        elif layer < 0 or layer > 3:
            raise Exception("Invalid layer passed as arg (must be 1, 2 or 3)")
        unfiltered = self.network.get_weights()
        index = (layer-1) * 2
        return unfiltered[index]

    def get_bias(self, layer):
        if layer == 0:
            raise Exception("Input layer does not have biases.")
        elif layer < 0 or layer > 3:
            raise Exception("Invalid layer passed as arg (must be 1, 2 or 3)")
        unfiltered = self.network.get_weights()
        index = (layer-1) * 2 + 1
        return unfiltered[index]