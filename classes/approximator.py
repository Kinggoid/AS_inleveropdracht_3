import tensorflow as tf
import numpy as np

class BaseNetwork:
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
        """Gives the output of the approximator neural network based
        on a single or a set of inputs."""
        if isinstance(inputs[0],list):
            arrayinputs = np.array(inputs).reshape(len(inputs),8)  # https://stackoverflow.com/questions/70362733/input-to-the-neural-network-using-an-array
        else:
            arrayinputs = np.array(inputs).reshape(1,8)
        return self.network.predict(arrayinputs)

    def save_network(self):
        """Returns the current model"""
        return self.network

    def load_network(self, loadnetwork):
        """Loads a given model; overwrites current with new model!"""
        self.network = loadnetwork

    def train_network(self, x, y):
        """Trains the network on a given set of X-values (state) and Y-values (Yqt)"""
        if isinstance(x[0],list):  # X-set prep
            arrayinputs = np.array(x).reshape(len(x),8)
        else:
            arrayinputs = np.array(x).reshape(1,8)

        if isinstance(y[0],list):  # Y-set prep
            arrayinputs = np.array(y).reshape(len(y),8)
        else:
            arrayinputs = np.array(y).reshape(1,8)

        self.network.fit(x, y)

    def set_weights(self, wandb: list, layer: int):
        """Adjusts the weights of the given layer."""
        temp = np.array(wandb)
        if temp.shape == self.network.layer[layer].shape:
            self.network.layer[layer].set_weights(wandb)  # TODO: Test of dit werkt in context! (Komt later)

    def get_weights(self):
        weights = []
        for layer in range(1,4):
            weights.append(np.array(self.network.layers[layer].get_weights()))
        return weights