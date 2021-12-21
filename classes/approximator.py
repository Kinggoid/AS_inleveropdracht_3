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
        # if isinstance(x[0],list) or isinstance(x[0], np.ndarray):  # X-set prep
        #     arrayinputs = np.array(x).reshape(len(x),8)
        # else:
        #     arrayinputs = np.array(x).reshape(1,8)
        #
        # if isinstance(y[0],list) or isinstance(y[0], np.ndarray):  # Y-set prep
        #     arrayinputs = np.array(y).reshape(len(y),8)
        # else:
        #     arrayinputs = np.array(y).reshape(1,8)

        self.network.fit(x, y)

    def set_weights(self, weights: list, biases: list, layer: int):
        """Adjusts the weights of the given layer."""
        tempw = np.array(weights)
        tempb = np.array(biases)
        if tempw.shape == self.network.layer[layer].weights[0].shape:
            self.network.layer[layer].set_weights(wandb)  # TODO: Test of dit werkt in context! (Komt later)

    def get_weights(self, layer):
        """Gets the inputs and the bias of a given layer."""

        # weights = np.array(self.network.layers[layer].weights)[0]
        # biases = np.array(self.network.layers[layer].weights)[-1]
        return weights, biases

test = BaseNetwork()
inp = np.random.random((100,8))
out = np.random.random((100,4))
test.train_network(inp,out)
print("Weights")
print(test.get_weights(1))
print("Biases")
print(test.get_weights(1))
