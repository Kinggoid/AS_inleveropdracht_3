import tensorflow as tf
import keras.backend as k
import numpy as np

class BaseNetwork:
    """Base class which defines the network"""
    def __init__(self):
        inputlayer = tf.keras.layers.Input(shape=(8,))
        hidden1 = tf.keras.layers.Dense(32)(inputlayer)
        hidden2 = tf.keras.layers.Dense(32)(hidden1)
        outputlayer = tf.keras.layers.Dense(4)(hidden2)
        self.network = tf.keras.models.Model(inputs=inputlayer, outputs=outputlayer)
        self.network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    def set_weights_manual(self):
        pass

    def get_weights(self):
        weights = []
        for layer in range(1,4):
            weights.append(self.network.layers[layer].get_weights())

base = BaseNetwork()
x = np.random.random((100,8))
y = np.random.random((100,4))
base.network.fit(x,y)
print(base.network.layers[1].get_weights())
# Layer 0 is input, geen weights.
# Query op layers 1 t/m 3.
