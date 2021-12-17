import tensorflow as tf
import keras.backend as k
import numpy

class BaseNetwork:
    """Base class which defines the network"""
    def __init__(self):
        inputlayer = tf.keras.layers.Input(shape=(8,))
        hidden1 = tf.keras.layers.Dense(32)(inputlayer)
        hidden2 = tf.keras.layers.Dense(32)(hidden1)
        outputlayer = tf.keras.layers.Dense(4)(hidden2)
        self.network = tf.keras.models.Model(inputs=inputlayer, outputs=outputlayer)
        self.network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")


print(BaseNetwork().network.layers[0].get_weights())
