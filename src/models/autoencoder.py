import numpy as np
import tensorflow as tf
from typing import List, Tuple

class Encoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_layers=(64, 32, 10), dropout_rate=0.0, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)      
        self.dropout_rate = dropout_rate

        ### Layers creation (dense + dropout * X, + latent layer)
        self.dense_layers = [
            tf.keras.layers.Dense(units, activation=activation)
            for units in self.hidden_layers[:-1] 
        ]
        self.dropout_layers = (
            [tf.keras.layers.Dropout(dropout_rate) for _ in self.dense_layers]
            if dropout_rate > 0 
            else []
        )
        self.latent_layer = tf.keras.layers.Dense(self.hidden_layers[-1], activation=None)

    def call(self, x, training=False):
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            if self.dropout_layers:
                x = self.dropout_layers[i](x, training=training)
        return self.latent_layer(x)


class Decoder(tf.keras.Model):
    def __init__(self, encoder: Encoder, activation='relu', output_activation=None):
        super(Decoder, self).__init__()
        self.hidden_layers = encoder.hidden_layers[-2::-1] # Reverse order of the encoder. No last layer (latent space)
        self.activation = activation

        ### Layers creation (dense + output layer)
        self.dense_layers = [
            tf.keras.layers.Dense(units, activation=self.activation) 
            for units in self.hidden_layers
        ]
        self.output_layer = tf.keras.layers.Dense(encoder.input_dim, activation=output_activation)

    def call(self, x):
        for dense in self.dense_layers:
            x = dense(x)
        x = self.output_layer(x)
        return x

class AutoEncoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_layers= (64, 32, 10), dropout_rate=0.0, activation='relu', output_activation=None, random_state=None):
        if random_state is not None:
            tf.keras.utils.set_random_seed(random_state)
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_layers, dropout_rate, activation)
        self.decoder = Decoder(self.encoder, activation, output_activation)

    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded)
        return encoded, decoded