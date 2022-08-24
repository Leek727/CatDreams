import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model
from get_data import x_train, largest

# inherit from tf Model class
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(2048, activation='relu'),
            #layers.Dense(784, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            #layers.Dense(784, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(largest[0]**2, activation='sigmoid'),
            layers.Reshape((largest[0], largest[1]))
        ])
    
    # needed for tensorflow model.call
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def save_model(self):
        self.decoder.save("catdream.h5")
        self.encoder.save("encoder.h5")


catdream = Autoencoder(4)

catdream.compile(optimizer=tf.keras.optimizers.Adam(lr=.001), loss='mse')

catdream.fit(x_train, x_train,
    epochs=50,
    shuffle=True,
    #validation_split=.2,
    batch_size=128

)

catdream.save_model()