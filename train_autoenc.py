import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model, load_model
from get_data import x_train, largest

# inherit from tf Model class
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(2048, activation='relu'),
            #layers.Dense(1024, activation='relu'),
            #layers.Dense(512, activation='relu'),
            #layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            #layers.Dense(256, activation='relu'),
            #layers.Dense(512, activation='relu'),
            #layers.Dense(1024, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(largest[0]**2, activation='sigmoid'),
            layers.Reshape((largest[0], largest[1]))
        ])
    
    # needed for tensorflow model.call
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def model_save(self):
        self.decoder.save("catdream.h5")
        self.encoder.save("encoder.h5")

    def model_load(self):
        self.encoder = load_model("encoder.h5")
        self.decoder = load_model("catdream.h5")

# create model / load old model
load_flag = input("Load old model (y/n)? : ").strip()
catdream = Autoencoder(50)
if 'y' in load_flag:
    print("loading...")
    catdream.model_load()

catdream.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =.0001), loss='binary_crossentropy', metrics=['accuracy'],)

catdream.fit(x_train, x_train,
    epochs=5,
    shuffle=True,
    validation_split=.2,
    batch_size=32

)

catdream.model_save()


"""
catdream.compile(optimizer=tf.keras.optimizers.Adam(lr=.0001), loss='binary_crossentropy', metrics=['accuracy'],)

catdream.fit(x_train, x_train,
    epochs=5,
    shuffle=True,
    validation_split=.2,
    batch_size=32

)
"""