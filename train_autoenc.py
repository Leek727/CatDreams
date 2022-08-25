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
            #layers.Dense(3072, activation='relu'),
            layers.Dense(32**2, activation='relu'),
            layers.Dense(32**2, activation='relu'),
            #layers.Dropout(.1),
            #layers.Dense(4096, activation='relu', kernel_regularizer='l1'),
            #layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim, activation='sigmoid'),
        ])

        self.decoder = tf.keras.Sequential([
            #layers.Dense(256, activation='relu'),
            #layers.Dense(4096, activation='relu', kernel_regularizer='l1'),
            #layers.Dropout(.1),
            layers.Dense(32**2, activation='relu'),
            layers.Dense(32**2, activation='relu'),
            #layers.Dense(3072, activation='relu'),
            layers.Dense(3*(largest[0]**2), activation='sigmoid'),
            layers.Reshape((largest[0], largest[1], 3))
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
catdream = Autoencoder(64)
if 'y' in load_flag:
    print("loading...")
    catdream.model_load()
    catdream.encoder.summary()
    catdream.decoder.summary()

#Adam( lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1.0, decay = 0.1 )
catdream.compile(optimizer=tf.keras.optimizers.Adam( learning_rate = 0.0001), loss='binary_crossentropy', metrics=['accuracy'],)

catdream.fit(x_train, x_train,
    epochs=10,
    shuffle=True,
    validation_split=.2,
    #batch_size=32
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