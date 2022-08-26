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
            layers.InputLayer(input_shape=(64, 64, 3)),
            layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu', use_bias=False),
            layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu', use_bias=False),
            
            layers.Flatten(),
            layers.Dense(units=8*8*32, activation='relu', use_bias=False),
            layers.Dense(units=8*8*32, activation='relu', use_bias=False),
            layers.Dense(latent_dim + latent_dim, use_bias=False),
        ])

        self.decoder = tf.keras.Sequential([
            #layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(units=4*4*32, activation=tf.nn.relu),
            layers.Reshape(target_shape=(4, 4, 32)),
            layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), padding='same',
                activation='relu', use_bias=False),
            layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), padding='same',
                activation='relu', use_bias=False),
            # No activation
            layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(2, 2), padding='same', use_bias=False),
            
            #layers.Dense(3072, activation='relu'),
            layers.Flatten(),
            layers.Dense(3*(largest[0]**2), activation='sigmoid', use_bias=False), #, activation='sigmoid'
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
catdream = Autoencoder(32)
if 'y' in load_flag:
    print("loading...")
    catdream.model_load()
    catdream.encoder.summary()
    catdream.decoder.summary()

#Adam( lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1.0, decay = 0.1 ) #tf.keras.optimizers.Adam(lr=.001)
catdream.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],)

catdream.fit(x_train, x_train,
    epochs=5,
    shuffle=True,
    #validation_split=.2,
    #batch_size=32
)

catdream.model_save()
