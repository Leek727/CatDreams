import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tensorflow as tf
from keras.models import load_model

model = load_model('catdream.h5')
model.summary()
latent_dim = 15

def gen(latent):
    img = model.predict(np.array([latent]))[0]
    #img = (img*255).astype(np.uint8)
    return img

# Create a subplot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=.5)

#plt.imshow(gen([0]*latent_dim))
 
# create sliders
sliders = []
for i in range(latent_dim):
    sliders.append(Slider(plt.axes([0.25, (i/100)*3, 0.65, 0.03]), f'Latent var {i} : ', 0.0, 1.0))

 
# Create function to be called when slider value is changed
def update(val):
    latent_space = []
    for i in sliders:
        latent_space.append(i.val)

    print(latent_space)
    ax.imshow(gen(latent_space))#,cmap='gray'
 
# Call update function when slider value is changed
for i in sliders:
    i.on_changed(update)

 
# Display graph
plt.show()