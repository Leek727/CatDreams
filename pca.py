from keras.models import load_model
import numpy as np
import os
from PIL import Image
import numpy as np
import tensorflow as tf

x_train = []

f = 0
for root, dirs, files in os.walk("cats", topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if ".png" in path:
            img = np.array(Image.open(path).convert('L').getdata(), dtype=np.uint8)
            img = img.reshape(64,64)
            x_train.append(img) # Note the dtype input
            

        if f > 100:
            break

        f+= 1
           

x_train = np.array(x_train).astype('float32') / 255

model = load_model('encoder.h5')
model.summary()


print()
"""
latent_dim = np.array([])
latent_dist = [0]*latent_dim
primary = {}
for f,img in enumerate(x_train):
    latent_space = model.predict(np.array([img]))[0]
    pc = np.argsort(latent_space)[::-1]
    

    for i in range(latent_dim):
        #print(latent_space[pc[i]])
        if pc[i] in primary:
            primary[pc[i]] += 1
        else:
            primary[pc[i]] = 1

    print(f)
print(latent_dist)

"""