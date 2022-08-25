import os
from PIL import Image
import numpy as np

x_train = []

f = 0
for root, dirs, files in os.walk("cats", topdown=False):
    for name in files:
        path = os.path.join(root, name)
        try:
            #img = np.array(Image.open(path).convert('L').getdata(), dtype=np.uint8)
            #img = img.reshape(64,64)
            img = np.array(Image.open(path))
            #img = np.array(img).astype('float32') / 255
            #b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            x_train.append(img) 

        except Exception as e:
            print(e)
            print(path)
        
        """if f > 10:
            break

        f+=1"""
            
x_train = np.array(x_train).astype('float32') / 255

largest = x_train[0].shape