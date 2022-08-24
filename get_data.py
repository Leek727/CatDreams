import os
from PIL import Image
import numpy as np

x_train = []

f = 0
for root, dirs, files in os.walk("cats", topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if ".png" in path:
            try:
                img = np.array(Image.open(path).convert('L').getdata(), dtype=np.uint8)
                img = img.reshape(64,64)
                x_train.append(img) # Note the dtype input

            except Exception as e:
                print(e)
                print(path)
            
x_train = np.array(x_train).astype('float32') / 255

largest = x_train[0].shape