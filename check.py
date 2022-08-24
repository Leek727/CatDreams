import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os
from PIL import Image
import cv2

model = load_model('catdream.h5')
aaa = load_model('encoder.h5')
model.summary()

a = [0,0,0,0]
f = 0
for root, dirs, files in os.walk("cats", topdown=False):
    for name in files:
        path = os.path.join(root, name)
        if ".png" in path:
            img = np.array(Image.open(path).convert('L').getdata(), dtype=np.uint8)
            img = img.reshape(64,64) / 255
            #print(img[0])

            latent = aaa.predict(np.array([img]))[0]
            for i in range(len(a)):
                if a[i] < latent[i]:
                    a[i] = latent[i]

            print(a)
            img = model.predict(np.array([latent]))[0]
           
            # resize image
            img = cv2.resize(img, (300,300))
            cv2.imshow("", img)
            cv2.waitKey(50)

        if f > 1000:
            break

        f+= 1