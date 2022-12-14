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

a = [0]*32
f = 0
for root, dirs, files in os.walk("cats", topdown=False):
    for name in files:
        path = os.path.join(root, name)
        
        #img = np.array(Image.open(path).convert('L').getdata(), dtype=np.uint8)
        #img = img.reshape(64,64) / 255
        img = np.array(Image.open(path)) / 255
        ff = img
            #print(img[0])

        latent = aaa.predict(np.array([img]))[0]
        for i in range(len(a)):
            a[i] += latent[i]

        #print(list(latent))
        print(latent.shape)
        img = model.predict(np.array([latent]))[0]
        #print(img[0])
           
        # resize image
        img = cv2.resize(img, (300,300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cv2.imshow("", img)#np.concatenate((img, cv2.resize(ff, (300,300))), axis=1))
        cv2.imwrite(f"outputs/{f}.png",img*255)
        #cv2.imshow("",ff)
        cv2.waitKey(50)

        if f > 50:
            break

        f+= 1

print([x/50 for x in a])