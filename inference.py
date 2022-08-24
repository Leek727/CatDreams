from posixpath import commonpath
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import cv2
import itertools

model = load_model('catdream.h5')
model.summary()

index = 0
#pca = [5,25,23]#,12,31,8]#,10,9,7,14,2,3,4]
pca = [1]*4
for components in itertools.product([x for x in range(50)], repeat=2):
    #latent_array = [0]*8
    #for i,x in enumerate(pca):
    #    latent_array[x] = components[i]
    latent_array = [x/50 for x in components]
    latent_array = [0, latent_array[0]*66, latent_array[1]*39, 0]

    #latent_space = np.array([i/100 for x in range(64)])
    img = model.predict(np.array([latent_array]))[0]
    img = (img*255).astype(np.uint8)
    
    # resize image
    img = cv2.resize(img, (300,300))

    # write image and show
    #cv2.imshow('Frame', img)
    cv2.imwrite(f"outputs/{index}.png", img)
    index += 1
    print(index)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()





"""import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import random

model = load_model('catdream.h5')
model.summary()

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    # generate random latent space
    latent_space = np.array([random.randint(0,100)/100 for x in range(64)])
    img = model.predict(np.array([latent_space]))[0]
    
    fig.add_subplot(rows, columns, i)
    plt.gray()
    plt.imshow(img)

plt.show()"""