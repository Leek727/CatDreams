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

scales = [0.0, 0.0, 0.0, 2.7438209104537963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.763633061647415, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.58061383664608, 2.7355423051118852, 0.0, 2.9872653317451476, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.422852528691292, 3.016243551969528, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0667327785491945, 0.0, 0.0, 0.0, 0.0, 0.0]
diff = 0
for i in scales:
    if i > 0:
        diff += 1

smoothness = 5
latent_dim = 64
for components in itertools.product([x for x in range(smoothness)], repeat=diff):
    comb_ind = 0
    latent_array = [0]*latent_dim
    for i,x in enumerate(scales):
        if x > 0:
            latent_array[i] = 2*x * (components[comb_ind]/(smoothness-1))
            comb_ind += 1
    
    
    #latent_array = [(x/10)*scales[i] for i,x in enumerate(components)]
    
    img = model.predict(np.array([latent_array]))[0]
    img = (img*255).astype(np.uint8)
    
    # resize image
    img = cv2.resize(img, (300,300))

    # write image and show
    cv2.imshow('Frame', img)
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