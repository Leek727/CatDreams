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

scales = [0.0, 0.0, 0.0, 3.2628458, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.1708374, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1773875, 3.6229262, 0.0, 3.5402703, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.890513, 3.2750554, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.4013338, 0.0, 0.0, 0.0, 0.0, 0.0]
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
            latent_array[i] = x * (components[comb_ind]/(smoothness-1))
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