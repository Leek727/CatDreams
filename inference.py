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

scales = [0.0, 6.239134166836738, 0.0, 5.405863894224167, 5.883739851117134, 6.177591247260571, 5.716404540538788, 0.0, 5.278323036432266, 5.776310132741928, 0.0, 0.0, 0.0, 6.258259708881378, 6.4977303221821785, 5.125401201248169, 0.0, 5.132884674072265, 6.108139206171035, 6.786262208223343, 6.791425176858902, 4.002850511074066, 5.305520109534264, 5.3863698595762255, 5.432120316028595, 5.617525144815445, 5.329741010665893, 6.1920134437084196, 0.0, 6.822000298500061, 5.147960944771767, 7.750986666679382, 4.848914276361466, 8.8656735599041, 0.0, 0.0, 0.0, 6.024914858937263, 6.102861387431622, 5.283858259320259, 4.128546927571296, 0.0, 4.671433743983507, 6.958697194457054, 0.0, 5.869447401762009, 5.724955952316523, 5.500033801198006, 7.968746550083161, 5.588805692195892]
diff = 0
for i in scales:
    if i > 0:
        diff += 1

smoothness = 10
latent_dim = 50
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