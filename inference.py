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

pca = [0.002354661077260971, 0.40112655114382506, -0.6483993274718522, 0.10794308144599199, 0.16733334198594094, -0.06752898339182138, 0.447611088193953, 0.24272811144590378, -0.025232470072805883, -0.47108399733901024, -0.56014005176723, -0.1263374139368534, 0.42400647163391114, -0.2534391575679183, 0.38182147204875944, 1.5553990802541375, 0.1619304571673274, -0.2861235439777374, 0.24072791039943695, -2.192128039300442, 0.4312834207713604, 2.6367121344804763, -0.2060701078362763, 0.019542199671268464, 0.7368915884196758, -0.39984714701771734, -0.2267510910332203, 0.05629863508045673, 0.7220342258736491, -0.28101025607436897, 0.1553202982619405, -0.17287942990660668]
for components in itertools.product([x for x in range(2)], repeat=64):
    #latent_array = [0]*8
    #for i,x in enumerate(pca):
    #    latent_array[x] = components[i]
    
    #latent_array = [x*pca[i]*2 for i,x in enumerate(components)]
    #latent_array = [pca[0] * latent_array[0], 0,0, pca[1] * latent_array[1]]

    #latent_space = np.array([i/100 for x in range(64)])
    img = model.predict(np.array([components]))[0]
    img = (img*255).astype(np.uint8)
    
    # resize image
    img = cv2.resize(img, (300,300))

    # write image and show
    cv2.imshow('Frame', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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