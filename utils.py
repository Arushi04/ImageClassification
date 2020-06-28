import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    print(img.shape)
    img = img / 2 + 0.5     # unnormalize to bring in range [0,1]
    npimg = img.numpy()     # (3,M,N)
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # Reshaping as imshow takes image in (M,N,3) format.
    plt.show()
