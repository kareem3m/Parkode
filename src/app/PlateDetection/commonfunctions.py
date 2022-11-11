import cv2
import os
#from numba import jit, cuda , double
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import re


# Convolution:

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb
from sklearn import cluster
from skimage.filters import threshold_otsu

import skimage.morphology as morph
import time

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def load_images_from_folder(folder):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    listFiles = sorted(os.listdir(folder),key= alphanum_key)
    images = []
    for filename in listFiles:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images