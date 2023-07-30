import numpy as np
# import pandas as pd
import cv2
import os
# from random import shuffle
from PIL import Image

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
#     print(hist.flatten())



    return hist.flatten()


def preprocess(filepath):
    image_size = 128
    img = cv2.imread(filepath)
    img = cv2.resize(img, (image_size, image_size))
    img = extract_color_histogram(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img