"""
canny_edge.py

cv2.Canny: noise reduction, intensity gradient, non-maximum suppression, hysterisis thresholding, aperture_size is Sobel kernal default 3


A full description of the research and references used can be found in README.md

"""

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

import statistics
import datetime
import importlib

# Using OpenCV for image analysis
import cv2 # Version 4.2.0
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage, stats


def canny_edge(im_cont, im_dir):
    """

    Input:

        image               RGB image matrix for analysis
        im_cand_dir         image directory to save plots in

    Output:

        im_contrast         image with enhanced contrast

    """

    # 2.1 convert to uint8 for edge detection
    print("2.1 converting to 8-bit for cv2.Canny algorithm")
    im_uint8 = im_cont.astype(np.uint8)


    # 2.2 Canny edge detection into binary image
    print("2.2 performing Canny edge detection")
    im_uint8_200_250 = cv2.Canny(im_uint8, 150, 200)
    cv2.imwrite(im_dir + "2. canny_200_250.png", im_uint8_200_250)

    return im_uint8_200_250
