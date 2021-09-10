"""
contrast.py

Takes segmented image and applies a contrast.
Takes image directory to save plots in.

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

#cwd = "/Users/apple/repos/dissEnv/results/run_plots/seg_test/"

def contrast(im, im_dir):
    """

    Input:

        image               RGB image matrix for analysis
        im_cand_dir         image directory to save plots in

    Output:

        im_contrast         image with enhanced contrast

    """

    # 1.1 making greyscale
    print("1.1 making image greyscale")
    im_bw =  cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    cv2.imwrite(im_dir + "1.1 greyscale.png", im_bw)


    # 1.2 preliminary data manipulation
    print("1.2 making image greyscale")
    im_bw_resh = np.reshape(im_bw, im_bw.shape[0] * im_bw.shape[1])
    im_bw_resh_no_black = [int(i) for i in np.delete(im_bw_resh, np.where(im_bw_resh == 0))]
    #counts, values = np.histogram(im_bw_resh_no_black, bins=range(256), range=(0, 255))


    # 1.3 get image's greyscale mode
    print("1.3 contrast enhancement")
    im_bw_mode = statistics.mode(im_bw_resh_no_black) # mode
    print("greyscale image mode is " + str(im_bw_mode))


    # 1.4 Apply contrast enhancement function
    print("1.4 Apply contrast enhancement function")
    im_bw_cont = np.zeros((im_bw.shape[0], im_bw.shape[1]))
    for row in range(im_bw.shape[0]):
        for col in range(im_bw.shape[1]):
            value = im_bw[row][col]

            if value < im_bw_mode:
                scaled_value = 0
                im_bw_cont[row][col] = scaled_value

            else:
                scaled_value = int((value - im_bw_mode) / (255 - im_bw_mode) * 255)
                im_bw_cont[row][col] = scaled_value


    # 1.5 write contrast enhanced image to directory
    print("1.5 writing contrast enhanced image to directory")
    cv2.imwrite(im_dir + "1.2 contrast.png", im_bw_cont)


    # 1.6 Blur to remove sparkle
    #print("1.6 Blurring greyscale image")
    #im_bw_cont_blur = cv2.GaussianBlur(im_bw_cont, (5,5), 0)
    #cont_blur_2 = cv2.GaussianBlur(cont_blur_1, (5,5), 0)
    #cv2.imwrite(im_dir + "1.3 contrast with blur.png", im_bw_cont_blur)

    return im_bw_cont
