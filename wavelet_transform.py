"""
wavelet_transform.py

Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary
(2019). PyWavelets: A Python package for wavelet analysis. Journal of Open
Source Software, 4(36), 1237, https://doi.org/10.21105/joss.01237.

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

import pywt
import pywt.data


def wavelet_transform(im_cont, im_dir):
    """
    Takes...

        contrasted image in uint 8

    Returns...


    """

    # 4.1 Convert to 8-bit
    print("4.1 converting greyscale image to 8-bit")
    im_cont_8 = im_cont.astype(np.uint8)


    # 4.2 define images to be generated
    print("4.2 defining images to be generated")
    titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
    cA, cD = pywt.dwt([1, 2, 3, 4], 'db1')


    # 4.3 get coefficient matrices
    print("4.3 getting coefficient matrices")
    LL, (LH, HL, HH) = pywt.dwt2(im_cont_8, 'haar')


    # 4.4 generate figure for coefficient matrices
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.savefig(im_dir + "4. Wavelet Transform.png")
    plt.close()

    return LL, (LH, HL, HH)
