"""
discrimination.py

Use vessel candidate features to decide whether candidate is a small
vessel or not.

Based on total shape and spectral features.

From research?

A full description of the research and references used can be found in README.md

"""

import sys, os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import importlib

# Using OpenCV for image analysis
import cv2
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from scipy import ndimage

def discrimination(length, breadth, area, lb_ratio, im_dir):
    """

    Input:

        length              estimated length of vessel candidate
        breadth             estimated breadth of vessel candidate
        area                estimated area of vessel candidate
        lb_ratio            estimated length-breadth ratio of vessel candidate

    Output:

        small_vessel        decision on fishing vessel or not, binary True/False

    """

    # is vessel below 25 meters?
    print("length: " + str(length))
    if length < 25.0:
        small_vessel = True

    else:
        small_vessel = False

    return small_vessel
