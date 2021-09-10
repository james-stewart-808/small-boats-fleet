"""
edge_detection_analysis.py


Analyse the edges detection via Canny method and use them to generate candidates.

uses cv2.connectedComponentsWithStats: takes binary image, gives...
    no_labels - total number of labels
    labels    - which component this pixel corresponds to
    stats     - leftest x; top y; width, height, area
    centroids - matrix of centroids


Costantino Grana, Daniele Borghesani, and Rita Cucchiara. Optimized Block-Based
Connected Components Labeling With Decision Trees. IEEE Transactions on Image
Processing, 19(6):1596–1609, 2010


Get details of components found in edge detection & plot their bounding
rectangles on original image.

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


def detection_analysis(im, im_edge, im_dir):
    """
    Takes...

        im_edge

    Returns...


    """

    # 3.1 Copy Canny edge binary image
    print("3.1 copying original image")
    im_copy = im.copy()
    im_edge_copy = im_edge.copy()


    # 3.2 perform component analysis
    print("3.2 performing component analysis")
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im_edge_copy, cv2.CCL_DEFAULT)


    # 3.3 set up original image for annotation
    print("3.3 setting up original image for annotation")
    plt.imshow(im_copy)
    ax = plt.gca()


    # 3.4 add each component to original image
    print("3.4 Save annotated component figure")
    for comp_idx in range(len(centroids)):

        # unpack features
        x, y = int(centroids[comp_idx][0]), int(centroids[comp_idx][1])
        width, height, area = stats[comp_idx][2], stats[comp_idx][3], stats[comp_idx][4]

        # define lw ratios
        r_hw = height / width
        r_wh = width / height

        # add rectangle to original image
        if area > 5 and height > 2 and height < 25 and width > 2 and width < 25 and r_hw < 3.5 and r_wh < 3.5:
            rect = Rectangle((x-(width/2), y-(height/2)), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)


    # 3.5 save annotated component figure
    print("3.5 Save annotated component figure")
    plt.savefig(im_dir + "3. Components and their rectangles.png")

    return labels, stats, centroids
