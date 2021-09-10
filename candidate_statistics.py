"""
candidate_image_statistics.py

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
import scipy
from scipy import ndimage, stats

import pywt
import pywt.data


def candidate_statistics(im, comp_idx, cand_im_dir, labels, stats, centroids):

    # 5. set candidate box length
    print("5.2." + str(comp_idx + 1) + ".1 setting box length")
    cand_box_length = 15

    # get centroid x & y
    print("5.2." + str(comp_idx + 1) + ".2 get candidate centroid")
    x, y = int(centroids[comp_idx][0]), int(centroids[comp_idx][1])

    # get component width, height & area
    print("5.2." + str(comp_idx + 1) + ".3 get candidate width, height and area")
    width, height, area = stats[comp_idx][2], stats[comp_idx][3], stats[comp_idx][4]

    # define lw ratios
    r_hw = height / width
    r_wh = width / height

    # define coordinates of highlighting box
    print("5.2." + str(comp_idx + 1) + ".4 define containing box")
    leftest, top = stats[comp_idx][0], stats[comp_idx][1]
    rightest, bottom = leftest + width, top + height


    # add validation criteria
    include = False
    if area > 5 and height > 2 and height < 25 and width > 2 and width < 25 and r_hw < 3.5 and r_wh < 3.5:
        include = True


    # Attain candidate image slicing boundaries - accounting for border cases
    print("5.2." + str(comp_idx + 1) + ".5 get candidate pixel slicing boundaries")

    cand_box_N = y + cand_box_length
    if cand_box_N < 0:
        cand_box_N = 0

    cand_box_S = y - cand_box_length
    if cand_box_S < 0:
        cand_box_S = 0

    cand_box_W = x - cand_box_length
    if cand_box_W < 0:
        cand_box_W = 0

    cand_box_E = x + cand_box_length
    if cand_box_E < 0:
        cand_box_E = 0


    # slicing original image and write figure
    print("5.2." + str(comp_idx + 1) + ".6 slicing original image and write figure")
    cand_im = im[cand_box_S: cand_box_N, cand_box_W: cand_box_E]

    if include == True:
        plt.figure()
        plt.imshow(cand_im)
        plt.savefig(cand_im_dir + "candidate " + str(comp_idx + 1) + ".png")
        plt.figure()

    #print(cand_im_dir + "candidate " + str(comp_idx) + ".png")
    #cv2.imwrite(cand_im_dir + "candidate " + str(comp_idx) + ".png", cand_im)


    # greyscale max, standard dev. & skew
    print("5.2." + str(comp_idx + 1) + ".7 greyscale max, standard dev. & skew")
    cand_im_bw = cv2.cvtColor(cand_im, cv2.COLOR_BGR2GRAY)
    cand_im_bw_resh = [int(i) for i in cand_im_bw.reshape(cand_im_bw.shape[0] * cand_im_bw.shape[1])]

    cand_im_bw_max = max(cand_im_bw_resh)
    cand_im_bw_mean, cand_im_bw_stddev = cv2.meanStdDev(cand_im_bw)
    cand_im_bw_skew = scipy.stats.skew(cand_im_bw_resh)


    # get entropy
    print("5.2." + str(comp_idx + 1) + ".8 calculating entropy")
    marg = np.histogramdd(np.ravel(cand_im_bw), bins = 256)[0]/cand_im_bw.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))


    # get average RGB values
    print("5.2." + str(comp_idx + 1) + ".9 get average RGB values")
    ave_R = np.mean(cand_im[:,:,2])
    ave_B = np.mean(cand_im[:,:,1])
    ave_G = np.mean(cand_im[:,:,0])


    return (x, y), width, height, area, cand_im_bw_max, cand_im_bw_stddev, cand_im_bw_skew, entropy, ave_R, ave_G, ave_B, include
