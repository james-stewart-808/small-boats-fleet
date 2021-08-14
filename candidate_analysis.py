"""
candidate_analysis.py

Takes an image slice, analyses it and returns its feature values.

Apply the candidate locations returned from detection() and use these to
create image slices. Once again apply a smoothing/threshold algorithm to
locate postiviely identified pixels within the image slices and create
data structures from these suitable for use in the CV2 PCA function.

Using PCA, obtain the angle deviation and correct the image accordingly.
Treating the resulting image as a rectangle, use a dataframe of postive
pixels to estimate the length, breadth, area and lb ratio.

The discrimination functions can be replaced with more advanced (addition of
spectral/textural features) or black box (CNN, deep learning) algorithms in
time.

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

def candidate_analysis(cand_img, im_cand_dir):
    """
    Input:

        cand_img            candidate image matrix, 60x60x3
        im_cand_dir         candidate image directory

    Output:

        length              estimated length of vessel candidate, px
        breadth             estimated breadth of vessel candidate, px
        area                estimated area of vessel candidate, px
        lb_ratio            estimated length-breadth ratio of vessel candidate, px

    """

    # Simple binary threshold to candidate slice
    print("3.1 apply greyscaling and thresholding to candidate image")
    cand_img_bw = cv2.cvtColor(cand_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    cand_img_bw_blur_1 = cv2.GaussianBlur(cand_img_bw, (5,5), 0)
    cand_img_bw_blur_2 = cv2.GaussianBlur(cand_img_bw_blur_1, (5,5), 0)
    th_cand_img = cv2.threshold(cand_img_bw_blur_2, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # save annotated image
    plt.imsave(im_cand_dir + "3.1 greyscaling and thresholding of candidate.png", th_cand_img)


    # get positive pixel x, y co-ordinates in lists
    print("3.2 get identified pixel coordinates and record")
    row_cent, col_cent = [], []
    for row in range(th_cand_img.shape[0]):
        for col in range(th_cand_img.shape[1]):
            if th_cand_img[row][col] == 255:
                row_cent.append(row)
                col_cent.append(col)


    # get spectral features 
    print("3.3 Get spectral features of candidate")
    spec_r, spec_g, spec_b = [], [], []
    for row in range(th_cand_img.shape[0]):
        for col in range(th_cand_img.shape[1]):
            if th_cand_img[row][col] == 255:
                spec_r.append(cand_img[row][col][0])
                spec_g.append(cand_img[row][col][1])
                spec_b.append(cand_img[row][col][2])


    # create coordinate arrays in form PCA function can use
    print("3.4 reshape positive pixel coordinates into form for PCA function")
    data_pts = np.empty((len(row_cent), 2), dtype=np.float64)

    for i in range(data_pts.shape[0]):
        data_pts[i,0] = row_cent[i]
        data_pts[i,1] = col_cent[i]


    # Complete PCA and get angle deviation
    print("3.5 complete PCA, derive angle deviation and apply")
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=np.empty((0)))
    angle_rad = math.atan2(eigenvectors[0,0], eigenvectors[0,1])
    angle_deg = math.degrees(angle_rad)


    # take angle derived from PCA and mitigate image slice accordingly
    cand_img_rot = ndimage.rotate(th_cand_img, angle_deg)


    # save fig
    plt.imsave(im_cand_dir + "3.5 rotated candidate image.png", cand_img_rot)


    # get newly rotated positive pixel locations
    print("3.6 get newly rotated positive pixel locations")
    row_img_rot, col_img_rot = [], []

    for row in range(cand_img_rot.shape[0]):
        for col in range(cand_img_rot.shape[1]):
            if cand_img_rot[row][col] == 255:
                row_img_rot.append(row)
                col_img_rot.append(col)


    # make pandas dataframe out of positive pixel coordinate lists
    print("3.7 generate positive pixel dataframe")
    rotated_df = pd.DataFrame(data=[row_img_rot, col_img_rot]).T
    rotated_df.columns = ['rows', 'cols']


    # obtain location of highest, lowest, most left and right pixels
    print("3.8 get fringe coordinates and generate feature values from it")
    p_north = min(rotated_df["rows"].values)
    p_east = max(rotated_df["cols"].values)
    p_south = max(rotated_df["rows"].values)
    p_west = min(rotated_df["cols"].values)


    # use these to generate estimates of length, breadth, area, lb_ratio
    length = p_east - p_west # width
    breadth = p_south - p_north # height
    area = len(rotated_df)
    lb_ratio = length / breadth

    return length, breadth, area, lb_ratio, np.average(np.array(spec_r)), np.average(np.array(spec_g)), np.average(np.array(spec_b))
