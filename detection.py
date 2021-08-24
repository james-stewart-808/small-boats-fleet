"""
detection.py

Apply land sea segmentation to the original RGB image and identify potential
vessel candidates from this.

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

def segmentation(image, im_dir):
    """

    Input:

        image               RGB image matrix for analysis

    Output:

        image_segmented     RGB image matrix with land masking applied

    """

    # making greyscale
    print("1.1 making image greyscale")
    image_bw =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # median blurring
    print("1.2 median blurring")
    median = cv2.medianBlur(image_bw,15)
    for it in range(8):
        median = cv2.medianBlur(median, 15)

    # thresholding step
    print("1.3 thresholding")
    ret_median, th_median = cv2.threshold(median, 50, 255, cv2.THRESH_BINARY_INV)

    # save fig
    plt.imsave(im_dir + "1.1 pre-segmentation greyscaling and thresholding.png", th_median)

    # apply segmentation
    print("1.4 masking image")
    image_segmented = image
    image_segmented

    for row in range(th_median.shape[0]):
        for col in range(th_median.shape[1]):
            if th_median[row][col] == 0:
                image_segmented[row][col][0] = 0
                image_segmented[row][col][1] = 0
                image_segmented[row][col][2] = 0
            else:
                pass

    plt.imsave(im_dir + "1.4 segmented image.png", image_segmented)
    print("finished image segmentation")

    return image_segmented



def detection(file_path, image_segmented, im_dir):
    """
    Apply a blurring/simple binary threshold detection algorithm and return
    candidate slices.

    Input:

        image_segmented         RGB image matrix with land masking applied

    Output:

        cand_img_arr            array of candidate images as RGB matrices

    """

    print("starting candidate detection")

    # preliminary blurring & convert to greyscale
    print("2.1 greyscale and blurring of segmented image")
    im_seg_blur = cv2.medianBlur(image_segmented, 15)
    im_seg_blur = cv2.medianBlur(im_seg_blur, 15)
    im_seg_blur_bw = cv2.cvtColor(im_seg_blur, cv2.COLOR_BGR2GRAY)


    # simple binary threshold
    print("2.2 binary threshold of segmented image")
    mean_intensity = int(np.average(im_seg_blur_bw))
    #if mean_intensity < 40:
    #    thresh_value = 60
    #elif mean_intensity > 100:
    #    thresh_value = 150
    #else:
    #    thresh_value = mean_intensity * 1.5

    thresh_value = 200 - (mean_intensity * 0.9)

    ret_seg_blur, th_seg_blur = cv2.threshold(im_seg_blur_bw, thresh_value, 255, cv2.THRESH_BINARY) # + Otsu? (like CA.py)

    # save fig
    plt.imsave(im_dir + "2.1 segmentation greyscaling and thresholding.png", th_seg_blur)

    # count how many detections there are
    print("2.3 count detections in image")
    no_detections = 0
    for row in range(th_seg_blur.shape[0]):
        for col in range(th_seg_blur.shape[1]):
            # if the detector hits a positive pixel, record pixel coordinates
            if th_seg_blur[row][col] == 255:
                no_detections += 1


    # allocate memory to and record detection locations
    print("2.4 allocate memory to and record detection locations")
    detections = np.array([[None, None] for i in range(no_detections)])
    detection_counter = 0

    for row in range(th_seg_blur.shape[0]):
        for col in range(th_seg_blur.shape[1]):

            # if the detector hits a positive pixel, record pixel coordinates
            if th_seg_blur[row][col] == 255:
                detections[detection_counter][0] = row
                detections[detection_counter][1] = col

                detection_counter += 1


    # if no detections found or positive pixels exceed 10% of image, exit funtion
    if len(detections) == 0:
        return None, [[1, 1]]

    if len(detections) > (image_segmented.shape[0] * image_segmented.shape[1] / 10):
        return None, [[1, 1]]


    # agglomorative single-linkage clustering with threshold distance of 2
    print("2.5 Agglomorative clustering")
    agg_clustering = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, linkage='single', distance_threshold=2).fit(detections)
    cluster_labels = agg_clustering.labels_
    no_clusters = len(np.unique(agg_clustering.labels_))


    # obtain cluster centroids
    print("2.6 get cluster centroids")
    centroids = np.array([[None, None] for i in range(no_clusters)])

    for cluster in range(no_clusters):
        row_cent, col_cent = [], []
        for detection in range(len(detections)):
            if cluster_labels[detection] == cluster:
                row_cent.append(detections[detection][0])
                col_cent.append(detections[detection][1])

        cent_row = int(round(np.average(row_cent)))
        cent_col = int(round(np.average(col_cent)))

        # otherwise, add centroid to the database
        centroids[cluster][0] = int(round(np.average(row_cent)))
        centroids[cluster][1] = int(round(np.average(col_cent)))


    # print clustering summary
    print("No. clusters: "+str(no_clusters))
    print("Centroids: \n")
    print(centroids)


    # make a numpy object of candidate 60px x 60px x 3 images
    print("2.7 generate candidate matrix object")
    cand_img_arr = np.ndarray([no_clusters, 260, 260, 3])


    # define box size from candidate size
    box_size = 260
    box_size_half = int(box_size / 2)


    # re-import original image to two variables to avoid overwriting
    print("2.8 add candidate boxes to original image")
    full_image = cv2.imread(file_path, 1) ### zero pad?
    full_image_borders = cv2.copyMakeBorder(full_image, box_size_half, box_size_half, box_size_half, box_size_half, borderType=cv2.BORDER_CONSTANT)
    new_image = cv2.imread(file_path, 1) ### filename


    for centroid in range(len(centroids)):

        # define box positions
        x1 = centroids[centroid][1] - box_size_half + box_size_half
        x2 = centroids[centroid][1] + box_size_half + box_size_half
        y1 = centroids[centroid][0] + box_size_half + box_size_half
        y2 = centroids[centroid][0] - box_size_half + box_size_half

        # append candidate image to cand_img_arr
        for row in range(y2, y1):
            for col in range(x1, x2):
                cand_img_arr[centroid][row - y1][col - x2] = full_image_borders[row, col]

        # annotate original image with detection box for this centroid
        image_with_boxes = cv2.rectangle(new_image, (x1 - box_size_half, y2 - box_size_half), (x2 - box_size_half, y1 - box_size_half), (255, 0, 0), 2)

    # save annotated image
    plt.imsave(im_dir + "2.8 candidate identification with boxes.png", image_with_boxes)

    return cand_img_arr, centroids
