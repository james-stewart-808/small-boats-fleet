"""
__main__.py

The main executable file to used in the processing of all files related to the
small-boats-fleet repository.


Research used in the study

        ...                                 ... (...)

A full description of the research and references used can be found in README.md

"""


# Import libraries and data processing files #
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

import pywt
import pywt.data

# import other scripts
from contrast import contrast
from canny_edge import canny_edge
from detection_analysis import detection_analysis
from component_analysis import component_analysis
from wavelet_transform import wavelet_transform
from candidate_statistics import candidate_statistics


def usage_check():
    """
    Check terminal call usage and load dataset into pandas dataframe. Entry
    should include directory with trainging data...

    python3 __main__.py data/Training_NSul

    Inputs:

            tr_dir      directory to training set.

    Outputs:

            df          pandas dataframe of stock price history.

    """

    if len(sys.argv) != 2:
        print("Usage: python3 __main__.py data/Training_NSul")
        sys.exit(1)

    else:
        print("Correct usage")
        data_dir = sys.argv[1]
        print(data_dir)

    return data_dir




def data_import(file_path):
    """
    Import folder of data images using command line argument data_dir.

    Input:

            data_dir    terminal argument representing image data directory

    Output:

            image       image array?

    """

    # Import image
    image = cv2.imread(file_path, 1)

    return image


if __name__ == '__main__':
    """
    Execution file.

    For each image in the directory, extract candidate slices and send them to
    candidate_analysis.py to obtain feature values.

    Use candidate characteristics to classify candidate slices into fishing
    vessel or not.

    """

    # check bash usage
    print("0. starting usage check")
    data_dir = usage_check()
    print("finished usage check")


    # get list of images in directory
    image_list = os.listdir(data_dir)
    if '.DS_Store' in image_list:
        image_list.remove('.DS_Store')
    image_list.sort()
    print(image_list)



    # define pandas dataframe for results
    print("creating results dataframe")
    cols = ['filename', 'cand_no', 'cent_x', 'cent_y', 'WT_coeff', 'width', 'height', 'area', 'max', 'sigma', 'skew', 'entropy', 'ave_R', 'ave_G', 'ave_B']
    results_df = pd.DataFrame(columns=cols)


    # loop through training images
    print("starting execution for loop")
    for filename in image_list:

        # print filename
        print(filename)

        # define image results directory to save plots in
        im_dir = r'results/run_plots/' + filename + '/'
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)

        # 0. import image
        print("0. starting data import")
        file_path = data_dir + '/' + filename
        im = data_import(file_path)
        print("finished data import")

        # save fig
        plt.imsave(im_dir + "0. Original image.png", im)

        # 1. contrast stage
        print("1. starting image contrast enhancement")
        im_cont = contrast(im, im_dir)
        print("1. finished image contrast enhancement")


        # 2. edge detection
        print("2. starting edge detection")
        im_edge = canny_edge(im_cont, im_dir)
        print("2. finished edge detection")


        # 3. get details of components found in edge detection & plot graph
        print("3. starting edge detection analysis and plotting")
        labels, stats, centroids = detection_analysis(im, im_edge, im_dir)
        print("3. finished component analysis")


        # 4. wavelet_transform
        print("4. starting wavelet transform")
        LL, (LH, HL, HH) = wavelet_transform(im_cont, im_dir)
        print("4. finished wavelet transform")


        # 5. compiling features for x candidates
        # set up candidate image directory
        print("5.1 set up candidate image directory if not already existing")
        cand_im_dir = im_dir + "5. Component images/"
        if not os.path.exists(cand_im_dir):
            os.makedirs(cand_im_dir)

        else:
            pass


        # run through candidates and extract their features.
        print("5.2 compiling features for " + str(len(centroids)) + " candidates")

        for comp_idx in range(1, len(centroids)):

            # Component analysis
            print("5.2." + str(comp_idx) + " Analysing Component")
            centroid, width, height, area, max, sigma, skew, entropy, ave_R, ave_G, ave_B, include  = candidate_statistics(im, comp_idx, cand_im_dir, labels, stats, centroids)

            # dont' include if component is too big or small to be a small vessel
            if include == False:
                continue

            # reference wavelet transform coefficient
            LL_coeff = LL[int(centroids[comp_idx][1] / 2), int(centroids[comp_idx][0] / 2)]


            print("starting results compilation")
            row_df = pd.DataFrame([[filename, comp_idx + 1, centroid[0], centroid[1], int(LL_coeff), width, height, area, max, np.round(sigma[0][0], 1), np.round(skew, 2), np.round(entropy, 2), int(ave_R), int(ave_G), int(ave_B)]])
            row_df.columns = cols
            results_df = pd.concat([results_df, row_df], ignore_index=False)
            print("finished results compilation")


    print("printing results to csv")
    now = datetime.datetime.now().strftime("%m.%d.%Y %H-%M-%S")

    results_df_alph = results_df.sort_values(by=['cand_no'], ascending = True)
    results_df_alph.to_csv("/Users/apple/repos/dissEnv/results/"+now+".csv", index=False)
