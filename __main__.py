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

# import other scripts
from detection import segmentation, detection
from candidate_analysis import candidate_analysis
from discrimination import discrimination


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

    if len(sys.argv) != 3:
        print("Usage: python3 __main__.py data/Training_NSul True")
        sys.exit(1)

    else:
        print("Correct usage")
        data_dir = sys.argv[1]
        seg_on = sys.argv[2]
        print(data_dir)

    return data_dir, seg_on




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
    data_dir, seg_on = usage_check()
    print("finished usage check")


    # get list of images in directory
    image_list = os.listdir(data_dir)
    #image_list = ["NSul_Makawidei_1.48_125.24.png"]
    if '.DS_Store' in image_list:
        image_list.remove('.DS_Store')
    print(image_list)


    # define pandas dataframe for results
    print("creating results dataframe")
    cols = ['filename', 'cand_no', 'cand_row', 'cand_col', 'length', 'breadth', 'area', 'lb_ratio', 'red', 'green', 'blue', 'small_vessel', 'classification']
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

        # import image
        print("0. starting data import")
        file_path = data_dir + '/' + filename
        image = data_import(file_path)
        print("finished data import")

        # save fig
        plt.imsave(im_dir + "0.0 original image.png", image)

        # detection stage
        if seg_on == True:
            print("1. starting image segmentation")
            image_segmented = segmentation(image, im_dir)
            print("finished image segmentation")

        else:
            image_segmented = image


        print("2. starting image detection")
        cand_img_arr, passed_centroids = detection(file_path, image_segmented, im_dir)
        print("finished image detection")

        if passed_centroids[0][0] == 1:
            row_df = pd.DataFrame([[filename, "none found", None, None, None, None, None, None, None, None, None, None, None]])
            row_df.columns = cols
            results_df = pd.concat([results_df, row_df], ignore_index=False)

            continue


        # candidate_analysis & discrimination stages
        print("starting candidate for loop")
        for img_index in range(cand_img_arr.shape[0]):

            # define image results directory to save plots in
            im_cand_dir = r'results/run_plots/' + filename + '/' + str(img_index + 1) + '/'
            if not os.path.exists(im_cand_dir):
                os.makedirs(im_cand_dir)

            print("3. starting candidate analysis")
            length, breadth, area, lb_ratio, r_ave, g_ave, b_ave = candidate_analysis(cand_img_arr[img_index].astype('float32'), im_cand_dir)
            print("finished candidate analysis")

            print("4. starting candidate discrimination")
            small_vessel = discrimination(length, breadth, area, lb_ratio, im_dir)
            print("finished candidate discrimination")

            #print("5. starting candidate classification")
            #if small_vessel == True:
                #classification = classification(length, breadth, area, lb_ratio, im_dir)
            #else:
            classification = "N/A"
            #print("finished candidate classification")

            print("starting results compilation")
            row_df = pd.DataFrame([[filename, img_index + 1, passed_centroids[img_index][0], passed_centroids[img_index][1], round(0.1*length, 2), round(0.1*breadth, 2), round(0.01*area, 2), round(lb_ratio, 2), int(r_ave), int(g_ave), int(b_ave), small_vessel, classification]])
            row_df.columns = cols
            results_df = pd.concat([results_df, row_df], ignore_index=False)
            print("finished results compilation")


    print("printing results to csv")
    now = datetime.datetime.now().strftime("%m.%d.%Y %H-%M-%S")
    results_df_alph = results_df.sort_values(by=['filename'], ascending = False)
    results_df_alph.to_csv("/Users/apple/repos/dissEnv/results/"+now+".csv")
