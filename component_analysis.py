"""
component_analysis.py

Costantino Grana, Daniele Borghesani, and Rita Cucchiara. Optimized Block-Based
Connected Components Labeling With Decision Trees. IEEE Transactions on Image
Processing, 19(6):1596–1609, 2010


cv2.connectedComponentsWithStats: takes binary image, gives...
    no_labels - total number of labels
    labels    - which component this pixel corresponds to
    stats     - leftest x; top y; width, height, area
    centroids - matrix of centroids


Used to generate image stats and

A full description of the research and references used can be found in README.md
"""


def component_analysis(im_edge, im_dir):
    """
    Takes...

        im_edge

    Returns...


    """

    # 3.1 Copy Canny edge binary image
    print("3.1 copying canny edge binary image")
    im_edge_copy = im_edge.copy()


    # 3.2 perform component analysis
    print("3.2 performing component analysis")
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(im_edge_copy, cv2.CCL_DEFAULT)


    # 3.3 set up original image for annotation
    print("3.3 setting up original image for annotation")
    plt.imshow(im_comp_explore)
    ax = plt.gca()


    # 3.4 set candidate couter
    print("3.4 setting candidate counter, box length and statistics matrices")
    no_candidates = 0
    cand_box_length = 13



    # 3.5 set up candidate image directory
    print("3.5 set up candidate image directory")
    cand_im_dir = im_dir + "3.5 Component images/"
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)


    # for each component...
    for comp_idx in range(len(centroids)):

        # 3.6 candidate analysis
        print("3.6 analysing candidate " + str(comp_idx))
        no_candidates += 1

        # get centroid x & y
        x, y = int(centroids[comp_idx][0]), int(centroids[comp_idx][1])


        # get component width, height & area
        width, height, area = stats[comp_idx][2], stats[comp_idx][3], stats[comp_idx][4]


        # define coordinates of highlighting box
        leftest, top = stats[comp_idx][0], stats[comp_idx][1]
        rightest, bottom = leftest + width, top + height


        # only areas of between 5 and 30 square meters considered
        if area > 5 and area < 30:

            # add rectangle to original image
            rect = Rectangle((x-(width/2), y-(height/2)), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)


            # Account for if image is next to border
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


            # write candidate images in new directory
            candidate_image = orig_image[cand_box_S: cand_box_N, cand_box_W: cand_box_E]
            cv2.imwrite(cand_im_dir + "candidate " + str(no_candidates) + ".png", candidate_image)


    # 3.7 save annotated component figure
    print("3.7 Save annotated component figure")
    plt.savefig(im_dir + "3. Components and their rectangles.png")


    return labels, stats, centroids
