"""
classification.py

Use vessel candidate features to decide whether candidate is a fishing
vessel or not.

From type-specific distributions.

From research?

A full description of the research and references used can be found in README.md
"""


def classification(length, breadth, area, lb_ratio, im_dir):
    """
    Input:
        length              estimated length of vessel candidate
        breadth             estimated breadth of vessel candidate
        area                estimated area of vessel candidate
        lb_ratio            estimated length-breadth ratio of vessel candidate
    Output:
        small_vessel        decision on fishing vessel or not, binary True/False
    """

    small_vessel = True

    return small_vessel
