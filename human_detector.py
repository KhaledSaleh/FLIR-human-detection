"""The human detector algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Dependency imports
import os
import glob
import itertools
import numpy as np
import cv2
import re
from tqdm import tqdm
from data_utils import total_files_counter, walk_dir


# Threshold value of the binary thresholding stage
THRESH_VALUE = 120
# The max threshold value each pixel below THRESH_VALUE is set to
MAX_THRESH_VALUE = 255
# Min and max values for contour areas of human body
MIN_CNTR_HUMN_AREA = 8
MAX_CNTR_HUMN_AREA = 350

def human_detection_simple(inp_grayscalesPth, inp_heatmapsPth, det_outImgsPth, upsample_ratio = None):
    """Detect and localise all human instances in input grayscale/heatmap images.
    
    * Firstly, the binary thresholding stage.
    * Secondly, the contour detection stage.
    * Thirdly, the human blob filtration stage.
    * Lastly, the human instances localised with bounding boxes.

    Args:
        inp_grayscalesPth: the path to the grayscale images.
        inp_heatmapsPth: the path to the heatmap images.
        det_outImgsPth: the path to the human detection result images.
        upsample_ratio: an optional ratio value to upsample the returned images with.

    Returns:
	    The path to the two directories of the generated human detection images.
    """
    # Output directories for the detected humans in grayscale and heatmap images
    heatmapsDetDir = os.path.join(det_outImgsPth, 'det_heatmaps')
    graysDetDir = os.path.join(det_outImgsPth, 'det_grays')
    # Check if the input grayscale images path exists or not
    if not os.path.exists(inp_grayscalesPth):
        print (inp_grayscalesPth + ' is not a valid path')
        exit(-1)
    # Check if the input heatmap images path exists or not
    if not os.path.exists(inp_heatmapsPth):
        print (inp_heatmapsPth + ' is not a valid path')
    # Count the total number of all *.png images in the inp_grayscalesPth
    image_counter = total_files_counter(inp_grayscalesPth, '.png')
    # Check whether there are grayscale images avaialable in the inp_grayscalesPth
    if not image_counter:
        print (inp_grayscalesPth + ' contains no images')
        exit(-1)
    # Check if the detected humans heatmaps directory is created or not
    if not os.path.exists(graysDetDir):
        os.makedirs(graysDetDir)
    # Check if the detected humans gryascale images directory is created or not
    if not os.path.exists(heatmapsDetDir):
        os.makedirs(heatmapsDetDir)
    # Iterate over all the gryascale and heatmap images
    for grayscale, heatmap in tqdm(itertools.izip(walk_dir(inp_grayscalesPth,'.png'),
                                             walk_dir(inp_heatmapsPth, '.png')),
                                             total=image_counter, desc = 'Generating human detection images'):
        # Read both grayscale and heatmap images
        grayscale_img = cv2.imread(grayscale,cv2.IMREAD_GRAYSCALE)
        heatmap_img = cv2.imread(heatmap)
        # Binary thresholding stage
        ret, thresh = cv2.threshold(grayscale_img, THRESH_VALUE, MAX_THRESH_VALUE, cv2.THRESH_BINARY)
        # Contour detection stage
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Calculate all the areas  of the detected contours
        areas = [cv2.contourArea(c) for c in contours]
        for idx, val in enumerate(areas):
            # Human blob filtration stage
            if MIN_CNTR_HUMN_AREA <= val <= MAX_CNTR_HUMN_AREA:
                cntr = contours[idx]
                # Fitting bounding boxes over our contours of interest (humans)
                x,y,w,h = cv2.boundingRect(cntr)
                # Final bounding box coordinates
                xmin = x
                ymin = y
                xmax = x+w
                ymax = y+h
                # Human bounding box instances detction stage
                cv2.rectangle(grayscale_img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
                cv2.rectangle(heatmap_img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        # Upsample the output detection images if the upsample ration exists
        if upsample_ratio is not None:
            grayscale_img = cv2.resize(grayscale_img, (upsample_ratio*grayscale_img.shape[1],
                                       upsample_ratio*grayscale_img.shape[0]),
                                       interpolation = cv2.INTER_NEAREST)
            heatmap_img = cv2.resize(heatmap_img, (upsample_ratio*heatmap_img.shape[1],
                                     upsample_ratio*heatmap_img.shape[0]),
                                     interpolation = cv2.INTER_NEAREST)
        # Write the detection images to the disk
        cv2.imwrite(os.path.join(graysDetDir, (os.path.splitext(os.path.basename(grayscale))[0] + '.png')), grayscale_img)
        cv2.imwrite(os.path.join(heatmapsDetDir, (os.path.splitext(os.path.basename(heatmap))[0] + '.png')), heatmap_img)
    return graysDetDir, heatmapsDetDir