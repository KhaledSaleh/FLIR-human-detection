"""Utilities for parsing and creating the dataset images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Dependency imports
import os
import glob
import json
import itertools
import numpy as np
import cv2
import re
from tqdm import tqdm


# The FLIR camera returned video packet width and height
FLIR_LEPTON_WIDTH = 80
FLIR_LEPTON_HEIGHT = 60
FLIR_LEPTON_FRAME_SIZE = FLIR_LEPTON_WIDTH*FLIR_LEPTON_HEIGHT
# The heatmap colorization value in OpenCV
HEAT_COLORMAP = 2
# Frame rate per second for the generated videos
FPS = 1

def parse_txt_dataValues(in_txtFile):
    """Parse the txt file of the radimoetry data from the two FLIR cameras.
    
    Args:
        in_txtFile: the .txt file which contains the dataValues coming from FLIR cameras

    Returns:
        The raw thermal two FLIR camera frames stitched.
    """
    with open(in_txtFile) as fp:
        txt_dict = json.load(fp)
        dataValues = txt_dict[0].get('dataValues').split(' ')
        dataValues = np.asarray(dataValues, dtype=np.float32)
        # Reshape to the two celsius radimoetry frames
        rad_flir1 = np.reshape(dataValues[:FLIR_LEPTON_FRAME_SIZE,],
                                (FLIR_LEPTON_HEIGHT, FLIR_LEPTON_WIDTH))
        rad_flir2 = np.reshape(dataValues[FLIR_LEPTON_FRAME_SIZE:,],
                                (FLIR_LEPTON_HEIGHT, FLIR_LEPTON_WIDTH))
        # Stitch the two frames side by side
        rad_flir_stitch = np.concatenate((rad_flir1, rad_flir2), axis=1)
        return rad_flir_stitch


def decode_as_grayscale(rad_flir, upsample_ratio=None, v_flip=False):
    """Decode the input radiometry frame of the FLIR camera as grayscale image.

    If the upsample_ratio set it will return an upsampled image with the ration equal
    to that value and if the v_flip flag is True the returned grayscale images will be
    vertically flipped.

    Args:
        rad_flir: the raw thermal stitched image from FLIR cameras.
        upsample_ration: ratio value to upsample the returned images with.
        v_flip: flag to indicate whether to flip the returned image
        vertically or not.

    Returns:
        The grayscale image from the raw thermal stitched image.
    """
    grayscale_img = np.zeros_like(rad_flir)
    # Normalize the raw radiometry frame values between 0..255
    cv2.normalize(rad_flir, grayscale_img, 0, 255, cv2.NORM_MINMAX)
    # Resize the output grayscale image with the upsample factor if exist
    if upsample_ratio is not None:
        grayscale_img = cv2.resize(grayscale_img, (upsample_ratio*grayscale_img.shape[1],
                              upsample_ratio*grayscale_img.shape[0]),
                              interpolation = cv2.INTER_NEAREST)
    # Flip the returned grayscale image vertically if True
    if v_flip:
        cv2.flip(grayscale_img, 0, grayscale_img)
    return grayscale_img


def decode_as_heatmap(grayscale_flir):
    """Decode the input grayscale thermal image as a colorized heatmap image.
    
    Args:
       grayscale_flir: the grayscale image of the stitched two FLIR camera frames.

    Returns:
        The colorized grayscale image in heat color map.
    """
    # Create a placeholder image equal in size to the input grayscale image with 3 channels (R,G,B) 
    heatmap_flir = np.zeros((grayscale_flir.shape[0], grayscale_flir.shape[1], 3), dtype=np.uint8)
    # Add new axis to the numpy array of the grayscale image, since it's only 2D
    heatmap_flir[:, :, :] = grayscale_flir[..., np.newaxis]
    # Apply the heatmap clorization to the grayscale image
    heatmap_flir = cv2.applyColorMap(heatmap_flir, HEAT_COLORMAP)
    return heatmap_flir


def numerical_sort(value):
    """Sort the parsed files from disk numerically rather than alphabitcally."""
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def walk_dir(folder, ext):
    """Walk through all files.ext in a directory."""
    for dirpath, dirs, files in os.walk(folder):
        for filename in sorted (files, key = numerical_sort):
            if filename.endswith(ext):
                yield os.path.abspath(os.path.join(dirpath, filename))


def total_files_counter(inp_path, ext):
    """Count the number of files.ext in a given input path directory."""
    filecounter = 0
    for dirpath, dirs, files in os.walk(inp_path):
        for filename in files:
            if filename.endswith(ext):
                filecounter += 1
    return filecounter


def generate_dataset_images(input_TxtPth, out_ImgsPth):
    """Generate the raw thermal, grayscale and heatmap images from *.txt files.

    The thermal, grayscale and heatmap images will be write to the directories
    out_ImgsPth/thermals/, out_ImgsPth/grays/, out_ImgsPth/heatmaps/ respectively, 
    if any directories were not created, they will be created firstly.

    Args:
        input_TxtPth: the path to the directory of the *.txt files.
        out_ImgsPth: The path to which the generated image directories will exist.

    Returns:
        The path to the three directories of the generated images.
    """
    # Check whether the input txt files path exists or not.
    if not os.path.exists(input_TxtPth):
        print (input_TxtPth + ' is not a valid path')
        exit(-1)
    # Count the total number of *.txt files in the input_TxtPth
    file_counter = total_files_counter(input_TxtPth, '.txt')
    # Check whether there are txt files avaialable in the input path
    if not file_counter:
        print (input_TxtPth + ' contains no txt files')
        exit(-1)
    # Output directories for the raw thermal, grayscale and heatmap images
    heatmapImgsDir = os.path.join(out_ImgsPth, 'heatmaps')
    thermalImgsDir = os.path.join(out_ImgsPth, 'thermals')
    grayImgsDir = os.path.join(out_ImgsPth, 'grays')
    # Check if heatmaps directory is created or not
    if not os.path.exists(heatmapImgsDir):
        os.makedirs(heatmapImgsDir)
    # Check if raw thermals directory is created or not
    if not os.path.exists(thermalImgsDir):
        os.makedirs(thermalImgsDir)
    # Check if gryascale images directory is created or not
    if not os.path.exists(grayImgsDir):
        os.makedirs(grayImgsDir)
    # Name index files iterator
    nameIdxIterator = 0
    # Loop over all the txt files to parse them with a prograss bar
    for txtFile in tqdm(walk_dir(input_TxtPth, '.txt'), total=file_counter, desc='Generating dataset images'):
        # Six digit incremental number for each image's name 
        fileName = '%06d' % (nameIdxIterator)
        raw_thermal_img = parse_txt_dataValues(txtFile)
        grayscale_img = decode_as_grayscale(raw_thermal_img)
        heatmap_img = decode_as_heatmap(grayscale_img)
        # Write each image to its corresponding directory
        cv2.imwrite(os.path.join(thermalImgsDir, (fileName + '.png')), raw_thermal_img)
        cv2.imwrite(os.path.join(grayImgsDir, (fileName + '.png')), grayscale_img)
        cv2.imwrite(os.path.join(heatmapImgsDir, (fileName + '.png')), heatmap_img)
        nameIdxIterator += 1
    return thermalImgsDir, grayImgsDir, heatmapImgsDir


def generate_detctions_video(inp_graysDetPth, inp_heatmapsDetPth, det_outVidsPth, video_name, upsample_ratio = None):
    """Generate videos (grayscale/heatmaps) of the output detection images.
    
    Args:
        inp_graysDetPth: the path to the detection grayscale images.
        inp_heatmapsDetPth: the path to the detection heatmap images.
        det_outVidsPth: the path to the detection result videos.
        video_name: name for the generated videos.
        upsample_ratio: an optional ratio value to upsample the returned images with.

    Returns:
        The path to the the generated human detection videos.
    """ 
    # Check if the input grayscale images path exists or not
    if not os.path.exists(inp_graysDetPth):
        print (inp_graysDetPth + ' is not a valid path')
        exit(-1)
    # Check if the input heatmap images path exists or not
    if not os.path.exists(inp_heatmapsDetPth):
        print (inp_heatmapsDetPth + ' is not a valid path')
    # Count the total number of all *.png images in the inp_graysDetPth
    image_counter = total_files_counter(inp_graysDetPth, '.png')
    # Check whether there are grayscale images avaialable in the inp_graysDetPth
    if not image_counter:
        print (inp_graysDetPth + ' contains no images')
        exit(-1)
    # Initialize video names and the FourCC codec
    grayscale_video_writer = None
    heatmap_video_writer = None
    grayscale_det_video_path = os.path.join(det_outVidsPth, (video_name + '_grayscale.mp4'))
    heatmap_det_video_path = os.path.join(det_outVidsPth, (video_name + '_heatmap.mp4'))
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    # Iterate over all the gryascale and heatmap detection images
    for grayDet, heatmapDet in tqdm(itertools.izip(walk_dir(inp_graysDetPth,'.png'),
                                             walk_dir(inp_heatmapsDetPth, '.png')),
                                             total=image_counter, desc = 'Generating human detection videos'):
        # Read both grayscale and heatmap images
        grayscale_detImg = cv2.imread(grayDet,cv2.IMREAD_GRAYSCALE)
        heatmap_detImg = cv2.imread(heatmapDet)
        # Upsample the output detection images if the upsample ration exists
        if upsample_ratio is not None:
            grayscale_detImg = cv2.resize(grayscale_detImg, (upsample_ratio*grayscale_detImg.shape[1],
                                       upsample_ratio*grayscale_detImg.shape[0]),
                                       interpolation = cv2.INTER_NEAREST)
            heatmap_detImg = cv2.resize(heatmap_detImg, (upsample_ratio*heatmap_detImg.shape[1],
                                     upsample_ratio*heatmap_detImg.shape[0]),
                                     interpolation = cv2.INTER_NEAREST)
        # Configure the video writer parameter based on the first image 
        if grayscale_video_writer is None:
            (h, w) = grayscale_detImg.shape[:2]
            grayscale_video_writer =  cv2.VideoWriter(grayscale_det_video_path,fourcc,FPS,(w,h))
            heatmap_video_writer =  cv2.VideoWriter(heatmap_det_video_path,fourcc,FPS,(w,h))
        # Starting writing the two videos to the disk
        grayscale_detImg_3channels = np.zeros_like(heatmap_detImg)
        grayscale_detImg_3channels[:, :, :] = grayscale_detImg[..., np.newaxis]
        grayscale_video_writer.write(grayscale_detImg_3channels)
        heatmap_video_writer.write(heatmap_detImg)
    # Release each video writer before the return
    grayscale_video_writer.release()
    heatmap_video_writer.release()