import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import crop_img
import math
import scipy
from tqdm import tqdm
from scipy import ndimage, datasets

input_folder_path = './PrePro/input'
output_folder_path = './PrePro/output'
from scipy.signal import find_peaks
from scipy import spatial


import numpy as np

def scan_line(image_blurred, image):
    res = []




    full_img_45_blurred = ndimage.rotate(image_blurred, 45, reshape=True)
    full_img_45 = ndimage.rotate(image, 45, reshape=True)*0


    image_s = full_img_45_blurred.shape


    y_sum = np.sum(full_img_45_blurred,0)
    x_sum = np.sum(full_img_45_blurred,1)


    valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=10000)
    valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=10000)

    for i in valleys_x:
        cv2.line(full_img_45, (0, i), (image_s[0],i) , 4095, 2)


    for i in valleys_y:
        cv2.line(full_img_45, ( i,0), (i,image_s[0]) , 4095, 2)

    full_img_45_done = ndimage.rotate(full_img_45, -45, reshape=True)[590:904+1,158:1337]
    #Oben links (158,590) Unten Rechts (1336,904)


    cv2.imshow('image',full_img_45_done)
    cv2.waitKey(30)

    #plt.imshow(full_img_45_done)
    #plt.plot(x_sum)
    #plt.plot(valleys_x, x_sum[valleys_x], "x")
    #plt.show()



def remove_small_regions(image,thresh):
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 4, cv2.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= thresh:  # keep
            result[labels == i + 1] = 255

    return result


def analyze_grid(filename,main_folder,refernce_grid):

    img = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    img_low_bit = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)

    img = crop_img(img,img_low_bit)


    #img = img[40:-40,80:-80]


    kernel_size = 21
    blur_gray = cv2.GaussianBlur(img , (kernel_size, kernel_size), 0)

    scan_line(blur_gray,img)

    return None

    """
    #plt.imshow(blur_gray)
    #plt.show()


    deadzone_y = 40
    deadzone_x = 80

    refernce_img = img[deadzone_y:-deadzone_y, deadzone_x:-deadzone_x]

    refernce_grid = refernce_grid[deadzone_y:-deadzone_y, deadzone_x:-deadzone_x]

    line_mask = line_mask[deadzone_y:-deadzone_y, deadzone_x:-deadzone_x]

    backtorgb = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2RGB)
    backtorgb[:, :, 0] = np.zeros([backtorgb.shape[0], backtorgb.shape[1]])
    backtorgb[:, :, 1] = np.zeros([backtorgb.shape[0], backtorgb.shape[1]])

    added_image = cv2.addWeighted(cv2.cvtColor(cv2.convertScaleAbs(refernce_img /16),cv2.COLOR_GRAY2RGB), 0.8, backtorgb, 0.3, 0)

    cv2.imshow('image', added_image)
    cv2.waitKey(100)
    """


