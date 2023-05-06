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


    full_img_45_blurred = ndimage.rotate(image_blurred, 45, reshape=True)
    full_img_45 = ndimage.rotate(image, 45, reshape=True)*0


    image_s = full_img_45_blurred.shape


    y_sum = np.sum(full_img_45_blurred,0)
    x_sum = np.sum(full_img_45_blurred,1)


    valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=50000)
    valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=50000)

    for i in valleys_x:
        cv2.line(full_img_45, (0, i), (image_s[0],i) , 255, 1)


    for i in valleys_y:
        cv2.line(full_img_45, ( i,0), (i,image_s[0]) , 255, 1)

    #cv2.imshow('image', full_img_45)
    #cv2.waitKey(30)

    full_img_45_done = ndimage.rotate(full_img_45, -45, reshape=True)[590:904+1,158:1338]
    #Oben links (158,590) Unten Rechts (1336,904)

    ret, full_img_45_done_thresh = cv2.threshold(full_img_45_done, 127, 4095, cv2.THRESH_BINARY)

    full_img_45_done_thresh = cv2.convertScaleAbs(full_img_45_done_thresh)



    #plt.imshow(full_img_45_done)
    #plt.plot(x_sum)
    #plt.plot(valleys_x, x_sum[valleys_x], "x")
    #plt.show()

    return full_img_45_done_thresh



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


def analyze_grid(filename,main_folder):

    img = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    img_low_bit = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)

    img = crop_img(img,img_low_bit)




    kernel_size = 11
    blur_gray = cv2.GaussianBlur(img , (kernel_size, kernel_size), 0)

    line_mask = scan_line(blur_gray,img)


    img = img[40:-40,80:-80]
    line_mask= line_mask[40:-40,80:-80]


    img_median_blurred = img



    maske_arr = np.array(line_mask, dtype=bool)
    mx_only_squares = np.ma.masked_array(img_median_blurred, mask=maske_arr)
    data = mx_only_squares[mx_only_squares.mask == False]

    std_dev = np.std(data)
    av = np.average(data)

    maske_arr = np.invert(np.array(line_mask, dtype=bool))
    mx = np.ma.masked_array(img_median_blurred, mask=maske_arr)
    data = mx[mx.mask == False]

    res = np.count_nonzero(data > av-0.5*std_dev)

    maske_arr = np.array(line_mask, dtype=bool)
    raw = np.multiply(maske_arr, img)

    heat = np.array( (raw > av-0.5*std_dev)*255, dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(heat, kernel, iterations=1)
    img_dilation_border = cv2.dilate(heat, kernel, iterations=3)

    img_dilation = np.subtract(img_dilation_border,img_dilation)

    print (res)

    if(res > 80000000):
        plt.imshow(img, cmap='gray')  # I would add interpolation='none'
        plt.imshow(img_dilation, cmap='rainbow', alpha=0.4 * (img_dilation > 0))  # interpolation='none'
        plt.show()

    return res
