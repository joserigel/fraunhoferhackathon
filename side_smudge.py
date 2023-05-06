import cv2 as cv
from PIL import Image
import numpy as np


import glob
import os


input_folder_path = './PrePro/input'
output_folder_path = './PrePro/output'

for filename in glob.glob(input_folder_path + '/*.tif'):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    template = cv.imread('cropped_manual.tif', cv.IMREAD_GRAYSCALE)
    mask = cv.imread("left_template.png", cv.IMREAD_UNCHANGED)


    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    # 1338x445 is the dimension of the template
    bottom_right = (top_left[0] + 1338, top_left[1] + 445)

    cropped_image = img[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]

    left_pic = cropped_image[:, 5:68]
    left_pic = cv.convertScaleAbs(left_pic, 0.5, 10)
    left_pic = cv.cvtColor(left_pic, cv.COLOR_GRAY2RGBA)
    mask = mask[:445, :63]
    left_pic = cv.subtract(left_pic, mask)
    

    arr = np.where(left_pic==0, np.nan, left_pic)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    #print(mean, std)
    anomalies = (np.abs(arr - mean) / std >= 1.0).any(axis=2)
    mask_u8 = anomalies.astype(np.uint8) * 255
    mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
    mask = cv.imread("left_template_nt.png", cv.IMREAD_COLOR)
    mask = mask[:445, :63]
    mask = cv.bitwise_not(mask)
    mask_u8 = cv.bitwise_and(mask_u8, mask)

    #mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
    #print(np.shape(mask_u8))
    #binMask = cv.bitwise_not(binMask)[:445, :63]
    #binMask = cv.threshold(binMask, 100, 255, cv.THRESH_BINARY)
    
    #print(np.shape(binMask))
    #smask_u8 = cv.subtract(mask_u8, mask)
    #grayscale = (np.abs(yuv - mean) / std / 5).max(axis=2)

    cv.imshow("mask", mask_u8)
    cv.waitKey(0)
