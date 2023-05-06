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
    mask = cv.imread("right_template.png", cv.IMREAD_UNCHANGED)


    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    # 1338x445 is the dimension of the template
    bottom_right = (top_left[0] + 1338, top_left[1] + 445)

    cropped_image = img[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]

    right_pic = cropped_image[:, -68:-5]
    right_pic = cv.GaussianBlur(right_pic, (3, 3), 0)

    right_pic = cv.convertScaleAbs(right_pic, 0.5, 10)
    right_pic = cv.cvtColor(right_pic, cv.COLOR_GRAY2RGBA)
    mask = mask[:445, :63]
    right_pic = cv.subtract(right_pic, mask)
    

    arr = np.where(right_pic==0, np.nan, right_pic)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)


    
    anomalies = (np.abs(arr - mean) / std >= 2.0).any(axis=2)
    mask_u8 = anomalies.astype(np.uint8) * 255
    mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
    mask = cv.imread("right_template_nt.png", cv.IMREAD_COLOR)
    mask = mask[:445, :63]
    mask = cv.bitwise_not(mask)
    mask_u8 = cv.bitwise_and(mask_u8, mask)

    cv.imshow("mask", mask_u8)
    cv.waitKey(100)
