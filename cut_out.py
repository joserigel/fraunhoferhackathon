import cv2 as cv
import numpy as np

import glob
import os


input_folder_path = './PrePro/input'
output_folder_path = './PrePro/output'

i = 0
for filename in glob.glob(input_folder_path + '/*.tif'):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    template = cv.imread('cropped_manual.tif', cv.IMREAD_GRAYSCALE)


    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    # 1338x445 is the dimension of the template
    bottom_right = (top_left[0] + 1338, top_left[1] + 445)

    cropped_image = img[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]
    cropped_image = cropped_image[65:-65, 78:-80]
    # 1185x312 dimension of grid area template
    top_left = min_loc
    top_left = (top_left[0], top_left[1])
    bottom_right = (top_left[0], top_left[1])
    
    cv.imshow('image', cropped_image)
    cv.imwrite(output_folder_path + "/" + str(i) + ".tiff", cropped_image)
    cv.waitKey(50)
    i+=1