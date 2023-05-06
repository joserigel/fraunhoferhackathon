import cv2 as cv
import numpy as np

import glob
import os

input_folder_path = './PrePro/input'
output_folder_path = './PrePro/output'

# for filename in glob.glob(input_folder_path + '/*.tif'):
#     img = cv2.imread(filename, -1)  # The -1 means read as is, including as a 12-bit image if it is one
#     largest_square = find_largest_square(img)
#     if largest_square is not None:
#         cv2.drawContours(img, [largest_square], -1, (4095), 3)  # 4095 is the max value for 12-bit images
#     output_filename = os.path.join(output_folder_path, os.path.basename(filename))
#     cv2.imwrite(output_filename, img)

for filename in glob.glob(input_folder_path + '/*.tif'):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    template = cv.imread('cropped_manual.tif', cv.IMREAD_GRAYSCALE)


    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    # 1338x445 is the dimension of the template
    bottom_right = (top_left[0] + 1338, top_left[1] + 445)

    cropped_image = img[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]


    # 1185x312 dimension of grid area template
    top_left = min_loc
    top_left = (top_left[0], top_left[1])
    bottom_right = (top_left[0], top_left[1])
    t, cropped_image = cv.threshold(cv.convertScaleAbs(cropped_image*16), 130, 255, cv.THRESH_BINARY)
    
    #cv.rectangle(cropped_image,(70, 65), (1500, 300), 0, 2)
    
    #cropped_image = cv.cvtColor(cropped_image, cv.COLOR_GRAY2RGB)
    cropped_image = cv.bitwise_not(cropped_image)
    #Top
    cropped_image = cv.rectangle(cropped_image, (0, 0), (1500, 80), 0, -1)
    #Bottom
    cropped_image = cv.rectangle(cropped_image, (0, 380), (15000, 4500), 0, -1)
    #Left
    cropped_image = cv.rectangle(cropped_image, (0, 0), (85, 4500), 0, -1)
    #Right
    cropped_image = cv.rectangle(cropped_image, (1258, 0), (15000, 4500), 0, -1)
    #grid_area = cropped_image[65:390, 70:1267]
    contours, heirarchy = cv.findContours(cropped_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cropped_image = cv.cvtColor(cropped_image, cv.COLOR_GRAY2RGB)
    cv.drawContours(cropped_image, contours, 0, (255,0,0), 3)
    #print(cnts)



    cv.imshow('image', cropped_image)
    cv.waitKey(50)