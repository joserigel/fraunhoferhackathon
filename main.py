import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import crop_img
import math

input_folder_path = './PrePro/input'
output_folder_path = './PrePro/output'


import numpy as np


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
numbers=[]

for filename in glob.glob(input_folder_path + '/*.tif'):

    img = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    img_low_bit = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)

    img = crop_img(img,img_low_bit)


    #img = img[40:-40,80:-80]


    kernel_size = 9
    blur_gray = cv2.GaussianBlur(img , (kernel_size, kernel_size), 0)

    #plt.imshow(blur_gray)
    #plt.show()


    blur_gray = cv2.adaptiveThreshold(cv2.convertScaleAbs(blur_gray / 16),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,31,2)

    #plt.imshow(blur_gray)
    #plt.show()

    contour, hier = cv2.findContours(blur_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(blur_gray, [cnt], 0, 255, -1)

    blur_gray = remove_small_regions(blur_gray, 60)

    #plt.imshow(blur_gray)
    #plt.show()



    low_threshold = 50
    high_threshold = 255
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    #plt.imshow(edges)
    #plt.show()



    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    removed_lines = np.copy(blur_gray)  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 40,minLineLength = 100,maxLineGap = 30)

    angle_tol = 5
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            if(abs(abs(angle)-45) < angle_tol):
                cv2.line(removed_lines, (x1, y1), (x2, y2), 0, 1)



    #cv2.imshow('image', removed_lines)
    #cv2.waitKey()

    result = remove_small_regions(removed_lines,60)
    line_mask = cv2.bitwise_not(result)
    line_mask = remove_small_regions(line_mask, 60)

    # Creating kernel
    kernel = np.ones((3, 3), np.uint8)

    # Using cv2.erode() method
    line_mask = cv2.erode(line_mask, kernel)

    deadzone_y = 0
    deadzon_x = 0

    #line_mask[0:deadzone_y,:] = 0
    #line_mask[-deadzone_y:-1,:] = 0


    #line_mask[:,0:deadzon_x] = 0
    #line_mask[:,-deadzon_x:-1] = 0


    backtorgb = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2RGB)
    backtorgb[:, :, 0] = np.zeros([backtorgb.shape[0], backtorgb.shape[1]])
    backtorgb[:, :, 1] = np.zeros([backtorgb.shape[0], backtorgb.shape[1]])

    added_image = cv2.addWeighted(cv2.cvtColor(cv2.convertScaleAbs(img /16),cv2.COLOR_GRAY2RGB), 0.8, backtorgb, 0.3, 0)

    #cv2.imshow('image', added_image)
    #cv2.waitKey(100)

    line_mask_red = line_mask / 255
    residual = np.multiply(line_mask_red,img)

    maske_arr = np.invert( np.array(line_mask,dtype=bool) )
    mx = np.ma.masked_array(img, mask=maske_arr)

    data = mx[mx.mask == False]

    numbers.append(data)






plt.hist(np.concatenate(numbers).ravel(), density=False, bins=4095)  # density=False would make counts
plt.ylabel('Probability')
plt.xlabel('Data');
plt.show()