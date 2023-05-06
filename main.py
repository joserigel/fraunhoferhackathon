import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

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


for filename in glob.glob(input_folder_path + '/*.tif'):

    img = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)

    plt.imshow(img)
    plt.show()

    img = img[20:-20,40:-40]
    img = cv2.convertScaleAbs(img*20)





    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    t,blur_gray = cv2.threshold(blur_gray,60,255,cv2.THRESH_BINARY)






    low_threshold = 50
    high_threshold = 255
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)




    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    removed_lines = np.copy(blur_gray)   # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)


    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(removed_lines, (x1, y1), (x2, y2), 0, 1)


    result = remove_small_regions(removed_lines,60)

    line_mask = cv2.bitwise_not(result)



    line_mask = remove_small_regions(line_mask, 60)

    # Creating kernel
    kernel = np.ones((3, 3), np.uint8)

    # Using cv2.erode() method
    line_mask = cv2.erode(line_mask, kernel)




    backtorgb = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2RGB)
    backtorgb[:, :, 0] = np.zeros([backtorgb.shape[0], backtorgb.shape[1]])
    backtorgb[:, :, 1] = np.zeros([backtorgb.shape[0], backtorgb.shape[1]])

    added_image = cv2.addWeighted(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB), 0.8, backtorgb, 0.3, 0)

    plt.imshow(added_image)
    plt.show()

