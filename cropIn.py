import cv2 as cv
import PIL
import numpy as np


def crop(img):
    template = cv.imread('cropped_manual.tif', cv.IMREAD_GRAYSCALE)

    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    
    # print(top_left)
    # 1338x445 is the dimension of the template
    # bottom_right = (top_left[0] + 1338, top_left[1] + 445)

    cropped_image = img[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]

    return cropped_image    