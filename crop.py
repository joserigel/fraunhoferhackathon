import cv2 as cv
import PIL
import numpy as np

def crop_img(img,img_low_bit):

    template = cv.imread('cropped_manual.tif', cv.IMREAD_GRAYSCALE)


    res = cv.matchTemplate(img_low_bit, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc

    # 1338x445 is the dimension of the template
    bottom_right = (top_left[0] + 1338, top_left[1] + 445)

    cropped_image_low_bit = img_low_bit[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]
    cropped_image = img[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]



    template_grid = cv.imread('cropped_grid_area.tif', cv.IMREAD_GRAYSCALE)
    bound = cv.matchTemplate(cropped_image_low_bit, template_grid, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(bound)

    # 1185x312 dimension of grid area template
    top_left = min_loc
    top_left = (top_left[0] + 10, top_left[1] + 10)
    bottom_right = (top_left[0] + 1160, top_left[1] + 280)

    #cv.rectangle(cropped_image,top_left, bottom_right, (255, 0, 0), 2)

    cropped_image = cropped_image[65:-65,78:-80]

    return cropped_image

