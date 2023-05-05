import cv2 as cv
import PIL
import numpy as np

img = cv.imread('cam.tif', cv.IMREAD_GRAYSCALE)
template = cv.imread('cropped_manual.tif', cv.IMREAD_GRAYSCALE)


res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = min_loc
print(top_left)
# 1338x445 is the dimension of the template
bottom_right = (top_left[0] + 1338, top_left[1] + 445)

cropped_image = img[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]



template_grid = cv.imread('cropped_grid_area.tif', cv.IMREAD_GRAYSCALE)
bound = cv.matchTemplate(cropped_image, template_grid, cv.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(bound)

# 1185x312 dimension of grid area template
top_left = min_loc
bottom_right = (top_left[0] + 1185, top_left[1] + 312)
print(top_left)
cv.rectangle(cropped_image,top_left, bottom_right, 255, 2)


cv.imshow('image', cropped_image)
cv.rectangle(img,top_left, bottom_right, 255, 2)
cv.waitKey(0)