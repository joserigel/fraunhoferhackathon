import cv2 as cv
import PIL
import numpy as np

img = cv.imread('../cam.tif', cv.IMREAD_GRAYSCALE)
template_inferior = cv.imread("teeth_inferior.tif", cv.IMREAD_GRAYSCALE)
if template_inferior is not None: 
    print("ok") 
else: 
    print("file could not be loaded")

print(img.shape)
print(template_inferior.shape)


res = cv.matchTemplate(img, template_inferior, cv.TM_SQDIFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

template_w = template_inferior.shape[1]
template_h = template_inferior.shape[0]
top_left =max_loc 


bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

cropped_image = img[top_left[1]:(top_left[1] + 1190), top_left[0]:(top_left[0] + 1420)]
cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
cv.imshow('Result', img)
cv.waitKey(0)
#cv.destroyAllWindows()

#cv.imshow('image', cv.convertScaleAbs(cropped_image)/16)
#cv.rectangle(img,top_left, bottom_right, 255, 2)
#cv.waitKey(0)
"""
top_left = min_loc
top_left = (top_left[0] + 10, top_left[1] + 10)
bottom_right = (top_left[0] + 1160, top_left[1] + 280)
print(top_left)
cv.rectangle(cropped_image,top_left, bottom_right, (255, 0, 0), 2)

cv.imshow('image', cv.convertScaleAbs(cropped_image)/16)
cv.rectangle(img,top_left, bottom_right, 255, 2)
cv.waitKey(0)
"""

templates = [
    "1.tif", 
    "2.tif", 
    "3.tif", 
    "4.tif", 
    "5.tif", 
    "6.tif", 
    "7.tif", 
    "8.tif", 
    "9.tif", 
]


