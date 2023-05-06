import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib as plt
import cv2 as cv


def detect_error(filename):
    # Open the PNG image
    image = Image.open(filename)
    # Convert the image to RGB mode (if it's not already in RGB mode)
    image = image.convert("RGB")
    #cv.drawContours(img, contours, -1, (0, 255, 0), 2)
    # Get the colors in the image
    ans  = set(image.getcolors(image.size[0] * image.size[1]))
    lower_bound = 50
    upper_bound = 255
    count = 0 
    for i in ans:
        if i[1][0]==0 and i[1][2] == 0 and i[1][1]>lower_bound and i[1][1]<=upper_bound:
            count+=1

    if  count>1 :
            return True
    return False


files  = ["protector_left_green.png","black.png"]

for f in files: 
     print(detect_error(f))