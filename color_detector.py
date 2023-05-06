import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib as plt
import cv2 as cv



"""
detects white markings that represent errors in the printing.  
"""
def detect_error(filename):
    image = Image.open(filename)
    image = np.array(image)

    threshold=200
    white_pixels = np.count_nonzero(image > threshold)  # count white pixels
    if white_pixels:
        return True
    return False


files  = [".png"]

for f in files: 
    print(detect_error(f))