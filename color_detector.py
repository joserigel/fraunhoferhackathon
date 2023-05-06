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

    colors, counts = np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    colors =[list(color) for color in colors] 
    threshold= 230
    count = 0
    for i in colors:
        if i[0]  >threshold and i[2] >threshold and i[1]>threshold:
            count+=1
    if  count>0 : 
            return True
    return False


files  = ["blacktest2.png", "black.png"]

for f in files: 
    print(detect_error(f))