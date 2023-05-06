import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib as plt
import cv2 as cv




"""
detects white makings that represent errors in the printing.  
"""
def detect_error(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    colors  = set(image.getcolors(image.size[0] * image.size[1]))
    threshold= 230
    count = 0
    for i in colors:
        if i[1][0]  >threshold and i[1][2] >threshold and i[1][1]>threshold:
            count+=1
    if  count>0 : #if there is 
            return True
    return False


files  = ["blacktest2.png", "black.png"]

for f in files: 
    print(detect_error(f))