import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib as plt
import cv2 as cv




"""
checks for the green makings in the heatmaps; which signalize that a priting error was detected. 
helper method to find out wether the generated heatmap contains an error signal or not. 
"""
def detect_error(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    ans  = set(image.getcolors(image.size[0] * image.size[1]))
    lower_bound = 70 #ideally lower than 70
    upper_bound = 255 #keep it a 255 to be sure
    gray_threshold = 40 #keep it lower than 50
    for i in ans:
        if i[1][0]<gray_threshold and i[1][2] < gray_threshold and i[1][1]>lower_bound and i[1][1]<=upper_bound:
            count+=1

    if  count>1 :
            return True
    return False


files  = ["protector_left_green.png","black.png", "grid_greens.png", "grid_black.png"]

for f in files: 
    print(detect_error(f))