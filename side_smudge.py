import cv2 as cv
import numpy as np
from cropIn import crop

def detect_side(isLeft: bool, img):
    mask = None
    if (isLeft):
        mask = cv.imread("left_template.png", cv.IMREAD_UNCHANGED)
    else:
        mask = cv.imread("right_template.png", cv.IMREAD_UNCHANGED)

    img = crop(img)
    
    pic = img[:, -68:-5]
    if (isLeft):
        pic = img[:, 5:68]

    pic = cv.GaussianBlur(pic, (3, 3), 0)
    pic = cv.convertScaleAbs(pic, 0.5, 10)
    pic = cv.cvtColor(pic, cv.COLOR_GRAY2RGBA)
    mask = mask[:445, :63]
    pic = cv.subtract(pic, mask)
    

    arr = np.where(pic==0, np.nan, pic)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)

    
    anomalies = (np.abs(arr - mean) / std >= 2.0).any(axis=2)
    mask_u8 = anomalies.astype(np.uint8) * 255
    mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
    if (isLeft):
        mask = cv.imread("left_template_nt.png", cv.IMREAD_COLOR)
    else:
        mask = cv.imread("right_template_nt.png", cv.IMREAD_COLOR)
    mask = mask[:445, :63]
    mask = cv.bitwise_not(mask)
    mask_u8 = cv.bitwise_and(mask_u8, mask)

    return mask_u8
