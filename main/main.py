import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def crop_img(img,img_low_bit):

    template = cv.imread('cropped_manual.tif', cv.IMREAD_GRAYSCALE)

    res = cv.matchTemplate(img_low_bit, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    print(top_left)
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
    print(top_left)
    #cv.rectangle(cropped_image,top_left, bottom_right, (255, 0, 0), 2)

    cropped_image = cropped_image[65:-65,78:-80]

    return cropped_image

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

def detect_side(isBottom: bool, img):
    mask = None
    if (isBottom):
        mask = cv.imread("left_template.png", cv.IMREAD_UNCHANGED)
    else:
        mask = cv.imread("top_template.png", cv.IMREAD_UNCHANGED)

    img = crop(img)
    
    pic = img[:32, 78:-77]
     
    if isBottom:
         pic = img[-32:, 78:-77]

    #blurs template to remove small spots
    pic = cv.GaussianBlur(pic, (3, 3), 0)
    #make picture brighter
    pic = cv.convertScaleAbs(pic, 0.5, 10)
    pic = cv.cvtColor(pic, cv.COLOR_GRAY2RGBA)
    #remove blank space with mask 
    pic = cv.subtract(pic, mask)
    

    arr = np.where(pic==0, np.nan, pic)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)

    
    #check for pixels above the standard deviation 
    # anomalies = (np.abs(arr - mean) / std >= 2.0).any(axis=2)
    # mask_u8 = anomalies.astype(np.uint8) * 255
    # mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
    # if isBottom:
    #     mask = cv.imread("left_template_nt.png", cv.IMREAD_COLOR)
    # else:
    #     mask = cv.imread("right_template_nt.png", cv.IMREAD_COLOR)
    # mask = mask[:445, :63]
    # mask = cv.bitwise_not(mask)
    # #remove blankspaces again
    # mask_u8 = cv.bitwise_and(mask_u8, mask)

    return mask
  
if __name__ == "__main__":
  pass