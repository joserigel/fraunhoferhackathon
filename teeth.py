import cv2 as cv
import numpy as np
from cropIn import crop

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
