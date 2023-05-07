import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from crop import crop_img
import math
import scipy
from tqdm import tqdm
from scipy import ndimage, datasets
import matplotlib.colors as mcolors
from scipy.interpolate import splrep, BSpline


input_folder_path = './PrePro/input'
output_folder_path = './PrePro/output'
from scipy.signal import find_peaks, peak_widths
from scipy import spatial

colors = [(0, '#00ff00'), (1, '#00ff00')]
colors = mcolors.LinearSegmentedColormap.from_list("", colors)

import numpy as np

def draw_line_topl_br(a,img,shape,w=1):
    for i in a:
        cv2.line(img, (0, i), (shape[0], i), 255, w)

def draw_line_bl_topr(a,img,shape,w=1):
    for i in a:
        cv2.line(img,  ( i,shape[1]), (i,0) , 255, w)



def scan_line(image_blurred, image):

    res=[]






    tolerance = 1
    base = 45
    tries = 21



    for i in np.linspace(base - tolerance, base + tolerance, tries):
        print(i)

        full_img_45_blurred = ndimage.rotate(image_blurred, i, reshape=True)

        y_sum = np.sum(full_img_45_blurred,0)
        x_sum = np.sum(full_img_45_blurred,1)


        valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=50000)
        valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=50000)

        valleys_x = valleys_x
        valleys_y = valleys_y

        widthsx, h_eval, left_ips, right_ips = peak_widths(-x_sum, valleys_x, rel_height=0.5)
        widthsy, h_eval, left_ips, right_ips = peak_widths(-y_sum, valleys_y, rel_height=0.5)
        res.append((np.sum(np.concatenate((widthsx,widthsy), axis=None)),i))

    smallest = min(res, key=lambda x: x[0])

    best_angle = smallest[1]


    #################################################################################### Correction uneven Lighting

    full_img_45_blurred = ndimage.rotate(image_blurred, best_angle, reshape=True)




    y_sum = np.sum(full_img_45_blurred,0)
    x_sum = np.sum(full_img_45_blurred,1)


    valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=50000)
    valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=50000)

    #corr_fac_x = np.poly1d(np.polyfit(valleys_x,x_sum[valleys_x],18))
    #corr_fac_y = np.poly1d(np.polyfit(valleys_y,y_sum[valleys_y],18))

    corr_fac_x = splrep(valleys_x,x_sum[valleys_x], s=int(len(valleys_x)/2))
    corr_fac_y = splrep(valleys_y,y_sum[valleys_y], s=int(len(valleys_y)/2))

    xp = np.linspace(0, len(x_sum), len(x_sum))
    yp = np.linspace(0, len(y_sum), len(y_sum))


    #c_x = corr_fac_x(xp)
    #c_y = corr_fac_y(yp)
    c_x = BSpline(*corr_fac_x)(xp)
    c_y = BSpline(*corr_fac_y)(yp)

    x_sum_corr = x_sum-c_x
    y_sum_corr = y_sum-c_y

    #Show Curve
    #plt.plot(c_x)
    #plt.plot(x_sum)
    #plt.plot(valleys_x, x_sum[valleys_x], "x")
    #plt.show()

    #Show Corrected
    #plt.plot(c_x)
    #plt.plot(x_sum_corr)
    #plt.plot(valleys_x, x_sum[valleys_x]-x_sum_corr[valleys_x], "x")
    #plt.show()

    ####################################################################################

    full_img_45_blurred = ndimage.rotate(image_blurred, best_angle, reshape=True)
    full_img_45 = ndimage.rotate(image, best_angle, reshape=True)*0


    image_s = full_img_45_blurred.shape


    y_sum = y_sum_corr
    x_sum = x_sum_corr


    valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=20000)
    valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=20000)

    valleys_x = valleys_x[1:-1]
    valleys_y = valleys_y[1:-1]


    ##### X Abschnitte

    a = np.extract((np.logical_and(valleys_x > 0, valleys_x <= 75)), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 7)

    a = np.extract((np.logical_and(valleys_x > 75, valleys_x <= 148)), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 6)

    a = np.extract((np.logical_and(valleys_x > 148, valleys_x <= 210)), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 4)

    a = np.extract((np.logical_and(valleys_x > 210, valleys_x <= 240)), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 2)

    a = np.extract(np.logical_and(valleys_x > 240, valleys_x <= 808), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 1)

    a = np.extract(np.logical_and(valleys_x > 808, valleys_x <= 840), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 2)

    a = np.extract(np.logical_and(valleys_x > 840, valleys_x <= 870), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 4)

    a = np.extract(np.logical_and(valleys_x > 870, valleys_x <= 960), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 6)

    a = np.extract(np.logical_and(valleys_x > 960, valleys_x <= 1050), valleys_x)
    draw_line_topl_br(a, full_img_45, image_s, 7)

    ##### YAbschnitte



    a = np.extract((np.logical_and(valleys_y > 0, valleys_y <= 75+20)), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 7)

    a = np.extract((np.logical_and(valleys_y > 75, valleys_y <= 148)), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 6)

    a = np.extract((np.logical_and(valleys_y > 148, valleys_y <= 178)), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 4)

    a = np.extract((np.logical_and(valleys_y > 178, valleys_y <= 195)), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 2)

    a = np.extract(np.logical_and(valleys_y > 195, valleys_y <= 808), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 1)

    a = np.extract(np.logical_and(valleys_y > 808, valleys_y <= 840), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 2)

    a = np.extract(np.logical_and(valleys_y > 840, valleys_y <= 870), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 4)

    a = np.extract(np.logical_and(valleys_y > 870, valleys_y <= 960), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 6)

    a = np.extract(np.logical_and(valleys_y > 960, valleys_y <= 1050), valleys_y)
    draw_line_bl_topr(a, full_img_45, image_s, 7)



    #plt.imshow(full_img_45)
    #plt.show()

    full_img_45_done = ndimage.rotate(full_img_45, -best_angle, reshape=True)[580:874+1,148:1306+2]#[590+10:904+1-10,158+10:1338-10]
    #full_img_45_raw_done = ndimage.rotate(full_img_45_raw, -best_angle, reshape=True)#[590:904+1-20,158:1338-20]

    #Oben links (158,590) Unten Rechts (1336,904)

    ret, full_img_45_done_thresh = cv2.threshold(full_img_45_done, 127, 4095, cv2.THRESH_BINARY)

    full_img_45_done_thresh = cv2.convertScaleAbs(full_img_45_done_thresh)

    #plt.imshow(image, cmap="gray")
    #plt.imshow(full_img_45_done_thresh, cmap=colors, alpha=0.5 * (full_img_45_done_thresh > 0))
    #plt.show()

    #plt.imshow(full_img_45_done)
    #plt.plot(x_sum)
    #plt.plot(valleys_x, x_sum[valleys_x], "x")
    #plt.show()




    return full_img_45_done_thresh

"""

    plt.plot(x_sum)
    plt.plot(valleys_x, x_sum[valleys_x], "x")
    plt.hlines(*results_half[1:], color="C2")
    plt.show()

    #single_pixel
    a = np.extract(np.logical_and(valleys_x>275,valleys_x < 788) ,valleys_x)
    draw_line_topl_br(a,full_img_45,image_s,1)

"""




def remove_small_regions(image,thresh):
    # do connected components processing
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 4, cv2.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= thresh:  # keep
            result[labels == i + 1] = 255

    return result


def analyze_grid(filename,main_folder):




    img = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    img_low_bit = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)

    img = crop_img(img,img_low_bit)

    #img = calibrate_image(img)



    kernel_size = 13
    blur_gray = cv2.GaussianBlur(img , (kernel_size, kernel_size), 0)

    img = img[10:-10,10:-10]
    blur_gray = blur_gray[10:-10,10:-10]
    line_mask = scan_line(blur_gray,img)



    analyze_silver(line_mask, img)

    return None
    """

    line_mask= line_mask





    maske_arr = np.array(line_mask, dtype=bool)
    mx_only_squares = np.ma.masked_array(img, mask=maske_arr)
    data = mx_only_squares[mx_only_squares.mask == False]

    std_dev = np.std(data)
    av = np.average(data)

    #plt.imshow(img, cmap='gray')  # I would add interpolation='none'
    #plt.imshow(line_mask, cmap=colors, alpha=0.5 * (line_mask > 0))  # interpolation='none'
    #plt.show()



    maske_arr = np.invert(np.array(line_mask, dtype=bool))
    mx = np.ma.masked_array(img, mask=maske_arr)
    data = mx[mx.mask == False]

    res = np.count_nonzero(data > av-0.5*std_dev)

    maske_arr = np.array(line_mask, dtype=bool)
    raw = np.multiply(maske_arr, img)

    heat = np.array( (raw > av-0.5*std_dev)*255, dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(heat, kernel, iterations=1)
    img_dilation_border = cv2.dilate(heat, kernel, iterations=3)

    img_dilation = np.subtract(img_dilation_border,img_dilation)

    print (res)

    #plt.imshow(img, cmap='gray')  # I would add interpolation='none'
    #plt.imshow(img_dilation, cmap=colors, alpha=0.5 * (img_dilation > 0))  # interpolation='none'
    #plt.show()
   

    if(res > 10): #580 filters roughly 5% of the dataset
        plt.imshow(img, cmap='gray')  # I would add interpolation='none'
        plt.imshow(img_dilation, cmap=colors, alpha=0.5 * (img_dilation > 0))  # interpolation='none'
        plt.show()

    return res
"""

def analyze_silver(mask,img):
    img = img[10:-10, 10:-10]
    mask = mask[10:-10, 10:-10]

    mask =  np.array(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_dil = cv2.dilate(mask, kernel, iterations=1)
    mask_dil = np.invert(mask_dil)/255

    mask_h, mask_w = mask_dil.shape

    #Top Left
    triangle_cnt = np.array([(0,0), (100,0), (0,100)])
    cv2.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

    # Top Right
    triangle_cnt = np.array([(mask_w, 0), (mask_w-100, 0), (mask_w, 100)])
    cv2.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

    # Bot. Left
    triangle_cnt = np.array([(0, mask_h), (100, mask_h), (0, mask_h - 100)])
    cv2.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

    # Bot. Right
    triangle_cnt = np.array([(mask_w, mask_h), (mask_w - 100, mask_h), (mask_w,mask_h- 100)])
    cv2.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)



    img_masked = np.multiply(mask_dil,img)

    plt.imshow(img, cmap="gray")
    plt.imshow(mask_dil, cmap=colors, alpha=0.2 * (mask_dil > 0))
    plt.show()

    mx_only_squares = np.ma.masked_array(img, mask=mask)
    data = mx_only_squares[mx_only_squares.mask == False]

    std_dev = np.std(data)
    av = np.average(data)