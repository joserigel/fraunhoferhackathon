import glob
from side_smudge import detect_side
import cv2 as cv

input_folder_path = './PrePro/input'
output_folder_path = './PrePro/output'

for filename in glob.glob(input_folder_path + '/*.tif'):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    img = detect_side(False, img)
    
    cv.imshow("mask", img)
    cv.waitKey(10)
    