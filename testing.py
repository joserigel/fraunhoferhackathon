import glob
from teeth import detect_side
import cv2 as cv

input_folder_path = './PrePro/input'
output_folder_path = './PrePro/output'

i = 0
for filename in glob.glob(input_folder_path + '/*.tif'):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    img = detect_side(False, img)
    
    
    cv.imwrite(output_folder_path + "/" + str(i) + ".png", img)
    cv.imshow("test", img)
    cv.waitKey(10)
    i += 1
    