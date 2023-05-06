from side_smudge import detect_side
from isolate import chk_area, defect_valid
from cropIn import crop
import cv2 as cv
import glob

input_folder_path =  "./PrePro/precropped"
output_folder_path = "./PrePro/output2"

for filename in glob.glob(input_folder_path + '/*.tif'):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    gradient = crop(img)
    gradient = cv.convertScaleAbs(img, -0.5, 16)
    img_cm = cv.applyColorMap(gradient, cv.COLORMAP_WINTER)

    cv.imshow("image",img_cm)
    cv.waitKey(0)
    
    #Right
    right = detect_side(False, img)
    if(defect_valid(right)):
        right = chk_area(right)
        
        cv.imshow("image",right)
        cv.waitKey(0)
        

    #Left
    left = detect_side(True, img)
    if(defect_valid(left)):
        left = chk_area(left)

        cv.imshow("image", left)
        cv.waitKey(0)

    


