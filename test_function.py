from cropIn import crop
import cv2 as cv

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
  
img = cv.imread('F:\hackathon\silver_defect\cam1_0013190123043946301.tif',cv.IMREAD_GRAYSCALE)
img = crop_img(img)
cv.imshow("",img)
cv.waitKey(0)
cv.destroyAllWindows()