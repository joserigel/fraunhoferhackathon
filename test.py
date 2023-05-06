import cv2 

img = cv2.imread('main\\test\good.tif',cv2.IMREAD_GRAYSCALE)
img = cv2.convertScaleAbs(img, 0.5, 10)
cv2.imwrite('image.png',img)