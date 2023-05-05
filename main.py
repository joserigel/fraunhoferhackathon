import cv2
import numpy as np
import os




# Function to calculate angle between 3 points
def angle_cos(p0, p1, p2):
    # We are taking vectors from p1 to p0 and from p1 to p2
    # and then we are normalizing them by their lengths.
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    # dot product of these vectors gives us cosine of the angle between vectors.
    # We return absolute value to ignore sign, as angle direction does not matter
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    # Apply Gaussian blur to the image to reduce noise and detail
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    # Split image into RGB channels
    for gray in cv2.split(img):
        # Apply different threshold levels
        for thrs in range(0, 255, 26):
            # Use Canny edge detection for threshold 0 (special case)
            if thrs == 0:
                # Detect edges in the image
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                # Expand the edges by dilating them
                bin = cv2.dilate(bin, None)
            else:
                # Apply thresholding for non-zero thresholds
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            # Find contours in the binary image
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                # Approximate the contour to a polygon
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                # If the polygon has 4 vertices, has a significant area, and is convex, then it might be a square
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    # Reshape the contour array
                    cnt = cnt.reshape(-1, 2)
                    # Calculate maximum cosine of the angle between joint edges to check if this is a rectangle (cosine should be ~0)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])
                    # If cosines of all angles is small (all angles are ~90 degree) then write this contour as a square
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def process_images_in_folder(input_folder_path, output_folder_path):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder_path):
        # Check if the file is an image
        if filename.endswith('.tif'):
            # Load the image
            img_path = os.path.join(input_folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)

            # Find squares in the image
            squares = find_squares(img)

            # Draw all detected squares on the original image
            cv2.drawContours(img, squares, -1, (0, 255, 0), 3)

            # Save the image with detected squares to the output folder
            output_path = os.path.join(output_folder_path, filename)
            cv2.imwrite(output_path, img)


img = cv2.imread('your_image.png', cv2.IMREAD_ANYDEPTH)
squares = find_squares(img)

# Draw all detected squares on the original image
cv2.drawContours(img, squares, -1, (0, 255, 0), 3)
# Display the image with squares
cv2.imshow('squares', img)
# Wait for user to press any key (this is necessary to keep the window open)
cv2.waitKey(0)
# Destroy all windows
cv2.destroyAllWindows()
