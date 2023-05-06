import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

Mat = np.ndarray[int, np.dtype[np.generic]]


class Image:

    def __init__(self, filename: str):
        self.filename = filename
        self.image_8_bit = self.load_image(is_8bit=True)
        self.image_12_bit = self.load_image()
        self.crop_image_8_bit,top_left = self.crop(self.image_8_bit)
        self.crop_image_12_bit = self.crop_grid(self.image_12_bit,self.crop_image_8_bit,top_left)
        
        #blackspot
        res = self.blackspot_detect()
        self.blackspot = {
            "right": res[0],
            "left" : res[1]
        }

    def load_image(self, is_8bit=False) -> Mat:
        """
                load image either 8 or 12 bit
        """
        if not os.path.isfile(self.filename):
            raise NameError
        
        if is_8bit:
            img = cv.imread(self.filename, cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(
                self.filename,  cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
        return img

    def crop(self, image: Mat) -> Mat:
        """
                crop to the whole blade
        """
        template = cv.imread('asset/cropped_manual.tif', cv.IMREAD_GRAYSCALE)
        res = cv.matchTemplate(image, template, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = min_loc
        cropped_image = image[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]
        return cropped_image, top_left

    def crop_grid(self, image_12_bit: Mat, cropped_image_8bit: Mat, top_left) -> Mat:
        """
                Crop Grid from the image
        """
        cropped_image = image_12_bit[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]
        template_grid = cv.imread('asset/cropped_grid_area.tif', cv.IMREAD_GRAYSCALE)
        bound = cv.matchTemplate(cropped_image_8bit, template_grid, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(bound)
        top_left = min_loc
        top_left = (top_left[0] + 10, top_left[1] + 10)
        bottom_right = (top_left[0] + 1160, top_left[1] + 280)

        cropped_image = cropped_image[65:-65,78:-80]

        return cropped_image

    def show_image(image: Mat) -> None:
        """
                show Image
        """
        cv.imshow("image", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def remove_small_regions(self, image, thresh: int):
        # todo
        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(
            image, None, None, None, 4, cv.CV_32S)
        areas = stats[1:, cv.CC_STAT_AREA]

        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= thresh:  # keep
                result[labels == i + 1] = 255

        return result

    def detect_side_lr(self, isLeft: bool):
        # todo
        mask = None
        if isLeft:
            mask = cv.imread("asset/left_template.png", cv.IMREAD_UNCHANGED)
        else:
            mask = cv.imread("asset/right_template.png", cv.IMREAD_UNCHANGED)


        pic = self.crop_image_8_bit[:, -68:-5]
        if isLeft:
            pic = self.crop_image_8_bit[:, 5:68]


        # blurs template to remove small spots
        pic = cv.GaussianBlur(pic, (3, 3), 0)
        # make picture brighter
        pic = cv.convertScaleAbs(pic, 0.5, 10)
        pic = cv.cvtColor(pic, cv.COLOR_GRAY2RGBA)
        mask = mask[:445, :63]
        # remove blank space with mask
        pic = cv.subtract(pic, mask)

        arr = np.where(pic==0, np.nan, pic)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)

        # check for pixels above the standard deviation
        anomalies = (np.abs(arr - mean) / std >= 2.0).any(axis=2)
        mask_u8 = anomalies.astype(np.uint8) * 255
        mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
        if isLeft:
            mask = cv.imread("asset/left_template_nt.png", cv.IMREAD_COLOR)
        else:
            mask = cv.imread("asset/right_template_nt.png", cv.IMREAD_COLOR)
        mask = mask[:445, :63]
        mask = cv.bitwise_not(mask)
        #remove blankspaces again
        mask_u8 = cv.bitwise_and(mask_u8, mask)
        return mask

    def check_area(self,img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0:
            return False

        cv.drawContours(img, contours, -1, (0, 255, 0), 2)
        area = np.max(np.array([cv.contourArea(x) for x in contours]))

        if area > 1.0:
            return img
        else:
            return np.empty_like(img)

    def defect_valid(self,img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0:
            return False
        area = np.max(np.array([cv.contourArea(x) for x in contours]))

        if area > 1.0:
            return True
        else:
            return False

    def blackspot_detect(self):
        """
        Blackspot detection"""
        right_side = self.detect_side_lr(False)
        left_side = self.detect_side_lr(True)
        
        if self.defect_valid(right_side):
            right_side = self.check_area(right_side)
            
        if self.defect_valid(left_side):
            left_side = self.check_area(left_side)
        return (right_side,left_side)
            
        

if __name__ == "__main__":
    image_directory = ""
    output_directory = ""
    test = Image('test.tif')
    right, left = test.blackspot_detect()
    cv.imshow("Left",left)
    cv.waitKey(0)
    cv.imshow("Right",right)
    cv.waitKey(0)
    cv.destroyAllWindows()
    