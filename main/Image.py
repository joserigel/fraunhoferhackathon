import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

Mat = np.ndarray[int, np.dtype[np.generic]]
import glob

class Image:
    def __init__(self, filename: str):
        self.cwd = os.getcwd()
        self.filename = filename
        self.image_8_bit = self.load_image(is_8bit=True)
        self.image_12_bit = self.load_image()
        self.crop_image_8_bit, top_left = self.crop(self.image_8_bit)
        self.crop_image_12_bit = self.crop_grid(top_left)

        # blackspot
        blackspot = self.blackspot_detect()
        self.blackspot = {
            "right": blackspot[0],
            "left": blackspot[1]
        }

        # teeth
        teeth = self.teeth_detect()
        self.teeth = {
            "top": teeth[0],
            "bottom": teeth[1]
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
        template = cv.imread(os.path.join(
            self.cwd, 'asset\\cropped_manual.tif'), cv.IMREAD_GRAYSCALE)
        res = cv.matchTemplate(image, template, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = min_loc
        cropped_image = image[top_left[1]:(
            top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]
        return cropped_image, top_left

    def crop_grid(self, top_left) -> Mat:
        """
                Crop Grid from the image
        """
        cropped_image = self.image_12_bit[top_left[1]:(
            top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]
        template_grid = cv.imread(os.path.join(
            self.cwd, 'asset/cropped_grid_area.tif'), cv.IMREAD_GRAYSCALE)
        bound = cv.matchTemplate(
            self.crop_image_8_bit, template_grid, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(bound)
        top_left = min_loc
        top_left = (top_left[0] + 10, top_left[1] + 10)
        bottom_right = (top_left[0] + 1160, top_left[1] + 280)

        cropped_image = cropped_image[65:-65, 78:-80]

        return cropped_image

    @staticmethod
    def show_image(image: Mat) -> None:
        """
                show Image
        """
        cv.imshow("image", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def remove_small_regions(image, thresh: int):
        # todo
        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(
            image, None, None, None, 4, cv.CV_32S)
        areas = stats[1:, cv.CC_STAT_AREA]

        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= thresh:  # keep
                result[labels == i + 1] = 255

        return result

    def detect_side_tb(self, isBottom: bool):
        mask = None
        if (isBottom):
            mask = cv.imread("asset/top_template_nt.png", cv.IMREAD_UNCHANGED)
        else:
            mask = cv.imread("asset/bottom_template_nt.png", cv.IMREAD_UNCHANGED)

        pic = self.crop_image_8_bit[5:20, 78:-77]

        if isBottom:
            pic = self.crop_image_8_bit[-20:-5, 78:-77]

        # blurs template to remove small spots
        #pic = cv.GaussianBlur(pic, (1, ), 0)
        # make picture brighter
        pic = cv.GaussianBlur(pic, (5, 5), 0)
        pic = cv.convertScaleAbs(pic, 1, 5)
        pic = cv.addWeighted(pic, 3, pic, 0, 10)
        
        
        pic = np.invert(pic)
        pic = cv.cvtColor(pic, cv.COLOR_GRAY2RGBA)
        mask = mask[:15,:1183]
        #mask = cv.GaussianBlur(mask, (5, 5), 0)
        
        pic = cv.subtract(pic, mask)

        arr = np.where(pic == 0, np.nan, pic)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)

        anomalies = (np.abs(arr - mean) / std >= 1.05).any(axis=2)
        mask_u8 = anomalies.astype(np.uint8) * 255
        mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
        if isBottom:
            mask = cv.imread("asset/top_template_nt.png", cv.IMREAD_COLOR)
        else:
            mask = cv.imread("asset/bottom_template_nt.png", cv.IMREAD_COLOR)
        mask = mask[:15,:1183]
        mask = cv.GaussianBlur(mask, (9, 9), 0)
        ret, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)
        mask = cv.bitwise_not(mask)
        #remove blankspaces again
        mask_u8 = cv.bitwise_and(mask_u8, mask)
        
        shapes = np.zeros((15, 1183, 3), np.uint8)
        mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_RGB2GRAY)
        ret, thresh = cv.threshold(mask_u8, 100, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv.contourArea(c) > 32]
        
        cv.drawContours(shapes, contours, -1, color=(255, 255, 255), thickness=cv.FILLED)

        return shapes

    def detect_side_lr(self, isLeft: bool):
        # todo
        mask = None
        if isLeft:
            mask = cv.imread(os.path.join(
                self.cwd, "asset/left_template.png"), cv.IMREAD_UNCHANGED)
        else:
            mask = cv.imread(os.path.join(
                self.cwd, "asset/right_template.png"), cv.IMREAD_UNCHANGED)

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

        arr = np.where(pic == 0, np.nan, pic)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)

        # check for pixels above the standard deviation
        anomalies = (np.abs(arr - mean) / std >= 2.0).any(axis=2)
        mask_u8 = anomalies.astype(np.uint8) * 255
        mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
        if isLeft:
            mask = cv.imread(os.path.join(
                self.cwd, "asset/left_template_nt.png"), cv.IMREAD_COLOR)
        else:
            mask = cv.imread(os.path.join(
                self.cwd, "asset/right_template_nt.png"), cv.IMREAD_COLOR)
        mask = mask[:445, :63]
        mask = cv.bitwise_not(mask)
        # remove blankspaces again
        mask_u8 = cv.bitwise_and(mask_u8, mask)
        return mask_u8

    @staticmethod
    def check_area(img):
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

    @staticmethod
    def defect_valid(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, 0)
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0:
            return False
        area = np.max(np.array([cv.contourArea(x) for x in contours]))

        if area > 1.0:
            return True
        else:
            return False

    def blackspot_detect(self):
        """
            Blackspot detection
        """
        right_side = self.detect_side_lr(False)
        left_side = self.detect_side_lr(True)

        if self.defect_valid(right_side):
            right_side = self.check_area(right_side)

        if self.defect_valid(left_side):
            left_side = self.check_area(left_side)
        return (right_side, left_side)

    def teeth_detect(self):
        top_side = self.detect_side_tb(False)
        bottom_side = self.detect_side_tb(True)

        return (top_side, bottom_side)

        if self.defect_valid(top_side):
            top_side = self.check_area(top_side)

        if self.defect_valid(bottom_side):
            bottom_side = self.check_area(bottom_side)

        return (top_side, bottom_side)

    @staticmethod
    def get_dim(img):
        return img.shape[:2]

    def combine_image(self):
        vis = np.zeros((), np.uint8)
        vertical = cv.vconcat([self.top, self.grid, self.bottom])
        res = cv.hconcat(
            [self.blackspot['right'], vertical, self.blackspot['left']])
        return res

    def combine_image_np(self):
        dim_left = self.get_dim(self.blackspot['left'])
        dim_right = self.get_dim(self.blackspot['right'])

        dim_top = self.get_dim(self.blackspot['right'])
        dim_grid = self.get_dim(self.blackspot['right'])
        dim_bottom = self.get_dim(self.blackspot['right'])

        # create zeros array of image size
        vis = np.zeros((
            max(dim_top[0]+dim_bottom[0]+dim_grid[0],
                max(dim_left[0], dim_right[0])),
            dim_left[1]+dim_right[1] +
            max(dim_grid[1], dim_top[1], dim_bottom[1])
        ), np.uint8)

        vis[:dim_left[0], :dim_left[1]] = self.blackspot['left']

        vis[:dim_top[0], dim_left[1]:dim_left[1]+dim_top[1]] = self.top

        vis[dim_top[0]:dim_top[0]+dim_grid[0],
            dim_left[1]:dim_left[1]+dim_grid[1]] = self.grid
        vis[dim_top[0]+dim_grid[0]:dim_top[0]+dim_grid[0]+dim_bottom[0],
            dim_left[1]:dim_left[1]+dim_bottom[1]] = self.bottom

        vis[:dim_right[0], dim_left[1]+dim_bottom[1]:] = self.blackspot['right']

        return vis


if __name__ == "__main__":

    
    input_folder_path = '../PrePro/teeth_defect'
    output_folder_path = './PrePro/output'

    i = 0
    for filename in glob.glob(input_folder_path + '/*.tif'):
        test = Image(filename)
        cv.imshow("top", test.teeth['top'])
        cv.waitKey(10)
    # image_directory = ""
    # output_directory = ""
    # test = Image('test.tif')
    # cv.imshow("Left", test.blackspot['left'])
    # cv.waitKey(0)
    # cv.imshow("Right", test.blackspot['right'])
    # cv.waitKey(0)
    # cv.imshow("top", test.teeth['top'])
    # cv.waitKey(0)
    # cv.imshow("bottom", test.teeth['bottom'])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
