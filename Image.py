import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import ndimage,datasets
from scipy.signal import find_peaks

Mat = np.ndarray[int, np.dtype[np.generic]]


class Image:
    def __init__(self, filename: str,grid_threshold=80000000):
        self.cwd = os.getcwd()
        self.filename = filename
        self.image_8_bit = self.load_image(is_8bit=True)
        self.image_12_bit = self.load_image()
        self.crop_image_8_bit, top_left = self.crop(self.image_8_bit)
        self.crop_image_12_bit = self.crop_grid(top_left)

        # blackspot
        blackspot = self.blackspot_detect()
        self.blackspot = {
            "right": np.sum(blackspot[0],axis=2),
            "left": np.sum(blackspot[1],axis=2)
        }

        # teeth
        teeth = self.teeth_detect()
        self.teeth = {
            "top": np.sum(teeth[0],axis=2),
            "bottom": np.sum(teeth[1],axis=2)
        }
        
        
        
        
        
        self.grid = self.grid_detect()
        
        self.processed_image = self.combine_image()

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
        template = cv.imread(os.path.join(self.cwd, 'asset\\cropped_manual.tif'), cv.IMREAD_GRAYSCALE)
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
        cropped_image = self.image_12_bit[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]
        template_grid = cv.imread(os.path.join(
            self.cwd, 'asset/cropped_grid_area.tif'), cv.IMREAD_GRAYSCALE)
        bound = cv.matchTemplate(
            self.crop_image_8_bit, template_grid, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(bound)
        top_left = min_loc
        top_left = (top_left[0] + 10, top_left[1] + 10)

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
            mask = cv.imread("asset/top_template.png", cv.IMREAD_UNCHANGED)
        else:
            mask = cv.imread("asset/top_template.png", cv.IMREAD_UNCHANGED)

        pic = self.crop_image_8_bit[:32, 78:-77]

        if isBottom:
            pic = self.crop_image_8_bit[-32:, 78:-77]

        # blurs template to remove small spots
        pic = cv.GaussianBlur(pic, (3, 3), 0)
        # make picture brighter
        pic = cv.convertScaleAbs(pic, 0.5, 10)
        pic = cv.cvtColor(pic, cv.COLOR_GRAY2RGBA)
        mask = mask[:32,:1183]
        
        pic = cv.subtract(pic, mask)

        arr = np.where(pic == 0, np.nan, pic)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)

        anomalies = (np.abs(arr - mean) / std >= 2.0).any(axis=2)
        mask_u8 = anomalies.astype(np.uint8) * 255
        mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_GRAY2RGB)
        if isBottom:
            mask = cv.imread("asset/top_template_nt.png", cv.IMREAD_COLOR)
        else:
            mask = cv.imread("asset/top_template_nt.png", cv.IMREAD_COLOR)
        mask = mask[:32,:1183]
        mask = cv.bitwise_not(mask)
        #remove blankspaces again
        mask_u8 = cv.bitwise_and(mask_u8, mask)

        return mask_u8

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
        
    def scan_line(self,blured,image):
        full_img_45_blurred = ndimage.rotate(blured, 45, reshape=True)
        full_img_45 = ndimage.rotate(image, 45, reshape=True)*0
        
        image_size = self.get_dim(full_img_45_blurred)
        
        y_sum = np.sum(full_img_45_blurred,0)
        x_sum = np.sum(full_img_45_blurred,1)
        valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=50000)
        valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=50000)
        for i in valleys_x:
            cv.line(full_img_45, (0, i), (image_size[0],i) , 255, 1)


        for i in valleys_y:
            cv.line(full_img_45, ( i,0), (i,image_size[0]) , 255, 1)
            
        full_img_45_done = ndimage.rotate(full_img_45, -45, reshape=True)[590:904+1,158:1338]
        _, full_img_45_done_thresh = cv.threshold(full_img_45_done, 127, 4095, cv.THRESH_BINARY)
        full_img_45_done_thresh = cv.convertScaleAbs(full_img_45_done_thresh)
        return full_img_45_done_thresh
    
    
    def grid_detect(self):
        img = self.crop_image_12_bit
        kernel_size = 11
        blur_gray = cv.GaussianBlur(img , (kernel_size, kernel_size), 0)
        
        line_mask = self.scan_line(blur_gray,img)
        
        img = img[40:-40,80:-80]
        line_mask= line_mask[40:-40,80:-80]
        img_median_blurred = img
        
        maske_arr = np.array(line_mask, dtype=bool)
        mx_only_squares = np.ma.masked_array(img_median_blurred, mask=maske_arr)
        data = mx_only_squares[mx_only_squares.mask == False]
        
        std_dev = np.std(data)
        av = np.average(data)
        
        maske_arr = np.invert(np.array(line_mask, dtype=bool))
        mx = np.ma.masked_array(img_median_blurred, mask=maske_arr)
        data = mx[mx.mask == False]
        
        res = np.count_nonzero(data > av-0.5*std_dev)

        maske_arr = np.array(line_mask, dtype=bool)
        raw = np.multiply(maske_arr, img)

        heat = np.array( (raw > av-0.5*std_dev)*255, dtype=np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        img_dilation = cv.dilate(heat, kernel, iterations=1)
        img_dilation_border = cv.dilate(heat, kernel, iterations=3)

        img_dilation = np.subtract(img_dilation_border,img_dilation)

        return img_dilation
    
    @staticmethod
    def remove_small_regions(image,thresh):
    # do connected components processing
        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(image, None, None, None, 4, cv.CV_32S)

        # get CC_STAT_AREA component as stats[label, COLUMN]
        areas = stats[1:, cv.CC_STAT_AREA]

        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= thresh:  # keep
                result[labels == i + 1] = 255

        return result

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

        if self.defect_valid(top_side):
            top_side = self.check_area(top_side)

        if self.defect_valid(bottom_side):
            bottom_side = self.check_area(bottom_side)

        return (top_side, bottom_side)

    @staticmethod
    def get_dim(img):
        return img.shape[:2]

    def combine_image(self):
        dim_left = self.get_dim(self.blackspot['left'])
        dim_right = self.get_dim(self.blackspot['right'])

        dim_top = self.get_dim(self.teeth['top'])
        dim_grid = self.get_dim(self.grid)
        dim_bottom = self.get_dim(self.teeth['bottom'])

        array_size = (
                max(dim_left[0], dim_right[0]),
            dim_left[1]+dim_right[1] +
            max(dim_grid[1], dim_top[1], dim_bottom[1])
        )
        # create zeros array of image size
        offset = ((array_size[0]-(dim_top[0]+dim_top[0]+dim_grid[0]))//2,
                  (array_size[1]-(dim_left[0]+dim_grid[0]+dim_right[0]))//2)
        
        vis = np.zeros((2*dim_top[0]+dim_grid[0]+2*offset[0],dim_right[1]+dim_left[1]+dim_grid[1]+2*offset[1]), np.uint8)
        

        vis[:dim_left[0], :dim_left[1]] = self.blackspot['left']

        vis[:dim_top[0], dim_left[1]:dim_left[1]+dim_top[1]] = self.teeth['top']

        vis[dim_top[0]+offset[0]:dim_top[0]+offset[0]+dim_grid[0],
            dim_left[1]+offset[1]:dim_left[1]+offset[1]+dim_grid[1]] = self.grid
        
        vis[dim_top[0]+offset[0]+dim_grid[0]+offset[0]:dim_top[0]+offset[0]+dim_grid[0]+offset[0]+dim_bottom[0],
            dim_left[1]:dim_left[1]+dim_bottom[1]] = self.teeth['bottom']

        vis[:dim_right[0], dim_left[1]+offset[1]+dim_grid[1]+offset[1]:] = self.blackspot['right']

        return vis


if __name__ == "__main__":

    image_directory = ""
    output_directory = ""
    test = Image('test\\teeth2.tif')
    cv.imshow("",test.processed_image)
    cv.waitKey(0)
    cv.destroyAllWindow()
