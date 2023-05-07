import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import ndimage,datasets
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import splrep, BSpline
import time
Mat = np.ndarray[int, np.dtype[np.generic]]


class Image:
    def __init__(self, filename: str,grid_threshold=580):
        self.cwd = os.getcwd()
        self.filename = filename
        self.image_8_bit = self.load_image(is_8bit=True)
        self.image_12_bit = self.load_image()
        self.crop_image_8_bit, top_left = self.crop(self.image_8_bit)
        self.crop_image_12_bit = self.crop_grid(top_left)
        self.grid_threshold = grid_threshold
        self.grid_value = 0

        time0 = time.time()
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
        
        mask,img = self.grid_detect()
        self.grid_silver = self.grid_silver(mask,img)
        self.grid_black = self.grid_black(mask,img)
        
        self.processed_image = self.combine_image()
        print(time.time()-time0)

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
        cv.imwrite('before.png',pic)
        pic = cv.convertScaleAbs(pic, 0.5, 10)
        cv.imwrite('after.png',pic)
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
        
    def draw_line_topl_br(self,a,img,shape,w=1):
        for i in a:
            cv.line(img, (0, i), (shape[0], i), 255, w)

    def draw_line_bl_topr(self,a,img,shape,w=1):
        for i in a:
            cv.line(img,  ( i,shape[1]), (i,0) , 255, w)
    
    def scan_line2(self,blurred,image):
        res = []
        
        tolerance = 1
        base = 45
        tries = 11
        
        for i in np.linspace(base - tolerance, base+tolerance,tries):
            full_img_45_blurred = ndimage.rotate(blurred, i, reshape=True)
            
            y_sum = np.sum(full_img_45_blurred,0)
            x_sum = np.sum(full_img_45_blurred,1)
            
            valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=50000)
            valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=50000)

            widthsx, h_eval, left_ips, right_ips = peak_widths(-x_sum, valleys_x, rel_height=0.5)
            widthsy, h_eval, left_ips, right_ips = peak_widths(-y_sum, valleys_y, rel_height=0.5)
            res.append((np.sum(np.concatenate((widthsx,widthsy), axis=None)),i))
        smallest = min(res, key=lambda x: x[0])
        best_angle = smallest[1]
        ##############################################
        full_img_45_blurred = ndimage.rotate(blurred, best_angle, reshape=True)
        y_sum = np.sum(full_img_45_blurred,0)
        x_sum = np.sum(full_img_45_blurred,1)
        valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=50000)
        valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=50000)
        corr_fac_x = splrep(valleys_x,x_sum[valleys_x], s=int(len(valleys_x)/2))
        corr_fac_y = splrep(valleys_y,y_sum[valleys_y], s=int(len(valleys_y)/2))
        xp = np.linspace(0, len(x_sum), len(x_sum))
        yp = np.linspace(0, len(y_sum), len(y_sum))
        c_x = BSpline(*corr_fac_x)(xp)
        c_y = BSpline(*corr_fac_y)(yp)
        x_sum_corr = x_sum-c_x
        y_sum_corr = y_sum-c_y
        ########################################################
        full_img_45_blurred = ndimage.rotate(blurred, best_angle, reshape=True)
        full_img_45 = ndimage.rotate(image, best_angle, reshape=True)*0
        image_s = full_img_45_blurred.shape
        y_sum = y_sum_corr
        x_sum = x_sum_corr
        valleys_x, _ = find_peaks(-x_sum, distance=10, prominence=20000)
        valleys_y, _ = find_peaks(-y_sum, distance=10, prominence=20000)
        valleys_x = valleys_x[1:-1]
        valleys_y = valleys_y[1:-1]
        a = np.extract((np.logical_and(valleys_x > 0, valleys_x <= 75)), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 7)
        a = np.extract((np.logical_and(valleys_x > 75, valleys_x <= 148)), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 6)
        a = np.extract((np.logical_and(valleys_x > 148, valleys_x <= 210)), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 4)
        a = np.extract((np.logical_and(valleys_x > 210, valleys_x <= 240)), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 2)
        a = np.extract(np.logical_and(valleys_x > 240, valleys_x <= 808), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 1)
        a = np.extract(np.logical_and(valleys_x > 808, valleys_x <= 840), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 2)
        a = np.extract(np.logical_and(valleys_x > 840, valleys_x <= 870), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 4)
        a = np.extract(np.logical_and(valleys_x > 870, valleys_x <= 960), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 6)
        a = np.extract(np.logical_and(valleys_x > 960, valleys_x <= 1050), valleys_x)
        self.draw_line_topl_br(a, full_img_45, image_s, 7)

        ##### YAbschnitte

        a = np.extract((np.logical_and(valleys_y > 0, valleys_y <= 75+20)), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 7)
        a = np.extract((np.logical_and(valleys_y > 75, valleys_y <= 148)), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 6)
        a = np.extract((np.logical_and(valleys_y > 148, valleys_y <= 178)), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 4)
        a = np.extract((np.logical_and(valleys_y > 178, valleys_y <= 195)), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 2)
        a = np.extract(np.logical_and(valleys_y > 195, valleys_y <= 808), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 1)
        a = np.extract(np.logical_and(valleys_y > 808, valleys_y <= 840), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 2)
        a = np.extract(np.logical_and(valleys_y > 840, valleys_y <= 870), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 4)
        a = np.extract(np.logical_and(valleys_y > 870, valleys_y <= 960), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 6)
        a = np.extract(np.logical_and(valleys_y > 960, valleys_y <= 1050), valleys_y)
        self.draw_line_bl_topr(a, full_img_45, image_s, 7)
        full_img_45_done = ndimage.rotate(full_img_45, -best_angle, reshape=True)[580:874+1,148:1306+2]
        ret, full_img_45_done_thresh = cv.threshold(full_img_45_done, 127, 4095, cv.THRESH_BINARY)
        full_img_45_done_thresh = cv.convertScaleAbs(full_img_45_done_thresh)
        return full_img_45_done_thresh
    
    def grid_silver(self,mask,img):
        
        mask =  np.array(mask, dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask_dil = cv.dilate(mask, kernel, iterations=1)
        mask_dil = np.invert(mask_dil)/255
        
        mask_h, mask_w = mask_dil.shape
         
         #Top Left
        triangle_cnt = np.array([(0,0), (100,0), (0,100)])
        cv.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

        # Top Right
        triangle_cnt = np.array([(mask_w, 0), (mask_w-100, 0), (mask_w, 100)])
        cv.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

        # Bot. Left
        triangle_cnt = np.array([(0, mask_h), (100, mask_h), (0, mask_h - 100)])
        cv.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

        # Bot. Right
        triangle_cnt = np.array([(mask_w, mask_h), (mask_w - 100, mask_h), (mask_w,mask_h- 100)])
        cv.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)
        
        img_masked = np.multiply(mask_dil,img)
        cv.imwrite("img_mask.png",img_masked)
        
        return mask_dil
        
        
    def grid_detect(self):
        img = self.crop_image_12_bit
        
        kernel_size = 13
        blur_gray = cv.GaussianBlur(img , (kernel_size, kernel_size), 0)
        img = img[10:-10,10:-10]
        blur_gray = blur_gray[10:-10,10:-10]
        line_mask = self.scan_line2(blur_gray,img)
        cv.imwrite('mask.png',line_mask)

        return line_mask, img
    
    
    def grid_black(self,mask_dil,img):
        mask_h, mask_w = mask_dil.shape
        #Top Left
        triangle_cnt = np.array([(0,0), (125,0), (0,125)])
        cv.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

        # Top Right
        triangle_cnt = np.array([(mask_w, 0), (mask_w-125, 0), (mask_w, 125)])
        cv.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

        # Bot. Left
        triangle_cnt = np.array([(0, mask_h), (125, mask_h), (0, mask_h - 125)])
        cv.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)

        # Bot. Right
        triangle_cnt = np.array([(mask_w, mask_h), (mask_w - 125, mask_h), (mask_w,mask_h- 125)])
        cv.drawContours(mask_dil, [triangle_cnt], 0, 0, -1)
        
        maske_arr = np.array(mask_dil, dtype=bool)
        mx_only_squares = np.ma.masked_array(img, mask=maske_arr)
        data = mx_only_squares[mx_only_squares.mask == False]
        
        std_dev = np.std(data)
        av = np.average(data)

        maske_arr = np.invert(np.array(mask_dil, dtype=bool))
        mx = np.ma.masked_array(img, mask=maske_arr)
        data = mx[mx.mask == False]

        res = np.count_nonzero(data > av-0.5*std_dev)

        maske_arr = np.array(mask_dil, dtype=bool)
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
        dim_grid = self.get_dim(self.grid_black)
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
            dim_left[1]+offset[1]:dim_left[1]+offset[1]+dim_grid[1]] = self.grid_black
        
        vis[dim_top[0]+offset[0]+dim_grid[0]+offset[0]:dim_top[0]+offset[0]+dim_grid[0]+offset[0]+dim_bottom[0],
            dim_left[1]:dim_left[1]+dim_bottom[1]] = self.teeth['bottom']

        vis[:dim_right[0], dim_left[1]+offset[1]+dim_grid[1]+offset[1]:] = self.blackspot['right']

        return vis


if __name__ == "__main__":

    image_directory = ""
    output_directory = ""
    test = Image('test.tif')
    print(test.get_dim(test.grid_silver))
    cv.imshow("",test.grid_silver)
    cv.imshow("s",test.grid_black)
    print(test.get_dim(test.grid_black))
    res = np.add(test.grid_black,test.grid_silver)
    cv.imshow('res',res)
    cv.waitKey(0)
    cv.destroyAllWindow()