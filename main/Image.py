import cv2 as cv
import matplotlib.pyplot as plt
import os


class Image:
  
  def __init__(self,filename:str):
    self.filename = filename
    self.image = self.load_image()
    self.crop_image = self.crop()
  
  def load_image(self): 
    img = cv.imread(self.filename, cv.IMREAD_GRAYSCALE)
    return img
  
  def crop(self):
    template = cv.imread('cropped_manual.tif', cv.IMREAD_GRAYSCALE)

    res = cv.matchTemplate(self.image, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc
    cropped_image = self.image[top_left[1]:(top_left[1] + 445), top_left[0]:(top_left[0] + 1338)]
    return cropped_image
  
  def 
    