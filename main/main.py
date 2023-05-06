import os
import cv2
import matplotlib.pyplot as plt
from Image import Image
import time

def get_files(path:str):
  if not os.path.isdir(path):
    raise FileExistsError

  files = []
  for image in os.listdir(path):
    files.append(image)
    
  return files

  
  
def main():
  image_folder = ""
  files = get_files(image_folder)
  for i in files:
    image = Image(i)
    
def main2():
  time0 = time.time()
  image = Image('test.tif')
  print(time.time()-time0)
  

main2()