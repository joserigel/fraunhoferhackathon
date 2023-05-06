import os
import cv2
import matplotlib.pyplot as plt
import PIL
from Image import Image
import time
import numpy as np
from tqdm import tqdm


def get_files(path: str):
    if not os.path.isdir(path):
        raise FileExistsError

    files = []
    for image in os.listdir(path):
        files.append(os.path.join(path, image))

    return files


"""
detects white makings that represent errors in the printing.  
"""


def detect_error(image):  # the input is assumed to be a np.array image
    colors, counts = np.unique(
        image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)
    colors = [list(color) for color in colors]
    threshold = 230
    count = 0

    for i in colors:
        if i[0] > threshold and i[2] > threshold and i[1] > threshold:
            count += 1
    if count > 0:
        return True  # there is an error
    return False  # there are no errors


def main():
    image_folder = os.path.join(os.getcwd(), "test")
    filename = "data.csv"
    files = get_files(image_folder)
    data = """original,filename,runtime,silver_defect,black_defeck,blackspot,teeth_defect\n
  """
    for index, i in tqdm(enumerate(files)):
        if index % 100 == 0:
            with open(filename, "a") as file:
                file.write(data)
            data = ""
        time0 = time.time()
        image = Image(i)
        time_delta = time.time()-time0
        new_file = os.path.join(os.getcwd(), 'imgs', str(index)+'.png')
        cv2.imwrite(new_file, image.processed_image)
        data += f"{image.filename},{new_file},{time_delta},{str(detect_error(image.processed_image))},{str(image.grid_value > image.grid_threshold)},{str(detect_error(image.blackspot['left']) or detect_error(image.blackspot['right']))},{str(detect_error(image.teeth['top']) or detect_error(image.teeth['bottom']))}\n"

def single_image():
    file_name = 'teeth.tif'
    data = ""
    time0 = time.time()
    image = Image(file_name)
    time_delta = time.time()-time0
    new_file = os.path.join(os.getcwd(), 'teeth.png')
    cv2.imwrite(new_file, image.processed_image)
    data += f"{image.filename},{new_file},{time_delta},{str(detect_error(image.processed_image))},{str(image.grid_value > image.grid_threshold)},{str(detect_error(image.blackspot['left']) and detect_error(image.blackspot['right']))},{str(detect_error(image.teeth['top']) and detect_error(image.teeth['bottom']))}\n"
    print(data)


single_image()
