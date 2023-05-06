import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def get_heat_map(images, destination_path):
    # Load an image using plt.imread() function

    fig_list = []
    # Loop through the images and plot the heatmap on each subplot
    for i, image_path in enumerate(images):
       # Load the image using plt.imread()
        img = plt.imread(image_path)

        # Convert the image to grayscale if necessary
        if img.ndim == 3:
            gray_img = np.mean(img, axis=2)
        elif img.ndim == 2:
            gray_img = img

        # Create a figure with a single subplot and plot the heatmap on it
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        sns.heatmap(gray_img, cmap='YlOrRd', ax=ax)
        ax.set_title('Heatmap')

        # Append the figure to the list
        fig_list.append(fig) 


    if not os.path.exists(destination_path):
        os.makedirs(destination_path) 

    for i,fig in enumerate(fig_list):
        fig.savefig(f'{destination_path}/heatmap_{i}.png')

example = ["cam.tif", "test.tif"]

get_heat_map(example, "heatmaps")
#plt.show()
