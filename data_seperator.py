import os
import pandas as pd
from PIL import Image


def preprocess_data(tif_folder, csv_file):
    # Read the CSV file containing the labels
    labels_df = pd.read_csv(csv_file)

    data = []

    # Create a dictionary to store preprocessed data
    data_dict = {"filename": [], "image": [], "label": []}

    # Iterate through the rows in the CSV file
    for index, row in labels_df.iterrows():
        # Get the filename and label from the CSV row
        filename, label = row['Id'], row['Label']

        # Check if the .tif file exists in the specified folder
        tif_file_path = os.path.join(tif_folder, filename)
        if os.path.isfile(tif_file_path):
            # Add the filename, image, and label to the data dictionary
            data_dict = {"filename": filename, "path": tif_file_path, "label": label}
            data.append(data_dict)
        else:
            print(f"File not found: {tif_file_path}")

    # Return the preprocessed data as a Pandas DataFrame
    return data


# Example usage:


tif_folder =  './PrePro/input/'
csv_file = './PrePro/labels_train.csv'
preprocessed_data = preprocess_data(tif_folder, csv_file)


for i in preprocessed_data:
    if i["label"] == "good":


