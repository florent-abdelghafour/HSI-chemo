# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:42:21 2024

@author: sebas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from hsi_utils import *
import spectral.io.envi as envi
import glob

# Define paths and constants
main_data_folder = "D:\\VNIR_sub_imtest"
object_folder = os.path.join(main_data_folder, 'Kernels_objects')
num_images_to_plot = 1  # Adjust this as needed
band_index = 80  # Select a spectral band to display (this is just an example)

# Function to plot a sub-image
def plot_sub_image(hypercube, pixel_coords, title=None):
    # Get the bounding box coordinates
    min_row = np.min(pixel_coords[:, 0])
    max_row = np.max(pixel_coords[:, 0])
    min_col = np.min(pixel_coords[:, 1])
    max_col = np.max(pixel_coords[:, 1])

    # Extract the sub-image
    sub_image = hypercube[min_row:max_row+1, min_col:max_col+1, :]

    # Plot the sub-image (choose a band to plot)
    plt.figure()
    plt.imshow(sub_image[:, :, band_index], cmap='gray')
    plt.title(title if title else 'Sub-Image')
    plt.axis('off')
    plt.show()

# Loop through a few object files and corresponding hyperspectral images
for idx in range(num_images_to_plot):
    # Use a pattern to search for the files with the new naming convention
    pattern = f"*_raw_ref.pkl"  # Adjust the pattern if necessary
    files = glob.glob(os.path.join(object_folder, pattern))

    if len(files) == 0:
        print(f"No files found matching pattern: {pattern}")
        continue

    # Select the first file for the current idx (you may want to modify this for other logic)
    file_path = files[idx] if idx < len(files) else files[-1]

    if os.path.exists(file_path):
        print(f"Reading file: {file_path}")
        with open(file_path, 'rb') as f:
            ordered_object_data = pickle.load(f)
    else:
        print(f"File not found: {file_path}")
        continue

    # Load the corresponding hyperspectral image
    HSIreader = HsiReader(HsiDataset(main_data_folder, data_ext='ref'))
    HSIreader.read_image(idx)  # Load the image corresponding to the same index
    hypercube = HSIreader.get_hsi()

    # Plot a few sub-images from the objects
    object_ids = list(ordered_object_data.keys())  # Get object IDs from the .pkl file

    for obj_id in object_ids:
        obj_data = ordered_object_data[obj_id]
        pixel_coords = obj_data['pixels']
        plot_sub_image(hypercube, pixel_coords, title=f'Object ID: {obj_id} (Image {idx})')

    print(f"Finished plotting for image {idx}")
