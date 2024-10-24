# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:21:05 2024

@author: sebas
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:26:08 2024

@author: sebas
"""

import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *
import pickle
import os

from sklearn.decomposition import PCA
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def pca__masked(subimage, mask, n_components=3):
    mask = mask.astype(bool)
    masked_data = subimage[mask]
    masked_data_flattened = masked_data.reshape(-1, subimage.shape[2])
    
    # Check for NaN values and skip if any are found
    if np.isnan(masked_data_flattened).any():
        print("NaN values found in subimage data. Skipping PCA for this subimage.")
        return None, None  # Return None to indicate skipping

    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(masked_data_flattened)
    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    height, width, _ = subimage.shape
    pca_scores_img = np.zeros((height, width, n_components), dtype=np.float32)
    mask_flat = mask.flatten()

    pca_scores_img_flat = pca_scores_img.reshape(-1, n_components)
    pca_scores_img_flat[mask_flat] = pca_scores

    pca_scores_img = pca_scores_img.reshape(height, width, n_components)

    return pca_scores_img, pca_loadings


def create_binary_mask(bbox, pixel_coords):
    min_row, min_col, max_row, max_col = bbox
    height = max_row - min_row
    width = max_col - min_col
    mask = np.zeros((height, width), dtype=np.bool_)
    
    for row, col in pixel_coords:
        if min_row <= row < max_row and min_col <= col < max_col:
            mask[row - min_row, col - min_col] = 1
    
    return mask


main_data_folder = "D:\\SWIR_germ_2nd_18\\ref_corrected"
kernel_folder = os.path.join(main_data_folder, 'Kernels_objects')

dataset = HsiDataset(main_data_folder, data_ext='ref')
HSIreader = HsiReader(dataset)

angles = [0, 45, 90]
features = ['ASM', 'Contrast', 'Correlation', 'entropy'] 

# Dictionary to store texture properties for each sub-image
sub_image_features = {}

for idx in range(len(dataset)):
    HSIreader.read_image(idx)
    metadata = HSIreader.current_metadata
    wv = HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    base_name = os.path.basename(HSIreader.dataset[idx]['data'])
    base_name = os.path.splitext(base_name)[0]
    
    kernels_file_path = os.path.join(kernel_folder, f'{base_name}.pkl')
    
    if os.path.exists(kernels_file_path):
        with open(kernels_file_path, 'rb') as file:
            kernel_data = pickle.load(file)
    
    sub_image_features[base_name] = {}  # Initialize dictionary for current sub-image

    for kernel_id, kernel_info in kernel_data.items():
        pixel_coords = kernel_info['pixels']
        min_row = np.min(pixel_coords[:, 0])
        max_row = np.max(pixel_coords[:, 0])
        min_col = np.min(pixel_coords[:, 1])
        max_col = np.max(pixel_coords[:, 1])

        if min_row >= max_row or min_col >= max_col:
            print(f"Invalid bounding box for kernel {kernel_id} in image {base_name}. Skipping...")
            continue

        bbox = (min_row, min_col, max_row, max_col)
        subimage = HSIreader.read_subimage(bbox)
        bm = create_binary_mask(bbox, pixel_coords)

        if subimage.size == 0 or bm.size == 0:
            print(f"Empty subimage or mask for kernel {kernel_id} in image {base_name}. Skipping...")
            continue

        avg_spectrum = np.mean(subimage[bm == 1], axis=0)
        pca_scores_img, pca_loadings = pca__masked(subimage, mask=bm, n_components=3)
     
        if pca_scores_img is None or pca_loadings is None:
             print(f"Skipping PCA for kernel {kernel_id} in image {base_name} due to NaN values.")
             continue

        for i in range(pca_scores_img.shape[2]):
            component_img = pca_scores_img[:, :, i]
            sub_image_features[base_name][f'Kernel_{kernel_id}_Component_{i}'] = {}
            
            # Normalize the component image to the range [0, 31]
            normalized_image = ((component_img - component_img.min()) / (component_img.max() - component_img.min())) * 31
            normalized_image = normalized_image.astype(np.uint8)  # Convert to uint8 type
        
            for angle in angles:
                # Convert angle to radians
                angle_rad = np.deg2rad(angle)
                
                # Compute GLCM using the normalized image
                glcm = graycomatrix(
                    normalized_image,
                    distances=[1],
                    angles=[angle_rad],
                    levels=32,
                    symmetric=True,
                    normed=True
                )
        
                # Compute texture features
                features_values = {
                    'ASM': graycoprops(glcm, 'ASM')[0, 0],
                    'Contrast': graycoprops(glcm, 'contrast')[0, 0],
                    'Correlation': graycoprops(glcm, 'correlation')[0, 0],
                    'entropy': -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))  # Manual entropy calculation
                }
                
                sub_image_features[base_name][f'Kernel_{kernel_id}_Component_{i}'][f'Angle_{angle}'] = features_values




#%%
import numpy as np

# Prepare a list of all components and angles to iterate over
components = [f'Component_{i}' for i in range(3)]  # Assuming 3 PCA components
angles = [0, 45, 90]  # Angles as specified in your code

# Initialize a dictionary to store the results for each component and angle
output_blocks = {}

for component in components:
    output_blocks[component] = {}
    
    for angle in angles:
        # Initialize a list to hold feature arrays for the current component and angle
        feature_list = []

        # Iterate over each subimage and extract the corresponding features
        for subimage_name, subimage_data in sub_image_features.items():
            for kernel_name, kernel_data in subimage_data.items():
                if component in kernel_name:
                    # Find the data for the given angle
                    angle_key = f'Angle_{angle}'
                    if angle_key in kernel_data:
                        features = kernel_data[angle_key]
                        feature_array = np.array([features['ASM'], features['Contrast'], features['Correlation'], features['entropy']])
                        feature_list.append(feature_array)

        # Convert the list to a NumPy array and store it in the output dictionary
        output_blocks[component][f'Angle_{angle}'] = np.array(feature_list)

#%%
import os
import numpy as np
import scipy.io as io

# Assuming 'output_blocks' is already created and filled with the desired data

# Define the main folder path
main_data_folder = "D:\\SWIR_germ_2nd_18\\ref_corrected"

# Define filenames with paths for saving
numpy_filename = os.path.join(main_data_folder, 'output_blocks.npy')
matlab_filename = os.path.join(main_data_folder, 'output_blocks.mat')

# Convert the nested dictionary structure to a format suitable for saving
flat_output_blocks = {}
for component, angle_data in output_blocks.items():
    for angle, feature_array in angle_data.items():
        flat_output_blocks[f'{component}_Angle_{angle}'] = feature_array

# Save the data as a numpy file in the main data folder
np.save(numpy_filename, flat_output_blocks)

# Save the data as a MATLAB file in the main data folder
io.savemat(matlab_filename, flat_output_blocks)

print(f"Data has been saved in {main_data_folder} as 'output_blocks.npy' and 'output_blocks.mat'.")

# plot the GLCM using the script below - logarithmic normalisation will be applied

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm

# # Assuming your GLCM is named 'glcm' and has the shape (256, 256, 1, 1)
# glcm_2d = glcm[:, :, 0, 0]  # Extract the 2D matrix for the first distance and angle

# # Plot the GLCM as a heatmap with logarithmic normalization
# plt.figure(figsize=(8, 6))
# plt.imshow(glcm_2d, cmap='jet', norm=LogNorm(vmin=glcm_2d.min() + 1e-10, vmax=glcm_2d.max()), interpolation='nearest')
# plt.colorbar(label='Co-occurrence Frequency (Log Scale)')
# plt.title('Gray-Level Co-occurrence Matrix (GLCM) - Log Scale')
# plt.xlabel('Gray Level')
# plt.ylabel('Gray Level')
# plt.show()

