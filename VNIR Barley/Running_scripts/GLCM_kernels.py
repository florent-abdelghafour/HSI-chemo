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

def GLCM(Img, mask, theta, d=1, nb_bins=32):
    theta = np.deg2rad(theta)
    
    if theta == 0:
        dy, dx = 0, d
    elif theta == np.pi / 4:  
        dy, dx = d, d
    elif theta == np.pi / 2:  
        dy, dx = d, 0
    else:
        raise ValueError("theta must be 0, 45, or 90 degrees")
    
    glcm = np.zeros((nb_bins, nb_bins), dtype=np.float32)
    
    height, width = Img.shape

    for y in range(height - dy):
        for x in range(width - dx):
            if mask[y, x] and mask[y + dy, x + dx]:
                pixel_1 = int(Img[y, x])
                pixel_2 = int(Img[y + dy, x + dx])
                pixel_1 = min(pixel_1, nb_bins - 1)
                pixel_2 = min(pixel_2, nb_bins - 1)
                glcm[pixel_1, pixel_2] += 1
                glcm[pixel_2, pixel_1] += 1 

    glcm /= np.sum(glcm)
    return glcm

def haralick(glcm, feature_names=None):
    features = {}
    i = np.arange(glcm.shape[0])
    j = np.arange(glcm.shape[1])

    if feature_names is None or 'ASM' in feature_names:
        asm = np.sum(glcm ** 2)
        features['ASM'] = asm

    if feature_names is None or 'Contrast' in feature_names:
        contrast = np.sum((i[:, None] - j) ** 2 * glcm)
        features['Contrast'] = contrast

    if feature_names is None or 'Correlation' in feature_names:
        mean_i = np.sum(i[:, None] * glcm)
        mean_j = np.sum(j * glcm)
        std_i = np.sqrt(np.sum((i[:, None] - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))
        correlation = np.sum((i[:, None] - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)
        features['Correlation'] = correlation

    if feature_names is None or 'Variance' in feature_names:
        variance = np.sum((i[:, None] - mean_i) ** 2 * glcm)
        features['Variance'] = variance

    if feature_names is None or 'IDM' in feature_names:
        idm = np.sum(1 / (1 + (i[:, None] - j) ** 2) * glcm)
        features['IDM'] = idm

    if feature_names is None or 'Sum Average' in feature_names:
        sum_avg = np.sum((i[:, None] + j) * glcm)
        features['Sum Average'] = sum_avg

    if feature_names is None or 'Sum Variance' in feature_names:
        sum_variance = np.sum(((i[:, None] + j) - sum_avg) ** 2 * glcm)
        features['Sum Variance'] = sum_variance

    if feature_names is None or 'Sum Entropy' in feature_names:
        sum_entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Sum Entropy'] = sum_entropy

    if feature_names is None or 'Entropy' in feature_names:
        entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Entropy'] = entropy

    if feature_names is None or 'Difference Variance' in feature_names:
        diff_var = np.sum(((i[:, None] - j) ** 2) * glcm)
        features['Difference Variance'] = diff_var

    if feature_names is None or 'Difference Entropy' in feature_names:
        diff_entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Difference Entropy'] = diff_entropy

    if feature_names is None or 'IMC1' in feature_names:
        imc1 = (entropy - (np.sum(glcm * np.log2(glcm + np.finfo(float).eps)) -
                           np.sum(np.log2(glcm + np.finfo(float).eps) * glcm))) / np.sqrt(np.sum(glcm))
        features['IMC1'] = imc1

    if feature_names is None or 'IMC2' in feature_names:
        imc2 = (np.sqrt(np.sum(glcm ** 2)) - np.sum(glcm * np.log2(glcm + np.finfo(float).eps))) / np.sqrt(np.sum(glcm))
        features['IMC2'] = imc2

    return features

main_data_folder = "D:\\SWIR_Barley_No_damage_matched"
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
        # Check if the bounding box is valid
        if min_row >= max_row or min_col >= max_col:
            print(f"Invalid bounding box for kernel {kernel_id} in image {base_name}. Skipping...")
            continue
        bbox = (min_row, min_col, max_row, max_col)
        subimage = HSIreader.read_subimage(bbox)
        bm = create_binary_mask(bbox, pixel_coords)
        # Check if subimage or mask is empty, and skip if so
        if subimage.size == 0 or bm.size == 0:
            print(f"Empty subimage or mask for kernel {kernel_id} in image {base_name}. Skipping...")
            continue
        avg_spectrum = np.mean(subimage[bm == 1], axis=0)
        pca_scores_img, pca_loadings = pca__masked(subimage, mask=bm, n_components=3)
     
        # Check if PCA was skipped (returned None)
        if pca_scores_img is None or pca_loadings is None:
             print(f"Skipping PCA for kernel {kernel_id} in image {base_name} due to NaN values.")
             continue  # Skip to the next kernel or subimage

    
        for i in range(pca_scores_img.shape[2]):
            component_img = pca_scores_img[:, :, i]
            sub_image_features[base_name][f'Kernel_{kernel_id}_Component_{i}'] = {}
            
            for angle in angles:
                glcm = GLCM(component_img, bm, d=1, theta=angle, nb_bins=32)
                features_values = haralick(glcm, feature_names=features)
                
                sub_image_features[base_name][f'Kernel_{kernel_id}_Component_{i}'][f'Angle_{angle}'] = features_values

# Now sub_image_features will contain the texture properties of each sub-image
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
                        feature_array = np.array([features['ASM'], features['Contrast'], features['Correlation']])
                        feature_list.append(feature_array)

        # Convert the list to a NumPy array and store it in the output dictionary
        output_blocks[component][f'Angle_{angle}'] = np.array(feature_list)

#%%
import os
import numpy as np
import scipy.io as io

# Assuming 'output_blocks' is already created and filled with the desired data

# Define the main folder path
main_data_folder = "D:\\SWIR_Barley_No_damage_matched"

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

