import numpy as np


def create_binary_mask(bbox, pixel_coords):
    # Get bounding box parameters
    min_row, min_col, max_row, max_col = bbox
    
    # Calculate the size of the mask without extra rows or columns
    height = max_row - min_row
    width = max_col - min_col
    # Create an empty binary mask with the size of the bounding box
    mask = np.zeros((height, width), dtype=np.bool_)
    
    # Set the corresponding pixels in the mask to 1
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
    

    # Initialize feature dictionary
    features = {}


    i = np.arange(glcm.shape[0])
    j = np.arange(glcm.shape[1])

    # 'ASM'
    if feature_names is None or 'ASM' in feature_names:
        asm = np.sum(glcm ** 2)
        features['ASM'] = asm

    #  'Contrast'
    if feature_names is None or 'Contrast' in feature_names:
        contrast = np.sum((i[:, None] - j) ** 2 * glcm)
        features['Contrast'] = contrast

    #  'Correlation'
    if feature_names is None or 'Correlation' in feature_names:
        mean_i = np.sum(i[:, None] * glcm)
        mean_j = np.sum(j * glcm)
        std_i = np.sqrt(np.sum((i[:, None] - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))
        correlation = np.sum((i[:, None] - mean_i) * (j - mean_j) * glcm) / (std_i * std_j)
        features['Correlation'] = correlation

    #  'Variance'
    if feature_names is None or 'Variance' in feature_names:
        variance = np.sum((i[:, None] - mean_i) ** 2 * glcm)
        features['Variance'] = variance

    #  'IDM'
    if feature_names is None or 'IDM' in feature_names:
        idm = np.sum(1 / (1 + (i[:, None] - j) ** 2) * glcm)
        features['IDM'] = idm

    #  'Sum Average'
    if feature_names is None or 'Sum Average' in feature_names:
        sum_avg = np.sum((i[:, None] + j) * glcm)
        features['Sum Average'] = sum_avg

    #  'Sum Variance'
    if feature_names is None or 'Sum Variance' in feature_names:
        sum_variance = np.sum(((i[:, None] + j) - sum_avg) ** 2 * glcm)
        features['Sum Variance'] = sum_variance

    #  'Sum Entropy'
    if feature_names is None or 'Sum Entropy' in feature_names:
        sum_entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Sum Entropy'] = sum_entropy

    #  'Entropy'
    if feature_names is None or 'Entropy' in feature_names:
        entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Entropy'] = entropy

    #  'Difference Variance'
    if feature_names is None or 'Difference Variance' in feature_names:
        diff_var = np.sum(((i[:, None] - j) ** 2) * glcm)
        features['Difference Variance'] = diff_var

    #  'Difference Entropy'
    if feature_names is None or 'Difference Entropy' in feature_names:
        diff_entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        features['Difference Entropy'] = diff_entropy

    #  'IMC1'
    if feature_names is None or 'IMC1' in feature_names:
        imc1 = (entropy - (np.sum(glcm * np.log2(glcm + np.finfo(float).eps)) -
                           np.sum(np.log2(glcm + np.finfo(float).eps) * glcm))) / np.sqrt(np.sum(glcm))
        features['IMC1'] = imc1

    #  'IMC2'
    if feature_names is None or 'IMC2' in feature_names:
        imc2 = (np.sqrt(np.sum(glcm ** 2)) - np.sum(glcm * np.log2(glcm + np.finfo(float).eps))) / np.sqrt(np.sum(glcm))
        features['IMC2'] = imc2

    return features