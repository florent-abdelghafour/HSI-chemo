import os
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
import pickle


main_data_folder = "D:\\VNIR_barley"     

# Initialize the HSI dataset and define file extension: contains all paths of hdr and data files
dataset =HsiDataset(main_data_folder,data_ext='hyspex')
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Define the path to save the corrected hyperspectral images
save_folder = os.path.join(main_data_folder, 'ref_corrected')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize the HSI reader: class containg info about the image and allow to load and operate on the hsi
HSIreader = HsiReader(dataset)


nb_pca_comp =3
pca = PCA(n_components=nb_pca_comp)
slice_step = 1000
min_kernel_size=1000
padding=100

# Loop through each hyperspectral image in the dataset
for idx in range(len(dataset)):
  if idx==0:  
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hsi=HSIreader.current_image
    n_rows, n_cols, n_channels =hsi.shape
    n_samples = 50000
    x_idx = np.random.randint(0, n_cols, size=n_samples)
    y_idx = np.random.randint(0, n_rows, size=n_samples)
    spectral_samples = np.zeros((n_samples, n_channels), dtype=hsi.dtype)
    coords = list(zip(y_idx, x_idx))
    spectral_samples = np.array(HSIreader.extract_pixels(coords))
   
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    
    pca_scores_imgs = []
    for start_row in range(0, n_rows, slice_step):
       
        end_row = min(start_row + slice_step, n_rows)
        subcube = hsi[start_row:end_row, :, :]
        
        subcube_flat =subcube.reshape(-1, n_channels)
        pca_scores = np.dot(subcube_flat, pca_loadings)
        pca_scores_img = pca_scores.reshape(end_row - start_row, n_cols, nb_pca_comp)
        pca_scores_imgs.append(pca_scores_img)
    final_pca_scores_img = np.concatenate(pca_scores_imgs, axis=0)
     
    score_pc_ref = final_pca_scores_img[:, :, 1]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=2)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    labeled_image = label(segmented)
    
    binary_image = labeled_image > 0
    
    filled_binary_image = binary_fill_holes(binary_image)
    
    labeled_image = label(filled_binary_image)
    labeled_image = label(remove_small_objects(labeled_image > 0, min_size=500))
    
    color_image = color_labels(labeled_image)
    
    plt.figure()
    plt.imshow(color_image)
    plt.show()
    
    
    regions = regionprops(labeled_image)
    print(f"Number of regions found: {len(regions)}")
    object_data = []
    
    for region in regions:
        # Get the object ID (label)
        obj_id = region.label
        
        # Skip the background (label 0)
        if obj_id == 0:
            continue
        
        # Filter out regions based on pixel area
        if region.area < min_kernel_size:
            continue
        
        # Get the centroid coordinates
        centroid = region.centroid
        
        # Get the coordinates of all pixels belonging to this object
        pixel_coords = np.array(region.coords)  # (num_pixels, 2) array
        
        # Get the original bounding box of the region
        min_row, min_col, max_row, max_col = region.bbox
        
        # Expand the bounding box by padding, ensuring it stays within the image boundaries
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(hsi.shape[0], max_row + padding)
        max_col = min(hsi.shape[1], max_col + padding)
        
        # Store in dictionary
        object_data.append({
            'id': obj_id,
            'centroid': centroid,
            'pixels': pixel_coords,
            'bbox': (min_row, min_col, max_row, max_col)  # Store the expanded bounding box
        })

    object_data.sort(key=lambda x: (x['centroid'][1], x['centroid'][0])) 
    
    centroids = [obj['centroid'] for obj in object_data]
    centroids_sorted = sorted(centroids, key=lambda c: (c[0], c[1]))

    # Calculate the unique x-coordinates to determine columns
    x_coords = sorted(set(int(c[1]) for c in centroids_sorted))
    num_cols = len(x_coords)

    # Determine the three middle columns
    if num_cols >= 3:
        middle_start = num_cols // 2 - 1
        middle_cols = x_coords[middle_start: middle_start + 3]
    else:
        print("Insufficient columns to identify three middle columns.")
        middle_cols = x_coords  # fallback if there are fewer than 3 columns

    # Extract centroids belonging to the three middle columns
    middle_col_centroids = [c for c in centroids_sorted if int(c[1]) in middle_cols]

    # Display results
    print("Middle columns centroids:")
    for idx, centroid in enumerate(middle_col_centroids, start=1):
        print(f"{idx}. Centroid coordinates: {centroid}")