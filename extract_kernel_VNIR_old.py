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
import json


main_data_folder = "D:/VNIR_barley"    

kernel_obj_folder = main_data_folder + '/kernel_obj'
if not os.path.exists(kernel_obj_folder ):
    os.makedirs(kernel_obj_folder ) 
# Initialize the HSI dataset and define file extension: contains all paths of hdr and data files
dataset =HsiDataset(main_data_folder,data_ext='hyspex')
# Initialize the HSI reader: class containg info about the image and allow to load and operate on the hsi
HSIreader = HsiReader(dataset)
nb_pca_comp =3
pca = PCA(n_components=nb_pca_comp)
slice_step = 1000
min_kernel_size=1000
padding=100
n_samples = 50000
horizontal_tolerance = 50  

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Loop through each hyperspectral image in the dataset
for idx in range(len(dataset)):  
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    image_name  = HSIreader.current_name
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hsi=HSIreader.current_image
    n_rows, n_cols, n_channels =hsi.shape
    
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

    object_data.sort(key=lambda obj: obj['centroid'][1])  # Sort by x-coordinate

    columns_of_centroids = []
    current_column = []

    for obj in object_data:
        if not current_column:
            current_column.append(obj)
        else:
            last_centroid_x = current_column[-1]['centroid'][1]
            if abs(obj['centroid'][1] - last_centroid_x) <= horizontal_tolerance:
                current_column.append(obj)
            else:
                columns_of_centroids.append(current_column)
                current_column = [obj]

    # Add the last column if not empty
    if current_column:
        columns_of_centroids.append(current_column)

    # Now process the last three columns
    last_three_columns = columns_of_centroids[-3:]

    # Convert list to dictionary with ordered keys
    for column in last_three_columns:
            column.sort(key=lambda obj: obj['centroid'][0])
            
    # Initialize index counter
    idx = 1
    n_rows = max(len(column) for column in last_three_columns)  # Find max number of rows among columns

    # Pad columns with None if they have fewer objects than the max row count
    for column in last_three_columns:
        while len(column) < n_rows:
            column.append(None)

    # Add grid_coord and idx to each object's dictionary
    for row in range(n_rows):
        for col, column in enumerate(last_three_columns):
            obj = column[row]
            if obj:
                obj['grid_coord'] = (row + 1, col + 1)  # Store as (row, col) tuple
                obj['idx'] = idx
                idx += 1

    # Sort the object_data list by 'idx' for final top-left to bottom-right order
    object_data_sorted = sorted([obj for obj in object_data if 'idx' in obj], key=lambda obj: obj['idx'])
    
    # plt.figure()
    # plt.imshow(color_image)
    # plt.title('Color-Mapped Labeled Image with Ordered Indices')
    # for obj in object_data_sorted:
    #     centroid_y, centroid_x = obj['centroid']
    #     id = obj['idx']
    #     plt.annotate(f"{id}", xy=(centroid_x, centroid_y), color='white', fontsize=8, 
    #                 ha='center', va='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
    # plt.axis('off')
    # plt.show()
    
    kernel_file_path = kernel_obj_folder+ f'/{image_name}/{image_name}_kernel_objetcs.json'
    if not os.path.exists(kernel_file_path ):
        os.makedirs(kernel_file_path) 
    with open(kernel_file_path, 'w') as json_file:
            json.dump(object_data_sorted, json_file, indent=4)
    
    

            
            
            
            
            
            
            
            
    ##########################################################################################################
    #                                ORDER AS ROWS AND COLS Ri, Cj
    ##########################################################################################################
    # plt.figure()
    # plt.imshow(color_image)
    # plt.title('Color-Mapped Labeled Image')
    
    # grid_ordered_objects = []
    # n_rows = max(len(column) for column in last_three_columns)  # Find the max number of rows among columns

    # # Pad columns with None if they have fewer objects than the max row count to align rows
    # for column in last_three_columns:
    #     while len(column) < n_rows:
    #         column.append(None)  # Use None for missing entries

    # # Create the grid by combining objects in a row-major order
    # for row in range(n_rows):
    #     for col in range(len(last_three_columns)):
    #         obj = last_three_columns[col][row]
    #         grid_ordered_objects.append((col + 1, row + 1, obj))  # (column, row, object)

    # # Annotate the image with grid positions
    # for col, row, obj in grid_ordered_objects:
    #     if obj:
    #         # Get the centroid coordinates for the annotation
    #         centroid_y, centroid_x = obj['centroid']  # Note: `centroid` is (row, col) format
    #         plt.annotate(f'R{row}C{col}', xy=(centroid_x, centroid_y), color='white', fontsize=8, 
    #                     ha='center', va='center', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

    # plt.axis('off')
    # plt.show()
    ##########################################################################################################
    ##########################################################################################################
    
    
    

    # object_folder = os.path.join(main_data_folder, 'object')
    # object_file = 'object_data.pkl'

    # # Create the object folder if it doesn't exist
    # os.makedirs(object_folder, exist_ok=True)

    # # Full path for saving the file
    # file_path = os.path.join(object_folder, object_file)

    # # Save the ordered dictionary to a file using pickle
    # with open(file_path, 'wb') as f:
    #     pickle.dump(middle_centroids, f)