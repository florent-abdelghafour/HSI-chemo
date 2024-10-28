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
    
    # plt.figure()
    # plt.imshow(color_image)
    # plt.show()
    
    
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
    
    # Set your vertical distance tolerance
    vertical_tolerance = 500  # Adjust this value based on your specific needs

    # Initialize a list to store rows of centroids
rows_of_centroids = []

# Create a copy of the sorted object_data to process
remaining_objects = sorted(object_data, key=lambda obj: (obj['centroid'][0], obj['centroid'][1]))

while remaining_objects:
    # Step 1: Find the most top-left object
    current_object = remaining_objects.pop(0)  # Get the top-left object
    current_centroid = current_object['centroid']  # Current centroid (row, col)
    current_y = current_centroid[0]  # Current y-coordinate (row)

    # Create a new row and add the current object
    current_row = [current_object]
    
    # Step 2: Find objects in the same row (within vertical tolerance)
    for obj in remaining_objects[:]:  # Use a copy of the list for safe iteration
        obj_centroid = obj['centroid']
        obj_y = obj_centroid[0]  # Y-coordinate (row)
        
        # Check if this object is within the vertical tolerance
        if abs(obj_y - current_y) <= vertical_tolerance:
            current_row.append(obj)
            remaining_objects.remove(obj)  # Remove from remaining
    
    # Step 3: Sort current_row by their x-coordinate (col)
    current_row.sort(key=lambda obj: obj['centroid'][1])  # Sort by column (x-coordinate)
    
    # Add the current row to rows_of_centroids
    rows_of_centroids.append(current_row)

# Step 4: Extract the three middle centroids from each row
middle_centroids = []

for row in rows_of_centroids:
    # Extract the centroids' x-coordinates
    x_coords = [obj['centroid'][1] for obj in row]  # Extract x-coordinates
    if len(x_coords) > 0:
        # Calculate the indices for the three middle centroids
        mid_index = len(row) // 2
        mid_centroids = []

        # Determine the range to extract the three middle centroids
        if len(row) % 2 == 0:  # Even number of centroids
            mid_centroids = row[mid_index-1:mid_index+2]  # Two centroids on either side
        else:  # Odd number of centroids
            mid_centroids = row[mid_index-1:mid_index+2]  # One on the left, one on the right

        middle_centroids.extend(mid_centroids)

# # Display the selected middle centroids
# print("Middle centroids from each fuzzy row:")
# for idx, obj in enumerate(middle_centroids, start=1):
#     print(f"{idx}. Object ID: {obj['id']}, Centroid coordinates: {obj['centroid']}")
    
    
# # Plotting the labeled image with object IDs
# plt.figure(figsize=(10, 8))
# plt.imshow(color_image)
# plt.title("Labeled Image with Object IDs")

# # Annotate the centroids with object IDs
# for obj in middle_centroids:
#     # Get the centroid coordinates and ID
#     centroid = obj['centroid']
#     obj_id = obj['id']
    
#     # Annotate the image with the object ID at the centroid location
#     plt.annotate(str(obj_id), 
#                  (centroid[1], centroid[0]),  # (col, row)
#                  color='white', 
#                  fontsize=10, 
#                  ha='center', 
#                  va='center',
#                  bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

# # Show the plot
# plt.axis('off')  # Hide axes
# plt.show()

object_folder = os.path.join(main_data_folder, 'object')
object_file = 'object_data.pkl'

# Create the object folder if it doesn't exist
os.makedirs(object_folder, exist_ok=True)

# Full path for saving the file
file_path = os.path.join(object_folder, object_file)

# Save the ordered dictionary to a file using pickle
with open(file_path, 'wb') as f:
    pickle.dump(middle_centroids, f)