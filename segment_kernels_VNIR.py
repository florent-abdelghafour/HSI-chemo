import os
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

import json


main_data_folder = 'D:/HSI data/VNIR_barley' 
# D:/HSI data/Barley_ground_30cm_SWIR
# 'D:/HSI data/VNIR_barley' 

dataset =HsiDataset(main_data_folder,data_ext='ref')
nb_images = len(dataset)
#check if data path exists !
if os.path.isdir(main_data_folder):
    if nb_images>0:
        print(f"dataset  is valid and contains {nb_images} image(s)")
    else:
        print('empty dataset')
else:
    raise FileNotFoundError(f"Invalid path: {main_data_folder}")


HSIreader = HsiReader(dataset)

nb_pca_comp=3
pca = PCA(n_components=nb_pca_comp)
slice_step = 1000

kernel_comp=1
min_kernel_size = 1000
horizontal_tolerance =100

h_t=100 # horizontal_threshold 
v_t=300 # vertical_threshold

for idx in range(len(dataset)): 
  if idx==0:  
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hsi=HSIreader.current_image
    image_name  = HSIreader.current_name
    
    pca_score_image = project_hsi_VNIR(HSIreader, slice_step, n_samples=50000, nb_pca_comp=3)
    score_pc_ref = pca_score_image[:,:,kernel_comp]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=2)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    # plt.figure()
    # plt.imshow(segmented)
    # plt.show()

    labeled_image = label(segmented)  #get a labelled image   
    filled_binary_image = binary_fill_holes(labeled_image)   #fill holes in small object
    labeled_image = label(filled_binary_image)
    labeled_image= label(remove_small_objects(labeled_image > 0, min_size=min_kernel_size)) # remove artefacts of segmentation

    color_image = color_labels(labeled_image)
    # plt.figure()
    # plt.imshow(color_image)
    # plt.title('Color-Mapped Labeled Image')
    # plt.axis('off')
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
        min_row = max(0, min_row )
        min_col = max(0, min_col)
        max_row = min(hsi.shape[0], max_row)
        max_col = min(hsi.shape[1], max_col)
        
        # Store in dictionary
        object_data.append({
            'id': obj_id,
            'centroid': centroid,
            'pixels': pixel_coords,
            'bbox': (min_row, min_col, max_row, max_col)  # Store the expanded bounding box
        })

    object_data,coord_to_obj  = grid_sort(object_data,horizontal_tolerance)
    
    #Check if kernel numbering is OK 
    # plt.figure(figsize=(10, 8))
    # plt.imshow(color_image)

    # # Step 5: Assign ID to each object and annotate
    # for i, obj in enumerate(object_data):
    #     obj['id'] = i + 1  # Assign the ID (index + 1)
    #     centroid = obj['centroid']  # Get the centroid (y, x) of the object
    #     row, col = obj['grid_coord']  # Get the row and column index
        
    #     # # Annotate the object with its ID at the centroid location
    #     # plt.text(centroid[1], centroid[0], f'{obj["id"]}', 
    #     #         color='white', fontsize=8, ha='center', va='center',
    #     #         fontweight='bold', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
        
    #     plt.text(centroid[1], centroid[0], f'{row},{col}', 
    #             color='white', fontsize=8, ha='center', va='center',
    #             fontweight='bold', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))

    # # Display the image with annotations
    # plt.title("Annotated Image with Object IDs")
    # plt.axis('off')  # Hide axes for better visualization
    # plt.show()
    
    merged_object_data = merge_clusters(object_data, horizontal_threshold=h_t, vertical_threshold=v_t)  
    merged_object_data,_ =grid_sort(merged_object_data,horizontal_tolerance) 

    #take last 3 columns
    n_cols = max(obj['grid_coord'][1] for obj in merged_object_data)
    merged_object_data = [obj for obj in merged_object_data if obj['grid_coord'][1] > n_cols - 3]
    
    print(len(merged_object_data))

    #Check if kernel numbering is OK after merging
    plt.figure(figsize=(10, 8))
    plt.imshow(color_image)

    # Step 5: Assign ID to each object and annotate
    for obj in merged_object_data:
        centroid = obj['centroid']  # Get the centroid (y, x) of the object
        # Annotate the object with its ID at the centroid location
        plt.text(centroid[1], centroid[0], f"{obj['id']}", 
                color='white', fontsize=8, ha='center', va='center',
                fontweight='bold', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'));

    # Display the image with annotations
    plt.title("Annotated Image with Object IDs")
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
    

    for i, obj in enumerate(merged_object_data):
        obj['id'] = i + 1  # Assign the ID (index + 1)
        merged_object_data[i] = convert_to_native_types(obj)

    kernel_file_path = main_data_folder+ f'/kernel_data/{image_name}'
    if not os.path.exists(kernel_file_path ):
        os.makedirs(kernel_file_path) 
    with open(kernel_file_path+'/kernel.json', 'w') as json_file:
            json.dump(merged_object_data, json_file, indent=4)




