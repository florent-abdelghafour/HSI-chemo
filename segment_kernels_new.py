import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist

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
    print('path invalid')


HSIreader = HsiReader(dataset)

nb_pca_comp=3
pca = PCA(n_components=nb_pca_comp)
slice_step = 1000


kernel_comp=1
min_kernel_size = 1000
padding=50
horizontal_tolerance =100

for idx in range(len(dataset)):
 if idx==1:
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hsi=HSIreader.current_image
    image_name  = HSIreader.current_name
    
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
        
        del subcube, subcube_flat, pca_scores, pca_scores_img

    final_pca_scores_img = np.concatenate(pca_scores_imgs, axis=0)
    del pca_scores_imgs
    
    score_pc_ref = final_pca_scores_img[:,:,kernel_comp]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=2)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    # plt.figure()
    # plt.imshow(segmented)
    # plt.show()
    
    #get a labelled image 
    labeled_image = label(segmented)

    #fill holes in small object
    filled_binary_image = binary_fill_holes(labeled_image)
    
    # plt.figure()
    # plt.imshow(filled_binary_image)
    # plt.show()


    #remove artefacts of segmentation
    labeled_image = label(filled_binary_image)
    labeled_image= label(remove_small_objects(labeled_image > 0, min_size=min_kernel_size))

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

   
    # object_data  = grid_sort(object_data,horizontal_tolerance)
    # max_col = max(obj['grid_coord'][1] for obj in object_data)
    # filtered_objects = [obj for obj in object_data if obj['grid_coord'][1] > max_col - 3]


    # #Check if kernel numbering is OK 
    plt.figure(figsize=(10, 8))
    plt.imshow(color_image)

    # Step 5: Assign ID to each object and annotate
    for i, obj in enumerate(object_data):
        obj['id'] = i + 1  # Assign the ID (index + 1)
        centroid = obj['centroid']  # Get the centroid (y, x) of the object
        row, col = obj['grid_coord']  # Get the row and column index
        
        # # Annotate the object with its ID at the centroid location
        # plt.text(centroid[1], centroid[0], f'{obj["id"]}', 
        #         color='white', fontsize=8, ha='center', va='center',
        #         fontweight='bold', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
        
        plt.text(centroid[1], centroid[0], f'{row},{col}', 
                color='white', fontsize=8, ha='center', va='center',
                fontweight='bold', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))

    # Display the image with annotations
    plt.title("Annotated Image with Object IDs")
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
    
    
    
    
  
    # total_distance = 0
    # num_pairs = len(object_data) - 1
    
    # for i in range(num_pairs):
    #     centroid1 = object_data[i]['centroid']
    #     centroid2 = object_data[i + 1]['centroid']
    #     distance = ((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2) ** 0.5
    #     total_distance += distance
    # mean_distance = total_distance / num_pairs
    

    # merge_range = 5  
    # abnormal_threshold = mean_distance * 0.5
    # bad_apples =[]
    # dist=[]
    
    # for k, obj in enumerate(object_data):
    #     centroid_k = obj['centroid']
   
    #     for offset in range(-merge_range, merge_range + 1):
    #         # Skip self-comparison and ensure index is in range
    #         neighbor_idx = k + offset
    #         if offset == 0 or not (0 <= neighbor_idx < len(object_data)):
    #             continue

    #         # Compute the distance
    #         neighbor = object_data[neighbor_idx]
    #         centroid_neighbor = neighbor['centroid']
    #         distance = (2*(centroid_k[0] - centroid_neighbor[0]) ** 2 + 
    #                     (centroid_k[1] - centroid_neighbor[1]) ** 2) ** 0.5

    #         # Check if the distance is abnormal
    #         if distance < abnormal_threshold:
    #             bad_apples.append(neighbor['centroid'])
    #             dist.append(distance)
                
    # plt.figure(figsize=(10, 8))
    # plt.imshow(color_image)

    # # Overlay an 'X' on each bad apple's centroid
    # for centroid in bad_apples:
    #     plt.plot(centroid[1], centroid[0], 'x', color='red', markersize=10, markeredgewidth=2)

    # # Add a title and hide the axes for clarity
    # plt.title("Bad Apples Identified")
    # plt.axis('off')  # Hide axes for better visualization
    # plt.show()
            




# # for i, obj in enumerate(object_data):
# #     obj['id'] = i + 1  # Assign the ID (index + 1)
# #     obj['centroid'] = list(obj['centroid'])  # Convert tuple to list
# #     obj['pixels'] = obj['pixels'].tolist()  # Convert ndarray to list


# # kernel_file_path = main_data_folder+ f'/kernel_data/{image_name}'
# # if not os.path.exists(kernel_file_path ):
# #     os.makedirs(kernel_file_path) 
# # with open(kernel_file_path+'/kernel.json', 'w') as json_file:
# #         json.dump(object_data, json_file, indent=4)























# # Analyze and merge potentially split objects
#     checked = set()
    
#     sub_kernel_size =3*min_kernel_size
#     ref_size= 7*min_kernel_size
#     k_h_sep = 100  # 
#     k_v_sep = 500  # 
    
#     # To store the centroids of merged objects
#     split_obj = []
#     reference_objects = []
    
#     i = 0
#     while i < len(object_data):
#         obj1 = object_data[i]
#         # Skip already merged objects
#         if obj1['id'] in checked:
#             i += 1
#             continue
        
#         if  len(obj1['pixels']) < sub_kernel_size:
#             obj1_centroid = np.array(obj1['centroid'])
#             merged_bbox = obj1['bbox']
#             merged_pixel_coords = obj1['pixels']
            
#             merged = False
#             for j in range(len(object_data)):
#                 obj2 = object_data[j]
#                 # Skip already merged objects or self-comparison
#                 if obj2['id'] in checked or obj1['id'] == obj2['id']:
#                     continue
    
#                 if len(obj2['pixels']) >= ref_size:
#                     obj2_centroid = np.array(obj2['centroid'])
#                     h_distance = abs(obj1_centroid[1] - obj2_centroid[1])  
#                     v_distance = abs(obj1_centroid[0] - obj2_centroid[0]) 
 
#                     if h_distance < k_h_sep and v_distance < k_v_sep:
#                         # Merge object2 into object1
#                         merged_pixel_coords = np.vstack([merged_pixel_coords, obj2['pixels']])
#                         merged_bbox = (
#                             min(merged_bbox[0], obj2['bbox'][0]),
#                             min(merged_bbox[1], obj2['bbox'][1]),
#                             max(merged_bbox[2], obj2['bbox'][2]),
#                             max(merged_bbox[3], obj2['bbox'][3]),
#                         )
#                         # Update the centroid of the merged object
#                         merged_centroid = np.mean(merged_pixel_coords, axis=0)
#                         split_obj.append(obj1_centroid)
#                         reference_objects.append(obj2_centroid)

#                         merged = True
#                         # Update the merged object
#                         merged_object = {
#                             'id': obj2['id'],  # Keep the ID of the first object
#                             'centroid': tuple(merged_centroid),
#                             'pixels': merged_pixel_coords,
#                             'bbox': merged_bbox
#                         }
    
#                         object_data[i] = merged_object
                    
#                         # Mark obj2 as merged
#                         checked.add(obj1['id'])
#                         # Remove obj2 from the list
#                         object_data = [obj for obj in object_data if obj['id'] != obj1['id']]

#                     break        
#             if  not merged:
#                 # No merging happened, so mark obj1 as checked and move to next object
#                 checked.add(obj1['id'])
#                 i += 1
#         else:
#             # If no merging is needed, just move to the next object
#             checked.add(obj1['id'])
#             i += 1
 
#     print (f" split obj found: {len(split_obj)}")
    
    
#     plt.figure()
#     plt.imshow(color_image)
#     plt.title('Color-Mapped Labeled Image')

#     # Plot the merged centroids
#     for centroid in split_obj:
#         plt.scatter(centroid[1], centroid[0], color='red', marker='x')
#     for centroid in reference_objects:
#         plt.scatter(centroid[1], centroid[0], color='green', marker='x')

#     plt.legend()
#     plt.axis('off')
#     plt.show()


 
    # plt.figure()
    # plt.imshow(color_image)
    # plt.title('Color-Mapped Labeled Image')

    # # Plot the centroids of the merged objects and annotate their sizes
    # for obj in object_data:
    #     centroid = obj['centroid']
    #     size = len(obj['pixels'])  # The size of the object is the number of pixels
    #     plt.scatter(centroid[1], centroid[0], color='red', marker='x')  # Plot the centroid
    #     plt.text(centroid[1], centroid[0], f'{size}', color='blue', fontsize=8)  # Annotate the size

    # plt.legend()
    # plt.axis('off')
    # plt.show()
 
 