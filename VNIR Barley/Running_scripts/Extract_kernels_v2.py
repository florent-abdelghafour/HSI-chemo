import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.measure import regionprops
import pickle
import numpy as np
import os

main_data_folder = "D:\\SWIR_germ_2nd_18\\ref_corrected"
dataset = HsiDataset(main_data_folder, data_ext='ref')

HSIreader = HsiReader(dataset)
# for swir use a padding of 1000 and pixel area of 1000
# for vnir use a padding of 10000 and pixel area of 10000
min_pixel_area = 1000
padding = 1000

for idx in range(len(dataset)):
    HSIreader.read_image(idx)
    metadata = HSIreader.current_metadata
    wv = HSIreader.get_wavelength()
    wv = [int(l) for l in wv]

    hypercube = HSIreader.get_hsi()
    n_samples = 10000

    x_idx = np.random.randint(0, hypercube.shape[0], size=n_samples)
    y_idx = np.random.randint(0, hypercube.shape[1], size=n_samples)

    spectral_samples = hypercube[x_idx, y_idx, :]

    nb_pca_comp = 3
    pca = PCA(n_components=nb_pca_comp)
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    score_img = HSIreader.project_pca_scores(pca_loadings)
    score_pc_ref = score_img[:, :, 1]
    thresholds = threshold_multiotsu(score_pc_ref, classes=2)
    segmented = np.digitize(score_pc_ref, bins=thresholds)

    labeled_image = label(segmented)
    binary_image = labeled_image > 0
    filled_binary_image = binary_fill_holes(binary_image)

    labeled_image = label(filled_binary_image)
    labeled_image = label(remove_small_objects(labeled_image > 0, min_size=20))

    regions = regionprops(labeled_image)
    object_data = []

    for region in regions:
        # Get the object ID (label)
        obj_id = region.label

        # Skip the background (label 0)
        if obj_id == 0:
            continue
        if region.area < min_pixel_area:
            continue

        # Get the centroid coordinates
        centroid = region.centroid

        # Get the coordinates of all pixels belonging to this object
        pixel_coords = np.array(region.coords)

        # Get the original bounding box of the region
        min_row, min_col, max_row, max_col = region.bbox

        # Adjust the padding to ensure it stays within the image boundaries
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(hypercube.shape[0], max_row + padding)
        max_col = min(hypercube.shape[1], max_col + padding)

        # Re-calculate height and width to ensure consistency
        height = max_row - min_row
        width = max_col - min_col

        # Ensure that the adjusted bounding box still contains the object
        if height <= 0 or width <= 0:
            print(f"Invalid padded bounding box for object {obj_id} in image {idx}. Skipping...")
            continue

        # Store in dictionary
        object_data.append({
            'id': obj_id,
            'centroid': centroid,
            'pixels': pixel_coords,
            'bbox': (min_row, min_col, max_row, max_col)
        })

    object_data.sort(key=lambda x: (x['centroid'][1], x['centroid'][0]))
    for new_id, obj in enumerate(object_data, start=1):
        obj['id'] = new_id

    # Convert list to dictionary with ordered keys
    ordered_object_data = {obj['id']: obj for obj in object_data}

    base_name = os.path.basename(HSIreader.dataset[idx]['data'])
    base_name = os.path.splitext(base_name)[0]
    object_folder = os.path.join(main_data_folder, 'Kernels_objects')

    # Create the object folder if it doesn't exist
    os.makedirs(object_folder, exist_ok=True)

    # Full path for saving the file
    file_path = os.path.join(object_folder, f'{base_name}.pkl')

    # Save the ordered dictionary to a file using pickle
    with open(file_path, 'wb') as f:
        pickle.dump(ordered_object_data, f)
