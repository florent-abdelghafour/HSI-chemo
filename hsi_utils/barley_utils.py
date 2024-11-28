from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
import numpy as np



def project_hsi_VNIR(HSIreader, slice_step, n_samples=50000, nb_pca_comp=3, return_loadings=False):
    """
    Applies PCA to a hyperspectral image (HSI) in a memory-efficient manner.
    
    Parameters:
        HSIreader (object): Reader object providing access to the current HSI via `HSIreader.current_image`
            and the ability to extract pixel data via `HSIreader.extract_pixels(coords)`.
        hsi (ndarray): Hyperspectral image with shape (n_rows, n_cols, n_channels).
        pca (object): Pre-fitted PCA model with `fit_transform` and `components_` methods.
        slice_step (int): Number of rows to process in each slice for memory efficiency.
        n_samples (int): Number of random samples for PCA fitting. Default is 50000.
        nb_pca_comp (int): Number of PCA components to retain. If None, all components are used.
    
    Returns:
        final_pca_scores_img (ndarray): PCA-transformed image with shape (n_rows, n_cols, nb_pca_comp).
        pca_scores (ndarray): PCA scores for the sampled spectral data.
        pca_loadings (ndarray): PCA loadings matrix.
    """
    # Get image dimensions
    
    hsi=HSIreader.current_image
    n_rows, n_cols, n_channels = hsi.shape
    
    # Randomly sample spectral data for PCA fitting
    x_idx = np.random.randint(0, n_cols, size=n_samples)
    y_idx = np.random.randint(0, n_rows, size=n_samples)
    coords = list(zip(y_idx, x_idx))
    spectral_samples = np.array(HSIreader.extract_pixels(coords))
    
    # Fit PCA and calculate loadings
    pca = PCA(n_components=nb_pca_comp)
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Retain only the desired number of PCA components
    if nb_pca_comp is not None:
        pca_loadings = pca_loadings[:, :nb_pca_comp]
    
    # Process the HSI slice by slice
    pca_scores_imgs = []
    for start_row in range(0, n_rows, slice_step):
        end_row = min(start_row + slice_step, n_rows)
        subcube = hsi[start_row:end_row, :, :]
        
        # Flatten the subcube and apply PCA transformation
        subcube_flat = subcube.reshape(-1, n_channels)
        pca_scores = np.dot(subcube_flat, pca_loadings)
        pca_scores_img = pca_scores.reshape(end_row - start_row, n_cols, pca_loadings.shape[1])
        pca_scores_imgs.append(pca_scores_img)
        
        # Clean up for memory efficiency
        del subcube, subcube_flat, pca_scores, pca_scores_img

    # Combine all slices into the final PCA-transformed image
    final_pca_scores_img = np.concatenate(pca_scores_imgs, axis=0)
    del pca_scores_imgs
    
    if return_loadings ==True:
        return final_pca_scores_img,pca_loadings
    else:
        return final_pca_scores_img
    
   
   







def grid_sort(object_data, horizontal_tolerance):
    """
    Group objects by their y-coordinate (row-wise) and assign grid coordinates.

    Args:
        object_data (list): List of objects with 'centroid' data.
        horizontal_tolerance (int): Tolerance for grouping objects into the same row.

    Returns:
        list: Updated object data with grid coordinates assigned.
    """
    object_data.sort(key=lambda obj: obj['centroid'][0])  # Sort by x-coordinate

    rows = []
    current_row = []

    # Group objects by their y-coordinate (row-wise)
    for obj in object_data:
        if not current_row:
            current_row.append(obj)
        else:
            last_centroid_y = current_row[-1]['centroid'][0]
            # If the current object is close enough in y-coordinate, it belongs to the same row
            if abs(obj['centroid'][0] - last_centroid_y) < horizontal_tolerance:
                current_row.append(obj)
            else:
                rows.append(current_row)  # Add the previous row
                current_row = [obj]  # Start a new row

    # Add the last row if exists
    if current_row:
        rows.append(current_row)

    # Assign grid coordinates
    coord_to_obj = {}  # Dictionary to map (row, col) -> object
    for row_idx, row in enumerate(rows):
        row.sort(key=lambda obj: obj['centroid'][1])  # Sort by x-coordinate (column-wise)
        for col_idx, obj in enumerate(row):
            grid_coord = (row_idx + 1, col_idx + 1)
            obj['grid_coord'] = grid_coord
            coord_to_obj[grid_coord] = obj

    # Sort objects by their grid coordinates
    object_data.sort(key=lambda obj: (obj['grid_coord'][0], obj['grid_coord'][1]))
    
    for i,obj in enumerate(object_data):
        obj['id']=i
        
    return object_data,coord_to_obj



def merge_clusters(object_data, horizontal_threshold=100, vertical_threshold=300):
    """
    Clusters objects based on proximity thresholds and merges clusters into new objects.
    
    Parameters:
        object_data (list): List of objects, each containing 'id', 'grid_coord', 'centroid', and 'pixels'.
        horizontal_threshold (int): Horizontal proximity threshold for clustering.
        vertical_threshold (int): Vertical proximity threshold for clustering.
    
    Returns:
        list: Merged object data with updated attributes.
    """
    clusters = []  
    visited = set()  
    
    # Clustering based on proximity thresholds
    for obj in object_data:
        if obj['id'] in visited:
            continue 
        cluster = [obj]
        visited.add(obj['id'])
        queue = [obj]
    
        while queue:
            ref_obj = queue.pop(0)
            ref_centroid = ref_obj['centroid']

            for candidate in object_data:  
                if candidate['id'] in visited:
                    continue        
                candidate_centroid = candidate['centroid']
                h = abs(ref_centroid[1] - candidate_centroid[1])  
                v = abs(ref_centroid[0] - candidate_centroid[0])  
                                             
                if h <= horizontal_threshold and v <= vertical_threshold:
                    # Add the neighbor to the cluster
                    cluster.append(candidate)
                    visited.add(candidate['id'])
                    queue.append(candidate) 
                        
        clusters.append(cluster)

    print(f"Number of clusters formed: {len(clusters)}")
    
    # Merging clusters into single objects
    merged_object_data = [] 
    for i, cluster in enumerate(clusters):
        if len(cluster) == 1:
            merged_object = cluster[0]
            merged_object['id'] = i + 1
            merged_object_data.append(merged_object)  
            continue

        merged_pixel_coords = []
        for obj in cluster:
            if len(merged_pixel_coords) == 0:
                merged_pixel_coords = obj['pixels']
            else:
                merged_pixel_coords = np.vstack([merged_pixel_coords, obj['pixels']])
        merged_centroid = np.mean(merged_pixel_coords, axis=0)

        # Merge bounding box - calculate min/max coordinates for the new bounding box
        x_min, y_min = np.min(merged_pixel_coords, axis=0)
        x_max, y_max = np.max(merged_pixel_coords, axis=0)

        # Create merged object
        merged_object = {
            'id': i + 1,
            'centroid': merged_centroid,
            'bbox': [x_min, y_min, x_max, y_max],
            'grid_coord': cluster[0]['grid_coord'],  # Take the grid_coord of the first object
            'original_objects': [obj['id'] for obj in cluster],  # Keep track of merged IDs
            'pixels': merged_pixel_coords  # Store the merged pixels (optional)
        }
        # Add merged object to the list
        merged_object_data.append(merged_object)
    
    return merged_object_data




def color_labels(labeled_image):
    num_colors = len(np.unique(labeled_image)) - 2  
    colors = generate_custom_colors(num_colors)
    # Initialize a color image
    color_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3))

    # Set color for label 0 (black)
    color_image[labeled_image == 0] = [0, 0, 0]

    # Set color for label 1 (white)
    color_image[labeled_image == 1] = [0.5, 0.5, 0.5]

    # Create a palette for other labels
    unique_labels = np.unique(labeled_image)
    for label_value in unique_labels:
        if label_value > 1:
            color_idx = (label_value - 2) % num_colors  # Ensure index is within range
            color_image[labeled_image == label_value] = colors[color_idx]
    
    
    return color_image


def generate_custom_colors(num_colors):
    """
    Generate a list of diverse colors using HSL color space.
    """
    colors = []
    np.random.seed(0)  # For reproducibility
    for _ in range(num_colors):
        hue = np.random.rand()  # Random hue value between 0 and 1
        saturation = np.random.uniform(0.5, 0.9)  # Random saturation to avoid too pure colors
        lightness = np.random.uniform(0.3, 0.7)  # Random lightness to avoid too bright or dark colors
        color = hsv_to_rgb([hue, saturation, lightness])  # Convert to RGB
        colors.append(color)
    return colors



def convert_to_native_types(obj):
    """Recursively convert numpy types, arrays, and other complex types to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar types to native Python types
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)  # Convert tuples recursively
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}  # Recursively convert dicts
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]  # Recursively convert lists
    else:
        return obj 