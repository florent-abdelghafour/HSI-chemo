from matplotlib.colors import hsv_to_rgb
import numpy as np

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