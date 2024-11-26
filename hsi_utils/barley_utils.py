

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
    for row_idx, row in enumerate(rows):
        row.sort(key=lambda obj: obj['centroid'][1])  # Sort by x-coordinate (column-wise)
        for col_idx, obj in enumerate(row):
            obj['grid_coord'] = (row_idx + 1, col_idx + 1)

    # Sort objects by their grid coordinates
    object_data.sort(key=lambda obj: (obj['grid_coord'][0], obj['grid_coord'][1]))
    return object_data