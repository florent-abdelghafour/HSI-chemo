import numpy as np
from skimage.color import  label2rgb
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image

def write_csv_header(output_file, header):
    """Write the header to the CSV file if it doesn't exist."""
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
def load_image(file_path):
    """Load an image and convert it to a NumPy array."""
    with Image.open(file_path) as img:
        print(f"Opened: {os.path.basename(file_path)}, Size: {img.size}, Mode: {img.mode}")
        return np.array(img) / 255.0     
           

def apply_multi_otsu_thresholding(channel, nb_cls):
    """Apply multi-Otsu thresholding to a single channel."""
    thresholds = threshold_multiotsu(channel, classes=nb_cls)
    segmented = np.digitize(channel, bins=thresholds)
    return segmented, thresholds

def clean_segmentation(seg, erode_thresh, custom_colors,target_classes):
    """Clean the segmentation by removing large background components."""
    new_label = int(np.nanmax(seg)) + 1
    for target in target_classes:
        mask = (seg == target)
        cc = label(mask)
        props = regionprops(cc)
        for prop in props:
            if prop.area > erode_thresh:
                seg[cc == prop.label] = new_label
    seg_int = np.nan_to_num(seg).astype(int)
    return label2rgb(seg_int, colors=custom_colors, bg_label=-1), seg


def clean_convex_hull(seg, new_label,target_classes):
    """Refine segmentation using Convex Hull."""
    leaf_mask = (seg == 0)
    labeled_leaf = label(leaf_mask)
    regions = regionprops(labeled_leaf)
    if not regions:
        return seg  # Return unchanged if no regions are found
    
     # Find the largest region
    largest_region = max(regions, key=lambda r: r.area)
    largest_coords = np.column_stack(largest_region.coords)
    
    
    leaf_coords = np.column_stack(largest_coords)  # Coordinates of the leaf region
    hull = ConvexHull(leaf_coords)
    delaunay = Delaunay(leaf_coords[hull.vertices])

    # Process pixels still labeled as 1 or 2
    for target in target_classes:
        target_coords = np.column_stack(np.where(seg == target))
        if len(target_coords) > 0:
            is_inside = delaunay.find_simplex(target_coords) >= 0
            outside_coords = target_coords[~is_inside]  # Points outside the hull
            for y, x in outside_coords:
                seg[y, x] = new_label  # Assign to new "outside hull" label

    return seg



def calculate_surface_areas(filled_leaf, leaf_mask):
    """Calculate the surface areas of the green and orange regions."""
    green_mask = filled_leaf
    orange_mask = np.logical_and(filled_leaf, ~leaf_mask)
    leaf_area = np.sum(green_mask)
    spot_area = np.sum(orange_mask)
    return green_mask, orange_mask, leaf_area, spot_area


def calculate_spot_properties(orange_mask):
    """Calculate properties of the orange regions (spots)."""
    regions = regionprops(label(orange_mask))
    sizes = [region.area for region in regions]
    min_size = np.min(sizes) if sizes else 0
    max_size = np.max(sizes) if sizes else 0
    mean_size = np.mean(sizes) if sizes else 0
    var_size = np.var(sizes) if sizes else 0
    num_spots = len(regions)
    return min_size, max_size, mean_size, var_size, num_spots


def plot_histograms(L_hist, L_bins, a_hist, a_bins, b_hist, b_bins, filename):
    """Plot histograms for the Lab channels."""
    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Lab Histograms - {filename}", fontsize=14)

    plt.subplot(1, 3, 1)
    plt.plot(L_bins[:-1], L_hist, color='gray')
    plt.title('L channel')
    plt.xlabel('Lightness')
    plt.ylabel('Pixel count')

    plt.subplot(1, 3, 2)
    plt.plot(a_bins[:-1], a_hist, color='red')
    plt.title('a channel')
    plt.xlabel('Green–Red')

    plt.subplot(1, 3, 3)
    plt.plot(b_bins[:-1], b_hist, color='blue')
    plt.title('b channel')
    plt.xlabel('Blue–Yellow')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    
    

def plot_segmentation_results(seg_rgb, title, legend_entries=None,cmap=None, save_path=None, filename=None):
    """Plot segmentation results with optional legends."""
    plt.figure(figsize=(6, 5))
    if cmap and legend_entries is None:
        plt.imshow(seg_rgb, cmap=cmap)
    else:
        plt.imshow(seg_rgb)
        
    plt.title(title)
    plt.axis('off')
    # Add legends if provided
    if legend_entries:
        y_pos = 0.95
        dy = 0.06  # Vertical spacing
        ax = plt.gca()
        for text, color in legend_entries:
            ax.text(0.02, y_pos, text,
                    transform=ax.transAxes,
                    fontsize=9,
                    color=color,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.5, pad=2, edgecolor='none'))
            y_pos -= dy

    # Save or show the plot
    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists
        save_file = os.path.join(save_path, f"{filename}.png")
        plt.savefig(save_file, bbox_inches='tight',dpi=900,pad_inches=0)
        plt.close()  # Close the plot to avoid displaying it
    else:
        plt.show()