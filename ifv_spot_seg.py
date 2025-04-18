from rgb_utils.rgb_func import*
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2lab,label2rgb
import pywt
from sklearn.cluster import KMeans
from skimage.segmentation import find_boundaries

# Configuration
format = 'jpg'
img_path = "D:/IFV/"


output_file = os.path.join(img_path, 'leaf_metrics.csv')
result_path = os.path.join(img_path, 'results')
os.makedirs(result_path, exist_ok=True)

nb = 64  # Number of bins for histograms
nb_cls = 3  # Number of classes for multi-Otsu thresholding
erode_thresh = 2000  # Threshold for removing large components

custom_colors = [(0.0, 1.0, 0.0),  # Green for leaf
                 (0.8, 0.8, 0.8),  # Light gray for background
                 (1.0, 1.0, 0.0),  # Yellow for halo
                 (0.0, 0.0, 0.0)]  # Black for outside convex hull


# Define new colors for the 3 clusters
cluster_colors = [
    (1.0, 0.0, 0.0),  # Red for Cluster 1
    (0.0, 0.0, 1.0),  # Blue for Cluster 2
    (1.0, 1.0, 0.0)   # Yellow for Cluster 3
]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) 
# Configuration
format = 'jpg'
img_path = "D:/IFV/"
output_file = os.path.join(img_path, 'leaf_metrics.csv')

nb = 64  # Number of bins for histograms
nb_cls = 3  # Number of classes for multi-Otsu thresholding
erode_thresh = 2000  # Threshold for removing large components

custom_colors = [(0.0, 1.0, 0.0),  # Green for leaf
                 (0.8, 0.8, 0.8),  # Light gray for background
                 (1.0, 1.0, 0.0),  # Yellow for halo
                 (0.0, 0.0, 0.0)]  # Black for outside convex hull

# Write the CSV header
header = ["Image Name", "Surface of Leaf (Green)", "Surface of Spot (Orange)",
          "% Coverage", "Min Spot Size", "Max Spot Size", "Mean Spot Size",
          "Var Spot Size", "Num Spots"]
write_csv_header(output_file, header)

# Process each image
for idx, filename in enumerate(os.listdir(img_path)):
#    if idx==1: 
    if filename.lower().endswith(format):
        file_path = os.path.join(img_path, filename)
        try:
            
            image_results_path = os.path.join(result_path, os.path.splitext(filename)[0])
            os.makedirs(image_results_path, exist_ok=True)
            # Load and process the image
            img_np = load_image(file_path)

            # Convert to Lab color space
            img_lab = rgb2lab(img_np)
            L, a, b = img_lab[..., 0], img_lab[..., 1], img_lab[..., 2]

            # Compute histograms
            L_hist, L_bins = np.histogram(L.ravel(), bins=nb, range=(0, 100))
            a_hist, a_bins = np.histogram(a.ravel(), bins=nb, range=(-128, 127))
            b_hist, b_bins = np.histogram(b.ravel(), bins=nb, range=(-128, 127))

            # Plot histograms
            # plot_histograms(L_hist, L_bins, a_hist, a_bins, b_hist, b_bins, filename)

            # Apply multi-Otsu thresholding
            a_seg, a_thresh = apply_multi_otsu_thresholding(a, nb_cls)
            
            a_label = label2rgb(a_seg, colors=custom_colors, bg_label=-1)
            legend_entries = [
                ("1 - Leaf", "green"),
                ("2 - Background", "gray"),
                ("3 - Halo", "yellow")
            ]
            # plot_segmentation_results(a_label, f"'a' Channel Thresholding - {filename}", legend_entries, save_path=image_results_path,
            #                            filename="a_channel_thresholding")


            # Clean segmentation
            a_seg_cleaned, seg = clean_segmentation(a_seg.astype(float), erode_thresh, custom_colors, target_classes=[1, 2])
             # Plot cleaned segmentation
            seg_cleaned_rgb = label2rgb(seg, colors=custom_colors, bg_label=-1)
            # plot_segmentation_results(seg_cleaned_rgb, f"Cleaned Segmentation - {filename}", legend_entries,save_path=image_results_path,
            #                            filename="cleaned_segmentation")

            new_label=int(np.nanmax(seg)) + 1

            # Refine segmentation with Convex Hull
            seg = clean_convex_hull(seg, new_label=new_label-1, target_classes=[0,1, 2])
            seg_hull_rgb = label2rgb(seg, colors=custom_colors, bg_label=-1)
            legend_entries.append(("4 - Outside Convex Hull", "black"))
            # plot_segmentation_results(seg_hull_rgb, f"Convex Hull Refinement - {filename}",   
            #                           legend_entries = [("1 - leaf", custom_colors[0]),("4 - outside convex hull", custom_colors[new_label-1])], save_path=image_results_path,
            #                            filename="convex_hull_refinement")
                                 



            # Create masks
            leaf_mask = seg == 0
            filled_leaf = binary_fill_holes(leaf_mask)
            # Plot filled leaf mask
            filled_leaf_visualization = np.zeros_like(seg, dtype=float)
            filled_leaf_visualization[filled_leaf] = 1  # Green color where the leaf is filled
            # plot_segmentation_results(filled_leaf_visualization, f"Filled Leaf Mask - {filename}",cmap='Greens',save_path=image_results_path,
            #                            filename="filled_leaf_mask")
            
            
            
            combined_mask = np.logical_and(filled_leaf, leaf_mask)
            combined_visualization = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=float)
            label_image = np.zeros((combined_visualization.shape[0], combined_visualization.shape[1]), dtype=int)
            combined_visualization[filled_leaf] =  [0.9,0.5,0.2]
            combined_visualization[leaf_mask] =  [0,1,0]
            # plot_segmentation_results(combined_visualization, f"Leaf Mask + spots - {filename}",save_path=image_results_path,
            #                            filename='Leaf Mask + spots')
            
            
            
            
            green_mask, orange_mask, leaf_area, spot_area = calculate_surface_areas(filled_leaf, leaf_mask)
            # Plot green mask (leaf)
            green_visualization = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=float)
            green_visualization[green_mask] = [0, 1, 0]  # Green for the leaf
            # plot_segmentation_results(green_visualization, f"Green Mask (Leaf) - {filename}",save_path=image_results_path,
            #                            filename="green_leaf_mask")

            # Plot orange mask (spots)
            orange_visualization = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=float)
            orange_visualization[orange_mask] = [0.9, 0.5, 0.2]  # Orange for the spots
            # plot_segmentation_results(orange_visualization, f"Orange Mask (Spots) - {filename}", save_path=image_results_path,
            #                            filename="spot_mask")
            # Calculate spot properties
            min_size, max_size, mean_size, var_size, num_spots = calculate_spot_properties(orange_mask)

            # Calculate % coverage
            coverage_percentage = (spot_area / leaf_area * 100) if leaf_area > 0 else 0

           

            wavelet_type = 'sym4'
          # Perform stationary wavelet decomposition globally for each channel
            L_coeffs = pywt.swt2(L, wavelet_type, level=1)  # Stationary Wavelet Transform
            a_coeffs = pywt.swt2(a, wavelet_type, level=1)
            b_coeffs = pywt.swt2(b, wavelet_type, level=1)

            # Extract approximation and detail coefficients globally
            L_cA, (L_cH, L_cV, L_cD) = L_coeffs[0]  # Level 1 coefficients
            a_cA, (a_cH, a_cV, a_cD) = a_coeffs[0]
            b_cA, (b_cH, b_cV, b_cD) = b_coeffs[0]
                                    
            # Label connected components in the orange mask
            labeled_mask = label(orange_mask)
            regions = regionprops(labeled_mask)   
            
            # Initialize a list to store features
            all_features = []
                     
            # Iterate over each connected component (region) in the orange mask
            for region in regions:
                # Get the coordinates of the current region
                coords = region.coords
                        
                # Gather wavelet coefficients for the region
                L_region_cA = L_cA[coords[:, 0], coords[:, 1]]
                L_region_cH = L_cH[coords[:, 0], coords[:, 1]]
                L_region_cV = L_cV[coords[:, 0], coords[:, 1]]
                L_region_cD = L_cD[coords[:, 0], coords[:, 1]]

                a_region_cA = a_cA[coords[:, 0], coords[:, 1]]
                a_region_cH = a_cH[coords[:, 0], coords[:, 1]]
                a_region_cV = a_cV[coords[:, 0], coords[:, 1]]
                a_region_cD = a_cD[coords[:, 0], coords[:, 1]]

                b_region_cA = b_cA[coords[:, 0], coords[:, 1]]
                b_region_cH = b_cH[coords[:, 0], coords[:, 1]]
                b_region_cV = b_cV[coords[:, 0], coords[:, 1]]
                b_region_cD = b_cD[coords[:, 0], coords[:, 1]]  
                
                # Compute mean and variance of coefficients for the region
                L_mean_cA, L_var_cA = np.mean(L_region_cA), np.var(L_region_cA)
                L_mean_cH, L_var_cH = np.mean(L_region_cH), np.var(L_region_cH)
                L_mean_cV, L_var_cV = np.mean(L_region_cV), np.var(L_region_cV)
                L_mean_cD, L_var_cD = np.mean(L_region_cD), np.var(L_region_cD)

                a_mean_cA, a_var_cA = np.mean(a_region_cA), np.var(a_region_cA)
                a_mean_cH, a_var_cH = np.mean(a_region_cH), np.var(a_region_cH)
                a_mean_cV, a_var_cV = np.mean(a_region_cV), np.var(a_region_cV)
                a_mean_cD, a_var_cD = np.mean(a_region_cD), np.var(a_region_cD)

                b_mean_cA, b_var_cA = np.mean(b_region_cA), np.var(b_region_cA)
                b_mean_cH, b_var_cH = np.mean(b_region_cH), np.var(b_region_cH)
                b_mean_cV, b_var_cV = np.mean(b_region_cV), np.var(b_region_cV)
                b_mean_cD, b_var_cD = np.mean(b_region_cD), np.var(b_region_cD)

                # Append features for the current region
                all_features.append([
                    L_mean_cA, L_var_cA, L_mean_cH, L_var_cH, L_mean_cV, L_var_cV, L_mean_cD, L_var_cD,
                    a_mean_cA, a_var_cA, a_mean_cH, a_var_cH, a_mean_cV, a_var_cV, a_mean_cD, a_var_cD,
                    b_mean_cA, b_var_cA, b_mean_cH, b_var_cH, b_mean_cV, b_var_cV, b_mean_cD, b_var_cD
                ])
                                
            #             # Convert features to a NumPy array
            features = np.array(all_features)
       
            labels = kmeans.fit_predict(features)

            # Map cluster labels back to the original image
            clustered_mask = np.zeros_like(orange_mask, dtype=int)
            
            for region, lab in zip(regions, labels):
                coords = region.coords
                clustered_mask[coords[:, 0], coords[:, 1]] = lab + 1  # Start labels from 1

            # # Create an RGB image for the final visualization
            final_image = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=float)

            # Overlay the clustered regions with their respective colors
            for cluster_label, color in enumerate(cluster_colors, start=1):
                final_image[clustered_mask == cluster_label] = color

            # Add the contours of the green leaf mask
            green_contours = find_boundaries(green_mask, mode='thick')
            final_image[green_contours] = [0.0, 1.0, 0.0]  # Green for the contours

            # Save and visualize the final image
            plot_segmentation_results(
                final_image,
                f"Final Image with Clusters and Green Contours - {filename}",
                save_path=image_results_path,
                filename="final_image_with_clusters_and_contours"
            )

            # result_dict = {
            # "Image Name": filename,
            # "Surface of Leaf (Green)": leaf_area,
            # "Surface of Spot (Orange)": spot_area,
            # "% Coverage": coverage_percentage,
            # "Min Spot Size": min_size,
            # "Max Spot Size": max_size,
            # "Mean Spot Size": mean_size,
            # "Var Spot Size": var_size,
            # "Num Spots": num_spots
            # }
                               
            #             # Write results to CSV
            # with open(output_file, 'a', newline='') as file:
            #     writer = csv.DictWriter(file, fieldnames=result_dict.keys())
            #     writer.writerow(result_dict)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")