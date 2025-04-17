from PIL import Image
import os
import numpy as np
from skimage.color import rgb2lab,label2rgb
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from scipy.spatial import ConvexHull, Delaunay
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
import csv

format ='jpg'
img_path="D:/IFV/"

output_file = os.path.join(img_path, 'leaf_metrics.csv')  

nb=64 #number of bins for histograms 
nb_cls =3 #number of classes for multi-Otsu thresholding
erode_thresh =2000

custom_colors = [(0.0, 1.0, 0.0),  # green for leaf
                 (0.8, 0.8, 0.8),  #light gray for background
                 (1.0, 1.0, 0.0), # yellow for halo
                 (0.0,0.0,0.0,)]  #black for real background = non within leaf

if not os.path.exists(output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Surface of Leaf (Green)", "Surface of Spot (Orange)",
                         "% Coverage", "Min Spot Size", "Max Spot Size", "Mean Spot Size", 
                         "Var Spot Size", "Num Spots"])

for idx,filename in enumerate(os.listdir(img_path)):
#  if idx<1:
    if filename.lower().endswith(format):
        file_path = os.path.join(img_path, filename)
        try:
            with Image.open(file_path) as img:
                print(f"Opened: {filename}, Size: {img.size}, Mode: {img.mode}")
                # You can add more processing here if needed
                
                 # Convert to NumPy array
                img_np = np.array(img)/ 255.0

                # Convert to Lab color space
                img_lab = rgb2lab(img_np)
                
                L = img_lab[..., 0]
                a = img_lab[..., 1]
                b = img_lab[..., 2]


                L_hist, L_bins = np.histogram(L.ravel(), bins=nb, range=(0, 100))
                a_hist, a_bins = np.histogram(a.ravel(), bins=nb, range=(-128, 127))
                b_hist, b_bins = np.histogram(b.ravel(), bins=nb, range=(-128, 127))
                
                # plt.figure(figsize=(12, 4))
                # plt.suptitle(f"Lab Histograms - {filename}", fontsize=14)

                # plt.subplot(1, 3, 1)
                # plt.plot(L_bins[:-1], L_hist, color='gray')
                # plt.title('L channel')
                # plt.xlabel('Lightness')
                # plt.ylabel('Pixel count')

                # plt.subplot(1, 3, 2)
                # plt.plot(a_bins[:-1], a_hist, color='red')
                # plt.title('a channel')
                # plt.xlabel('Green–Red')

                # plt.subplot(1, 3, 3)
                # plt.plot(b_bins[:-1], b_hist, color='blue')
                # plt.title('b channel')
                # plt.xlabel('Blue–Yellow')

                # plt.tight_layout(rect=[0, 0, 1, 0.95])
                # plt.show(block=False)
                
                
                 # Apply multi-Otsu thresholding with 3 classes for each channel
                L_thresh = threshold_multiotsu(L, classes=nb_cls)
                a_thresh = threshold_multiotsu(a, classes=nb_cls)
                b_thresh = threshold_multiotsu(b, classes=nb_cls)
                
                L_seg = np.digitize(L, bins=L_thresh)
                a_seg = np.digitize(a, bins=a_thresh)
                b_seg = np.digitize(b, bins=b_thresh)
                
                
                # Convert segmentation maps to labeled (colored) images.
                # label2rgb converts an array of labels into an RGB image assigning a unique color per label.
                # Here, we do not use a background label, so all regions are colored.
                L_label = label2rgb(L_seg, colors=custom_colors, bg_label=-1)
                a_label = label2rgb(a_seg, colors=custom_colors, bg_label=-1)
                b_label = label2rgb(b_seg, colors=custom_colors, bg_label=-1)
                
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                fig.suptitle(f"Labeled Segmentation (Multi-Otsu 3 Classes) - {filename}", fontsize=14)
                
                # Titles for each channel
                channel_titles = ['L channel', 'a channel', 'b channel']
                thresholds = [L_thresh, a_thresh, b_thresh]
                labeled_images = [L_label, a_label, b_label]
                
                # Legend text and corresponding colors for each class
                legend_entries = [("1 - leaf", custom_colors[0]),
                                  ("2 - Background", custom_colors[1]),
                                  ("3 - halo", custom_colors[2])]
                
                # Loop over each channel subplot to show the image and add legend text.
                for ax, title, thresh, label_img in zip(axes, channel_titles, thresholds, labeled_images):
                    ax.imshow(label_img)
                    ax.set_title(f"{title}\nThresh: {np.around(thresh, 2)}")
                    ax.axis('off')
               
                    y_pos = 0.95
                    dy = 0.05  # vertical spacing
                    for text, color in legend_entries:
                        ax.text(0.02, y_pos, text,
                                transform=ax.transAxes,
                                fontsize=9,
                                color=color,
                                verticalalignment='top',
                                bbox=dict(facecolor='black', alpha=0.4, pad=2, edgecolor='none'))
                        y_pos -= dy
                
                plt.tight_layout(rect=[0, 0, 1, 0.92])
                # plt.show(block=False)
                
                a_seg_mod = a_seg.astype(float)
                seg = a_seg_mod.copy()
                new_label = int(np.nanmax(seg)) + 1
                
                for target in [1, 2]:
                    mask = (seg == target)  # binary mask for current label
                    cc = label(mask)        # label connected components
                    props = regionprops(cc)

                   

                    for prop in props:
                        if prop.area > erode_thresh:  # only remove large components
                            seg[cc == prop.label] = new_label

                            
                            
                seg_int = np.nan_to_num(seg).astype(int)
                a_seg_cleaned = label2rgb(seg_int, colors=custom_colors, bg_label=-1)

                # Plot result
                # plt.figure(figsize=(6, 5))
                # plt.imshow(a_seg_cleaned)
                # plt.title(f"'a' channel - large background/halo removed")
                # plt.axis('off')

                # Optional legend
                legend_entries = [("1 - leaf", custom_colors[0]),                                           
                                ("4 - removed (black)", custom_colors[new_label-1])]

                y_pos = 0.95
                dy = 0.06
                ax = plt.gca()
                for text, color in legend_entries:
                    ax.text(0.02, y_pos, text,
                            transform=ax.transAxes,
                            fontsize=9,
                            color=color,
                            verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.5, pad=2, edgecolor='none'))
                    y_pos -= dy

                # plt.tight_layout()
                # plt.show(block=False) 
        

                leaf_coords = np.column_stack(np.where(seg == 0))# coordinates of the leaf region   
                hull = ConvexHull(leaf_coords)
                delaunay = Delaunay(leaf_coords[hull.vertices])

                # We will now process pixels that are still labeled as 1 or 2
                for target in [1, 2]:
                    target_coords = np.column_stack(np.where(seg == target))
                    if len(target_coords) > 0:
                        is_inside = delaunay.find_simplex(target_coords) >= 0
                        outside_coords = target_coords[~is_inside]  # points outside hull
                        for y, x in outside_coords:
                            seg[y, x] = new_label  # assign to new "outside hull" label
                 
                seg_int = np.nan_to_num(seg).astype(int)
                seg_rgb = label2rgb(seg_int, colors=custom_colors, bg_label=-1)

                # Show plot
                # plt.figure(figsize=(6, 5))
                # plt.imshow(seg_rgb)
                # plt.title(f"'a' channel – final cleaned + convex hull mask")
                # plt.axis('off')

                # legend_entries = [("1 - leaf", custom_colors[0]),
                #                 ("4 - outside convex hull", custom_colors[new_label-1])]

                # y_pos = 0.95
                # dy = 0.06
                # ax = plt.gca()
                # for text, color in legend_entries:
                #     ax.text(0.02, y_pos, text,
                #             transform=ax.transAxes,
                #             fontsize=9,
                #             color=color,
                #             verticalalignment='top',
                #             bbox=dict(facecolor='white', alpha=0.5, pad=2, edgecolor='none'))
                #     y_pos -= dy

                # plt.tight_layout()
                # plt.show(block=False)   
                
                
                leaf_mask = seg == 0
                filled_leaf = binary_fill_holes(leaf_mask)
                filled_leaf_visualization = np.zeros_like(seg, dtype=float)
                filled_leaf_visualization[filled_leaf] = 1  # Green color where the leaf is filled
                # Step 3: Plot the image
                # plt.figure(figsize=(6, 5))
                # plt.imshow(filled_leaf_visualization, cmap='Greens', vmin=0, vmax=1)
                # plt.title('Filled Leaf Mask (Green)')
                # plt.axis('off')
                # plt.show(block=False)
            
                combined_mask = np.logical_and(filled_leaf, leaf_mask)
                combined_visualization = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=float)
                label_image = np.zeros((combined_visualization.shape[0], combined_visualization.shape[1]), dtype=int)
                combined_visualization[filled_leaf] =  [0.9,0.5,0.2]
                combined_visualization[leaf_mask] =  [0,1,0]
                
            
                
                # plt.figure(figsize=(6, 5))
                # plt.imshow(combined_visualization)
                # plt.title('Leaf Mask + spots')
                # plt.axis('off')
                # plt.show(block=False)
                
                green_mask = filled_leaf  # Mask for the green region (leaf)
                orange_mask = np.logical_and(filled_leaf, ~leaf_mask)  # Mask for the orange region (spots)

                green_visualization = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=float)
                green_visualization[green_mask] = [0, 1, 0]  # Green for the leaf
                # Optional: Visualize the masks to verify
                # plt.figure(figsize=(6, 5))
                # plt.imshow(green_visualization)
                # plt.title('Green Mask (Leaf)')
                # plt.axis('off')
                # plt.show(block=False)

                # Visualize the orange mask (spots) with a neutral background
                orange_visualization = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=float)
                orange_visualization[orange_mask] = [0.9, 0.5, 0.2]  # Orange for the spots

                # plt.figure(figsize=(6, 5))
                # plt.imshow(orange_visualization)
                # plt.title('Orange Mask (Spots)')
                # plt.axis('off')
                # plt.show(block=False)
                
               
                # Surface areas
                leaf_area = np.sum(green_mask)
                spot_area = np.sum(orange_mask)

                # Step 4: Calculate the properties of the orange regions (spots)
                regions = regionprops(label(orange_mask)) 
                sizes = [region.area for region in regions]
                min_size = np.min(sizes) if sizes else 0
                max_size = np.max(sizes) if sizes else 0
                mean_size = np.mean(sizes) if sizes else 0
                var_size = np.var(sizes) if sizes else 0
                num_spots = len(regions)
                # Calculate % coverage
                coverage_percentage = (spot_area / leaf_area * 100) if leaf_area > 0 else 0
                
                result_dict = {
                'filename': filename,
                'leaf_area': leaf_area,
                'spot_area': spot_area,
                'coverage_percentage': coverage_percentage,
                'min_size': min_size,
                'max_size': max_size,
                'mean_size': mean_size,
                'var_size': var_size,
                'num_spots': num_spots  # Add number of spots
                }
                with open(output_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([result_dict['filename'], result_dict['leaf_area'], result_dict['spot_area'],
                     result_dict['coverage_percentage'], 
                     result_dict['min_size'], result_dict['max_size'], result_dict['mean_size'],
                     result_dict['var_size'], result_dict['num_spots']])
            
             
        except Exception as e:
            print(f"Failed to open {filename}: {e}")
        
        
        
        