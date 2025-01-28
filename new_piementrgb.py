import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu

def safe_divide(numerator, denominator):
    """Divide safely, replacing zero denominators with one."""
    return np.divide(numerator, np.where(denominator == 0, 1, denominator), out=np.zeros_like(numerator, dtype=float))

def safe_log(x, epsilon=1e-10):
    """Safely compute log2, avoiding log(0) or log(negative) by adding a small epsilon."""
    return np.log2(np.maximum(x, epsilon))

def create_transforms():
    """Create a dictionary of transformations based on common formulas."""
    color_channels = ['r', 'g', 'b']
    transforms = {}

    # Iterate over the color channels for the numerator (x), denominator (y, z)
    for x in color_channels:
        for y in color_channels:
            for z in color_channels:
                if x != y and x != z and y != z:  # Ensure x, y, z are distinct
                    transforms[f'{x}/({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(x), eval(f'({y}+{z})'))
                    transforms[f'{x}**2/({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(f'{x}**2'), eval(f'({y}+{z})'))
                    transforms[f'{x}**3/(({y}+{z})**2)'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(f'{x}**3'), eval(f'(({y}+{z})**2)'))
                    transforms[f'log2({x})/log2({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(safe_log(eval(x)), safe_log(eval(f'({y}+{z})')))
                    transforms[f'log2({x}**2)/log2({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(safe_log(eval(f'{x}**2')) , safe_log(eval(f'({y}+{z})')))
                    transforms[f'{x}/sqrt({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(x), np.sqrt(eval(f'({y}+{z})')))
                    transforms[f'{x}**2/sqrt({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(f'{x}**2'), np.sqrt(eval(f'({y}+{z})')))
                    transforms[f'{x}**3/sqrt({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(f'{x}**3'), np.sqrt(eval(f'({y}+{z})')))
                    
    return transforms

def normalize_within_mask(image, mask):
    """
    Normalize the image values within the mask to the range [0, 1], ignoring background values.
    """
    valid_values = image[mask > 0]  

    min_val = np.min(valid_values)
    max_val = np.max(valid_values)
    normalized_image = (image - min_val) / (max_val - min_val)
    normalized_image = normalized_image*255
    # Preserve NaN values
    normalized_image[~np.isfinite(image)] = np.nan

    return normalized_image

# Create transforms
transforms = create_transforms()

data_root = "D:\\Piment7RGB\\Piment7RGB.mat"
output_base_dir = "D:\\Piment7RGB\\index_maps\\"
formula_txt_path = "D:\\Piment7RGB\\formula_mapping.txt"

dataset = scipy.io.loadmat(data_root)
list_img = [key for key in dataset.keys() if not key.startswith('__')]

# Organized data
organized_data = {
    "recto": {key: dataset[key] for key in list_img if key.endswith('r')},
    "verso": {key: dataset[key] for key in list_img if key.endswith('v')}
}
del dataset
# Define variable for processing recto or verso
side_to_process = "verso"  


# Index dictionary for storing results
indexes_dict = {transform: [] for transform in transforms.keys()}

# Writing the formula mapping to a text file
with open(formula_txt_path, "w") as formula_file:
    for formula_idx, (formula, transform) in enumerate(transforms.items()):
        # Directory for this formula index
        formula_folder = os.path.join(output_base_dir, f"formula_{formula_idx}")
        # Save the formula mapping in the text file
        formula_file.write(f"{formula_idx}: {formula}\n")

# Loop through the selected side (recto or verso)
for key, img in organized_data[side_to_process].items():
    print(f"Image: {key}")
    
    r = img[:, :, 0]  # Red channel
    g = img[:, :, 1]  # Green channel
    b = img[:, :, 2]  # Blue channel
    
    # Convert to grayscale using a weighted sum of RGB channels
    grayscale_img = 0.2989 * r + 0.5870 * g + 0.1140 * b 
    otsu_threshold = threshold_otsu(grayscale_img)
    binary_mask = (grayscale_img > otsu_threshold).astype(np.uint8)
    
    # Create and save the mask
    mask_folder = os.path.join(os.path.dirname(data_root), side_to_process, 'binary_masks')
    os.makedirs(mask_folder, exist_ok=True)
    mask_save_path = os.path.join(mask_folder, f"{key}_mask.mat")
    scipy.io.savemat(mask_save_path, {'binary_mask': binary_mask})
     
    # Apply mask to the color channels
    r_masked = r * binary_mask
    g_masked = g * binary_mask
    b_masked = b * binary_mask
    n_samples = np.sum(binary_mask)
    
    # Apply transformations and save the results
    for formula_idx, (formula, transform) in enumerate(transforms.items()):
        transformed_image = transform(r_masked, g_masked, b_masked)
        transformed_image = normalize_within_mask(transformed_image, binary_mask)
        transformed_image[binary_mask == 0] = np.nan
        cmap = plt.cm.coolwarm  # Base colormap
        cmap.set_bad(color='black')  # Define black for NaN values

        # Plot and save the transformed image
        plt.imshow(transformed_image, cmap=cmap)
        plt.axis('off')  # Hide axes
        plt.colorbar()
        formula_folder = os.path.join(os.path.dirname(data_root), side_to_process, f"formula_{formula_idx}")
        os.makedirs(formula_folder, exist_ok=True)
        save_path = os.path.join(formula_folder, f"{key}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
        # Compute and store the index
        index = np.nansum(transformed_image) / n_samples
        indexes_dict[formula].append(index)

# Plotting the spectral index results
fig, ax = plt.subplots(figsize=(18, 9))

# Iterate through the transforms dictionary and plot the means
for transform_name, index in indexes_dict.items():
    ax.plot(index, label=transform_name)

# Set the x-ticks based on the length of the selected side's images
x_ticks = range(len(organized_data[side_to_process].items()))
ax.set_xticks(x_ticks)

# Label the x-axis with the names of the images or just indices
ax.set_xticklabels([f"T {i+1}" for i in x_ticks], rotation=45, ha="right")

# Adding a title and labels to axes
ax.set_title("Spectral index")
ax.set_xlabel("Time-point Image")
ax.set_ylabel("Mean spectral index")

# Make the legend more readable
ax.legend(title="Transforms", loc='upper left', bbox_to_anchor=(1.05, 1), ncol=2, frameon=False)

# Optional: add grid for better readability
ax.grid(True)

# Adjust layout to make room for the legend
plt.tight_layout()
save_path = os.path.join(os.path.dirname(data_root),side_to_process,  f"spectral_index.png")
plt.savefig(save_path, bbox_inches='tight')

# Show the plot
plt.show()
