import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_otsu

def safe_divide(numerator, denominator):
    """Divide safely, replacing zero denominators with one."""
    return np.divide(numerator, np.where(denominator == 0, 1, denominator), out=np.zeros_like(numerator, dtype=float))


def create_transforms():
    """Create a dictionary of transformations based on common formulas."""
    color_channels = ['r', 'g', 'b']
    transforms = {}

    # Iterate over the color channels for the numerator (x), denominator (y, z)
    for x in color_channels:
        for y in color_channels:
            for z in color_channels:
                if x != y and x != z and y != z:  # Ensure x, y, z are distinct
                    # Define the transformation with fixed 'x', 'y', 'z' in the closure
                    transforms[f'{x}/({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(x), eval(f'({y}+{z})'))
                    transforms[f'{x}/({y}**2+{z}**2)'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(x), eval(f'({y}**2+{z}**2)'))
                    transforms[f'{x}/({y}**3+{z}**3)'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(x), eval(f'({y}**3+{z}**3)'))
                    transforms[f'{x}**2/({y}+{z})'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(f'{x}**2'), eval(f'({y}+{z})'))
                    transforms[f'{x}**2/({y}**2+{z}**2)'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(f'{x}**2'), eval(f'({y}**2+{z}**2)'))
                    transforms[f'{x}**2/({y}**3+{z}**3)'] = lambda r, g, b, x=x, y=y, z=z: safe_divide(eval(f'{x}**2'), eval(f'({y}**3+{z}**3)'))
    return transforms


def normalize_bands(transformed_image):
    """Normalize the image to the range [0, 255]."""
    # Normalize to [0, 1]
    transformed_image = (transformed_image - np.min(transformed_image)) / (np.max(transformed_image) - np.min(transformed_image))
    # Scale to [0, 255]
    transformed_image *= 255
    # Ensure values are within [0, 255]
    transformed_image = np.clip(transformed_image, 0, 255)
    # Cast to integers
    return transformed_image.astype(np.uint8)
# Generate the transformations

transforms = create_transforms()

data_root= "D:\\Piment7RGB\\Piment7RGB.mat"
output_base_dir = "D:\\Piment7RGB\\index_maps\\"
formula_txt_path = "D:/Piment7RGB/formula_mapping.txt"


dataset = scipy.io.loadmat(data_root)
list_img = [key for key in dataset.keys() if not key.startswith('__')]

organized_data = {
    "recto": {key: dataset[key] for key in list_img if key.endswith('r')},
    "verso": {key: dataset[key] for key in list_img if key.endswith('v')}
}

del dataset


indexes_dict = {transform: [] for transform in transforms.keys()}

with open(formula_txt_path, "w") as formula_file:
    for formula_idx, (formula, transform) in enumerate(transforms.items()):
        # Directory for this formula index
        formula_folder = os.path.join(output_base_dir, f"formula_{formula_idx}")
        # Save the formula mapping in the text file
        formula_file.write(f"{formula_idx}: {formula}\n")

for key, img in organized_data["recto"].items():
    print(f"Key: {key}")
    
    r = img[:, :, 0]  # Red channel
    g = img[:, :, 1]  # Green channel
    b = img[:, :, 2]  # Blue channel
    
    grayscale_img = 0.2989 * r + 0.5870 * g + 0.1140 * b 
    otsu_threshold = threshold_otsu(grayscale_img)
    binary_mask = (grayscale_img > otsu_threshold).astype(np.uint8)
    mask_folder = os.path.join(os.path.dirname(data_root), 'recto','binary_masks')
    os.makedirs(mask_folder, exist_ok=True)
    mask_save_path = os.path.join(mask_folder, f"{key}_mask.mat")
    scipy.io.savemat(mask_save_path, {'binary_mask': binary_mask})
     
    r_masked = r * binary_mask
    g_masked = g * binary_mask
    b_masked = b * binary_mask
    
    
    for formula_idx, (formula, transform) in enumerate(transforms.items()):
        transformed_image = transform(r_masked, g_masked, b_masked)
        transformed_image=normalize_bands(transformed_image)
        
        transformed_image[binary_mask == 0] = np.nan
        cmap = plt.cm.magma  # Base colormap
        cmap.set_bad(color='black')  # Define black for NaN values

        # Plot and save the transformed image
        plt.imshow(transformed_image, cmap=cmap)
        plt.axis('off')  # Hide axes
        formula_folder = os.path.join(os.path.dirname(data_root), 'recto', f"formula_{formula_idx}")
        os.makedirs(formula_folder, exist_ok=True)
        if not os.path.exists(formula_folder):
            os.makedirs(formula_folder)
        save_path = os.path.join(formula_folder, f"{key}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
        
        
        
        h,w = np.shape(transformed_image)
        index = np.sum(transformed_image) /(h*w)
        indexes_dict[formula].append(index)
        
     


fig, ax = plt.subplots(figsize=(10, 6))

# Iterate through the transforms dictionary and plot the means
for transform_name, index in indexes_dict.items():
    ax.plot(index, label=transform_name)

# Set the x-ticks based on the length of the recto images
x_ticks = range(len(organized_data["recto"].items()))
ax.set_xticks(x_ticks)

# Label the x-axis with the names of the images or just indices
ax.set_xticklabels([f"T {i+1}" for i in x_ticks], rotation=45, ha="right")

# Adding a title and labels to axes
ax.set_title("Spectral index")
ax.set_xlabel("Time-point Image")
ax.set_ylabel("Mean spectrall index")

# Make the legend more readable
ax.legend(title="Transforms", bbox_to_anchor=(1.05, 1), loc='upper left')

# Optional: add grid for better readability
ax.grid(True)
# Adjust layout to make room for the legend
plt.tight_layout()
# Show the plot
plt.show()

