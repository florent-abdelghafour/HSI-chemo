import os
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt

class HsiDataset:
    def __init__(self, main_dir, data_ext="dat", hdr_ext="hdr"):
        self.main_dir = main_dir
        self.data_ext = data_ext
        self.hdr_ext = hdr_ext
        
        self.data = []
        self.hdr = []

        self._collect_paths()
        
        
    def _collect_paths(self):
    # Walk through the directory and collect paths for raw data and header files
        for root, _, files in os.walk(self.main_dir):
            for file in files:
                if file.endswith(self.data_ext):
                    data_path = os.path.join(root, file)
                    hdr_path = self._get_corresponding_hdr(data_path)
                    
                    # Check if the corresponding header file exists
                    if os.path.exists(hdr_path):
                        self.data.append(data_path)
                        self.hdr.append(hdr_path)
                        
                    else:
                        # Skip the data file if no corresponding hdr file is found
                        print(f"Warning: No header file found for {data_path}, skipping this file.")
                        
    def _get_corresponding_hdr(self, data_path):
        # Get the header path by replacing the data file extension with the header extension
        base_path, _ = os.path.splitext(data_path)
        return f"{base_path}.{self.hdr_ext}"
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        # Allows indexing into the dataset to get a data-header pair
        if idx >= len(self.data):
            raise IndexError("Index out of range")
        return {"data": self.data[idx], "hdr": self.hdr[idx]}
   
    def get_all_paths(self):
        # Returns a list of all data-header pairs
        return [{"data": d, "hdr": h} for d, h in zip(self.data, self.hdr)]
    
    
    
class HsiReader:
    def __init__(self, dataset):
        if not isinstance(dataset, HsiDataset):
            raise TypeError("dataset must be an instance of HsiDataset")
        self.dataset = dataset
        self.current_image = None
        self.current_metadata = None
        self.current_idx = None
        self.hypercube = None  

        
    def read_image(self, idx):
        """
        Reads the image metadata for a specific index on-demand.
        Stores the metadata and file paths temporarily in memory.
        """
        if idx >= len(self.dataset):
            raise IndexError("Index out of range")

        data_path = self.dataset[idx]["data"]
        hdr_path = self.dataset[idx]["hdr"]

        try:
            # Read the image and metadata using the spectral library
            img = envi.open(hdr_path, data_path)
            metadata=img.metadata
            metadata={k.lower(): v for k, v in metadata.items()} 
            self.current_metadata = metadata  # Store the metadata
            self.current_image = img  # Store the image object for later loading
            self.current_idx = idx  # Track which image is currently read

        except Exception as e:
            print(f"Error reading {data_path} and {hdr_path}: {e}")
            self.current_metadata = None
            self.current_image = None
            self.current_idx = None
            
            
            
    def extract_subimage(self, bbox):
        if not self.current_image:
            raise ValueError("No current image loaded. Call `read_image` first.")
        
        min_row, min_col, max_row, max_col = bbox
        bands = list(range(self.current_image.shape[2]))
        try:
            # Check if 'bands' is None or not
            if bands is None:
                bands = list(range(self.current_image.shape[2]))  # Assuming last dimension is bands
            
            # Read the specific sub-cube from the image
            subcube = self.current_image.read_subregion(
                (min_row, max_row),
                (min_col, max_col),
                bands=bands
            )
            return subcube
        

        except Exception as e:
            print(f"Error reading subimage: {e}")
            return None
        
    def extract_pixels(self, coords):
        """
        Reads the spectral values at specified pixel coordinates.
        Args:
            coords (list of tuples): A list of (row, col) tuples representing pixel coordinates.
        Returns:
            list: A list of spectra corresponding to the provided coordinates.
        """
        if self.current_image is None:
            print("No image loaded. Please call `read_image` first.")
            return None
        
        spectra = []
        
        for row, col in coords:
            try:
                # Extract the spectral data for the pixel at (row, col)
                spectrum = self.current_image.read_pixel(row, col)
                spectra.append(spectrum)
            except Exception as e:
                print(f"Error reading pixel at ({row}, {col}): {e}")
        
        return spectra
        
        
        
          
    def get_hsi(self, idx=None):
        """
        Loads the image data for the specified index if provided.
        If no index is provided, uses the current index, defaulting to not load anything if none is read.
        """
        if idx is not None:
            self.read_image(idx)

        # Ensure the hypercube is loaded
        if self.hypercube is None:
            if self.current_image is not None:
                try:
                    self.hypercube = self.current_image.load()
                except Exception as e:
                    print(f"Error loading image data: {e}")
                    return None
            else:
                print("No image has been read yet and no index provided. Doing nothing.")
                return None
        
        return self.hypercube
      
             
    def get_rgb(self, idx=None):
        if idx is not None:
            self.read_image(idx)
            
        if self.hypercube is None:
            self.get_hsi()
            
        if self.hypercube is None or self.current_metadata is None:
            print("No image or metadata available. Ensure `read_image` has been called.")
            return None
        
        try:
            rgb_channels = [int(i) for i in self.current_metadata.get('default bands', [])]
            print(f"RGB channels: {rgb_channels}")

            # Construct the RGB image
            rgb_image = np.asarray(self.hypercube[:, :, rgb_channels]).astype(np.float32)
            
            # Normalize the RGB image
            for ch in range(3):
                channel = rgb_image[:, :, ch]
                channel_min, channel_max = channel.min(), channel.max()
                if channel_max > channel_min:
                    rgb_image[:, :, ch] = (channel - channel_min) / (channel_max - channel_min) * 255
                else:
                    rgb_image[:, :, ch] = 0
            
            return rgb_image.astype(np.uint8)
        
        except KeyError:
            print("No 'default bands' key found in metadata.")
            return None
        
        
      
    def get_wavelength(self, idx=None):
        """
        Retrieves the wavelength information from the current image's metadata.
        If an index is provided, it will read the image at that index.
        """
        if idx is not None:
            self.read_image(idx)
        
        if self.current_metadata is None:
            print("No metadata available")
            return None
        
        wv = np.array([np.float32(i) for i in self.current_metadata.get('wavelength', [])])
        return wv    
         
    def clear_cache(self):
        """
        Clears the current image and metadata from memory.
        """
        self.current_image = None
        self.current_metadata = None
        self.current_idx = None
        self.hypercube = None
        
    def get_spectrum(self, idx=None):
        """
        Retrieves spectral information for the image at the specified index.
        If no index is provided, uses the current index.
        """
        if idx is not None:
            self.read_image(idx)
                  
        if self.hypercube is None:
            if self.current_image is None:
                print("No image or hypercube available. Ensure `read_image` has been called.")
                return None
            else:
                try:
                    self.hypercube = self.current_image.load()
                except Exception as e:
                    print(f"Error loading image data: {e}")
                    return None
                
        if self.current_metadata is None or self.hypercube is None:
            print("No image or metadata available.")
            return None

        try:
            rgb_image = self.get_rgb()
            if rgb_image is None:
                print("Failed to get RGB image.")
                return None
            wv =self.get_wavelength()

            
            all_spec = []
            all_pos = []
            color_list = plt.cm.tab20(np.linspace(0, 1, 20))
         
            legends = []
          
            fig2, ax2 = plt.subplots(constrained_layout=True)
            ax2.imshow(rgb_image)
            plt.title("Click on the image to sample spectra. Press Enter to finish.")
            plt.draw()
            
            pts = plt.ginput(-1, timeout=-1)
            
            fig, ax = plt.subplots()
            for idx, pos in enumerate(pts):
                x, y = int(pos[0]), int(pos[1])
                
                color = color_list[idx % len(color_list)]
                
                ax2.plot(x, y, 'x', color=color, markersize=8)
                ax2.text(x, y - 50, f'({x}, {y})', color=color, fontsize=10, verticalalignment='bottom', horizontalalignment='center')
                
                legend_marker = ax.scatter( [], [], marker='x', s=50, color=color, label=f'x={x}, y={y}')
                legends.append(legend_marker)
                
                spectrum = self.hypercube[y, x, :].reshape(-1)
                all_spec.append(spectrum)
                all_pos.append((x, y))
            
                ax.plot(wv, spectrum, label=f'({x}, {y})', color=color)
                ax.legend(handles=legends, loc='center left',
                  bbox_to_anchor=(1, 0.5), title='samples')
                
            ax.legend(handles=legends, loc='center left', bbox_to_anchor=(1, 0.5), title='Samples')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity")
            plt.grid()
            plt.show()
            
         
            
            return all_spec, all_pos
        
        except Exception as e:
            print(f"Error processing spectrum: {e}")
            return None, None
            
            
    def project_pca_scores(self, loadings, mask=None):
        """
        Compute PCA scores for the entire hyperspectral image using precomputed PCA loadings.
        
        Parameters:
        - loadings (np.ndarray): Precomputed PCA loadings (n_bands, n_components).
        - mask (np.ndarray): Optional 2D boolean mask (height, width), where True indicates the pixel to use.
        
        Returns:
        - pca_scores_img (np.ndarray): The PCA scores reshaped to the image's spatial dimensions (height, width, n_components).
        """
        # Ensure the HSI data is loaded
        if self.hypercube is None:
            if self.current_image is None:
                print("No image has been read. Please call `read_image()` first.")
                return None
            else:
                self.hypercube = self.current_image.load()

        # Get the shape of the hypercube
        height, width, n_bands = self.hypercube.shape
        
        # Ensure loadings have the correct shape
        if loadings.shape[0] != n_bands:
            raise ValueError(f"Loadings should have shape (n_bands, n_components), but got {loadings.shape}.")
        
        # Apply the mask if provided
        if mask is not None:
            # Ensure the mask is the same size as the image
            if mask.shape != (height, width):
                raise ValueError("Mask shape must match the spatial dimensions of the hypercube.")
            
            # Flatten the hypercube only at mask locations
            flattened_hypercube = self.hypercube[mask].reshape(-1, n_bands)
        else:
            # Flatten the entire hypercube
            flattened_hypercube = self.hypercube.reshape(-1, n_bands)
        
        # Project the spectral data onto the PCA loadings (principal components)
        pca_scores = np.dot(flattened_hypercube, loadings)
        
        # If a mask was applied, we need to re-insert the PCA scores into the full image shape
        if mask is not None:
            # Create an empty array to hold the full PCA score image
            pca_scores_img = np.zeros((height, width, loadings.shape[1]))
            
            # Insert the PCA scores only at the mask locations
            pca_scores_img[mask] = pca_scores
        else:
            # Reshape the PCA scores back to the original image's spatial dimensions
            pca_scores_img = pca_scores.reshape(height, width, loadings.shape[1])
        
        return pca_scores_img
    

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

from matplotlib.colors import hsv_to_rgb
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