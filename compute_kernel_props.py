import os
import matplotlib.pyplot as plt
from hsi_utils import *
import json
import scipy.io as sio

from skimage.feature import graycomatrix, graycoprops

main_data_folder = 'D:\\HSI data\\VNIR_barley' 
# D:\\HSI data\\Barley_ground_30cm_SWIR
# 'D:/HSI data/VNIR_barley' 

dataset =HsiDataset(main_data_folder,data_ext='ref')
nb_images = len(dataset)
#check if data path exists !
if os.path.isdir(main_data_folder):
    if nb_images>0:
        print(f"dataset  is valid and contains {nb_images} image(s)")
    else:
        print('empty dataset')
else:
    raise FileNotFoundError(f"Invalid path: {main_data_folder}")
HSIreader = HsiReader(dataset)

angles = [0, 45, 90]
features=['ASM', 'Contrast', 'Correlation','entropy'] 
num_features = len(features) * len(angles)       
nb_bins =32


for idx in range(len(dataset)):
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hsi=HSIreader.current_image
    image_name  = HSIreader.current_name
    
    pca_score_image = project_hsi_VNIR(HSIreader, 1000, n_samples=50000, nb_pca_comp=3)
    
    kernel_dir = os.path.join(main_data_folder, 'kernel_data', image_name)
    kernel_file_path = os.path.join(kernel_dir, 'kernel.json')
    
    if os.path.isfile(kernel_file_path):
        with open(kernel_file_path, 'r') as json_file:
            obj_file = json.load(json_file)
                   
        feature_maps = {
            'image_name': image_name,  
            'kernels': []  
        }

        
        for obj in obj_file:
            kernel_id = obj['id']
            coords = obj['pixels']
            bbox = obj['bbox']
            min_row, min_col, max_row, max_col = bbox
            
            # Extract spectral samples and calculate the average spectrum
            spectral_samples = np.array(HSIreader.extract_pixels(coords))
            subimage = HSIreader.extract_subimage(bbox)
            bm = create_binary_mask(bbox, coords)
            avg_spectrum = np.mean(subimage[bm == 1], axis=0)
            
            # Initialize dictionary for the kernel data
            kernel_data = {
                'id': kernel_id,
                'spectral_samples': spectral_samples,
                'avg_spectrum': avg_spectrum,
                'haralick_features': {}
            }
            
            for i in range(pca_score_image.shape[2]):  # For each PCA component
                component_img = pca_score_image[min_row:max_row, min_col:max_col, i]
                iso_glcm = None
                
                normalized_image = ((component_img - component_img.min()) / (component_img.max() - component_img.min())) * (nb_bins-1)
                normalized_image = normalized_image.astype(np.uint8)
                 
                for angle in angles:
                   
                    angle_rad = np.deg2rad(angle)
                
                    # Compute GLCM using the normalized image
                    glcm = graycomatrix(
                        normalized_image,
                        distances=[1],
                        angles=[angle_rad],
                        levels=nb_bins,
                        symmetric=True,
                        normed=True
                )
                    
                    features_values=haralick(glcm, feature_names=features)
                    # Add Haralick features for this angle
                    kernel_data['haralick_features'].setdefault(f'PCA_{i + 1}', {})[f'angle_{angle}'] = features_values
                    if iso_glcm is None:
                        iso_glcm = glcm
                    else:
                        iso_glcm += glcm
                        
                 # Compute isotropic Haralick features
                features_values_iso = haralick(iso_glcm, feature_names=features)
                kernel_data['haralick_features'][f'PCA_{i + 1}']['iso'] = features_values_iso

            # Append kernel data to the 'kernels' list
            feature_maps['kernels'].append(kernel_data)
                 
    mat_file_path = os.path.join(kernel_dir, 'feature_maps.mat')
    npy_file_path = os.path.join(kernel_dir, 'feature_maps.npy')
    sio.savemat(mat_file_path, feature_maps)
    print(f"Feature maps saved as MATLAB file at: {mat_file_path}")

    np.save(npy_file_path, feature_maps)
    print(f"Feature maps saved as NumPy file at: {npy_file_path}")
        
            
            
        
            
            
            
            
        
            
    

   
    
   
    
    
    