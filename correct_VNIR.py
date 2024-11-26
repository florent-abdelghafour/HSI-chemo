import os
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from skimage.measure import label
# from scipy.ndimage import median_filter

"""
    Automatically correct HSI in reflectance from the reference in the image
    Segment image with PCA projection -> get the reference
    Get an avg reference spectrum per column of the image
    all cols will be corrected accordingly
    
    Save the corrected image and copy hdr
    
    For VNIR img, normalize to 16 bits
"""

# Define the path to the main data folder: code will iterate trough relvant files
main_data_folder = "D:/HSI data/VNIR_barley"
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']     

# Initialize the HSI dataset and define file extension: contains all paths of hdr and data files
dataset =HsiDataset(main_data_folder,data_ext='hyspex')
nb_images = len(dataset)
#check if data path exists !
if os.path.isdir(main_data_folder):
    if nb_images>0:
        print(f"dataset  is valid and contains {nb_images} image(s)")
    else:
        print('empty dataset')
else:
    print('path invalid')


# Define the path to save the corrected hyperspectral images
save_folder = main_data_folder + '/ref_corrected'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print(f'corrected images will be saved in : {save_folder}')


# Initialize the HSI reader: class containg info about the image and allow to load and operate on the hsi
HSIreader = HsiReader(dataset)

nb_pca_comp =3
pca = PCA(n_components=nb_pca_comp)
slice_step = 1000
pc_comp_ref=0 ## component that shows best the ref

# # size of median filter, deal with dead pixels and spikes
# filter_size=(7, 7, 1)

#if true correct only foreground, ref = 1 & background = 0
mask_foreground=True

# Loop through each hyperspectral image in the dataset
for idx in range(len(dataset)):
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hsi=HSIreader.current_image
    n_rows, n_cols, n_channels =hsi.shape
    n_samples = 50000
    x_idx = np.random.randint(0, n_cols, size=n_samples)
    y_idx = np.random.randint(0, n_rows, size=n_samples)
    spectral_samples = np.zeros((n_samples, n_channels), dtype=hsi.dtype)
    coords = list(zip(y_idx, x_idx))
    spectral_samples = np.array(HSIreader.extract_pixels(coords))
   
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    
    pca_scores_imgs = []
    for start_row in range(0, n_rows, slice_step):
        end_row = min(start_row + slice_step, n_rows)
        subcube = hsi[start_row:end_row, :, :] 
        
        subcube_flat =subcube.reshape(-1, n_channels)
        pca_scores = np.dot(subcube_flat, pca_loadings)
        pca_scores_img = pca_scores.reshape(end_row - start_row, n_cols, nb_pca_comp)
        pca_scores_imgs.append(pca_scores_img)
        
        del subcube, subcube_flat, pca_scores, pca_scores_img

    final_pca_scores_img = np.concatenate(pca_scores_imgs, axis=0)
    del pca_scores_imgs
     
    score_pc_ref = final_pca_scores_img[:,:,pc_comp_ref]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=3)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    labeled_image = label(segmented)
    binary_image = labeled_image > 0 #label 0 = background 
    labeled_image=label(binary_image)    
    # #get the reference object compute row average -> 1 ref spectrum per column
    reference_mask = labeled_image == 1
    background_mask = labeled_image == 0
    foreground_mask = np.logical_and(np.logical_not(background_mask), np.logical_not(reference_mask))
    
    # #check if reference is truly extracted
    # plt.figure()
    # plt.imshow(reference_mask)
    # plt.show()
    
    coordinates = np.argwhere(reference_mask)
    y_min, x_min = coordinates.min(axis=0)  
    y_max, x_max = coordinates.max(axis=0)  
    bbox_spectralon = (x_min, y_min, x_max, y_max)
    spectralon = hsi[y_min:y_max, :, :]
    # spectralon = median_filter(spectralon, size=filter_size)
    avg_spectralon = np.mean(spectralon, axis=0).astype(np.uint16)
    
    
    # plt.figure()
    # for i in range (avg_spectralon.shape[0]):
    #     plt.plot(wv,avg_spectralon[i,:])
    # plt.show()

    foreground_mask_3d = np.repeat(foreground_mask[:, :, np.newaxis], hsi.shape[2], axis=2)
    
    hypercube_slices = []
    for start_row in range(0, n_rows, slice_step):  
        end_row = min(start_row + slice_step, n_rows)
        subcube = (hsi[start_row:end_row, :, :]).astype(np.float32)
        # subcube = median_filter(subcube, size=filter_size)
         
         # adapt the casting in rows  to slice size , in order to have col-wise divide
        spectralon_size = min(slice_step, subcube.shape[0])
        avg_spectralon_expanded = np.repeat(avg_spectralon[np.newaxis, :, :], spectralon_size, axis=0)
        
        if mask_foreground == True:
            slice_foreground_mask = foreground_mask_3d[start_row:end_row,:]
            #correct only foregound
            subcube[slice_foreground_mask] /= avg_spectralon_expanded[slice_foreground_mask]
            
            slice_background_mask = background_mask[start_row:end_row,:]
            subcube[slice_background_mask,:] = np.zeros(n_channels, dtype=hsi.dtype)
            slice_reference_mask = reference_mask[start_row:end_row,:]
            subcube[slice_reference_mask,:] = np.ones(n_channels, dtype=hsi.dtype)
        else:
            subcube /= avg_spectralon_expanded
        
        subcube = np.clip(subcube * 65535, 0, 65535).astype(np.uint16)
        hypercube_slices.append(subcube)
        del subcube,slice_foreground_mask,slice_background_mask,slice_reference_mask
        
    hypercube_slices = np.concatenate(hypercube_slices, axis=0)
 
    # save new corrected image in new folder with corresponding header
    base_filename = os.path.splitext(os.path.basename(HSIreader.dataset[idx]['data']))[0]
    save_path = os.path.join(save_folder, f"{base_filename}_ref.hdr")
    header_path = HSIreader.dataset[idx]['hdr']
    header = envi.read_envi_header(header_path)
    
    envi.save_image(save_path, hypercube_slices,ext='ref', dtype='uint16', force=True, metadata=header) 

    HSIreader.clear_cache()
    print(f"img_{idx} corrected and saved : {save_path} ")
   
    
    
    