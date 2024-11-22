import os
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from skimage.measure import label
from scipy.ndimage import median_filter



"""
    Automatically correct HSI in reflectance from the reference in the image
    Segment image with PCA projection -> get the reference
    Get an avg reference spectrum per column of the image
    all cols will be corrected accordingly
    
    Save the corrected image and copy hdr
    
    For VNIR img, normalize to 16 bits
"""



# Define the path to the main data folder: code will iterate trough relvant files
main_data_folder = 'D:/HSI data/Barley_ground_30cm_SWIR'
# D:/HSI data/Barley_ground_30cm_SWIR
# 'D:/VNIR_barley' 

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
save_folder = main_data_folder   +  '/ref_corrected'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
print(f'corrected images will be saved in : {save_folder}')

# Initialize the HSI reader: class containg info about the image and allow to load and operate on the hsi
HSIreader = HsiReader(dataset)

# Principal component corresponding to best segmentaion of reference and background
pc_comp_ref=0
# size of median filter, deal with dead pixels and spikes
filter_size=(7, 7, 1)

# Loop through each hyperspectral image in the dataset
for idx in range(len(dataset)):  
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    # Get the hyperspectral data
    hypercube= HSIreader.get_hsi()
    hypercube = median_filter(hypercube, size=filter_size)
    
    
##############################################################################################
###            Sample spectral data and compute PCA loadings : project on hypercube        ###
############################################################################################## 
    
    #Sample some spectra to determine generic pca laodings
    n_samples = 50000
    x_idx = np.random.randint(0, hypercube.shape[0], size=n_samples)
    y_idx = np.random.randint(0, hypercube.shape[1], size=n_samples)
    
    spectral_samples = hypercube[x_idx, y_idx, :]
    
    nb_pca_comp =3
    pca = PCA(n_components=nb_pca_comp)
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    #project back the laodings on the entire hsi to get scores
    score_img = HSIreader.project_pca_scores(pca_loadings)
    score_pc_ref = score_img[:,:,pc_comp_ref]   
    # Threshold score images  to get object classes
    thresholds = threshold_multiotsu(score_pc_ref, classes=3)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    labeled_image = label(segmented)
    del pca_scores
   
    # plt.figure()
    # plt.imshow(labeled_image)
    # plt.show(block=False)

##############################################################################################
###            Extract binary masks of reference, background and foreground                ###
##############################################################################################   
    
    #fill holes in small object
    binary_image = labeled_image > 0
    binary_image = labeled_image > 0 #label 0 = background 
    labeled_image=label(binary_image)   
    
    #get the reference object compute row average -> 1 ref spectrum per column
    reference_mask = labeled_image == 1
    # # #check if reference is truly extracted
    # plt.figure()
    # plt.imshow(reference_mask)
    # plt.show()
    #check if reference  (label = 1) is value 1 i.e in yellow and rest is value O i.e. in dark blue
    
    background_mask = labeled_image == 0
    #check if Background (label =0) is value 1 i.e in yellow and rest is value O i.e. in dark blue
    # plt.figure()
    # plt.imshow(background_mask)
    # plt.show()
    
    # Foreground is not reference not background
    foreground_mask = np.logical_and(np.logical_not(background_mask), np.logical_not(reference_mask))
    
    # plt.figure()
    # plt.imshow(foreground_mask)
    # plt.show()
    
##############################################################################################
##############################################################################################

##############################################################################################
###               Extract Data of the reference (spectralon) per column                    ###
##############################################################################################
    reference_mask_3d=np.repeat(reference_mask[:, :, np.newaxis], hypercube.shape[2], axis=2)
    foreground_mask = np.repeat(foreground_mask[:, :, np.newaxis], hypercube.shape[2], axis=2)

    spectralon = np.where(reference_mask_3d, hypercube, 0)
    subimage_mask = np.any(reference_mask_3d, axis=2)  # Collapse bands to get 2D spatial mask
    row_indices, col_indices = np.where(subimage_mask)
    # Get the bounding box of the subimage
    row_min, row_max = row_indices.min(), row_indices.max() + 1
    col_min, col_max = col_indices.min(), col_indices.max() + 1
    # Extract the subimage of the spectralon
    subimage_spectralon = spectralon[row_min:row_max, col_min:col_max, :]
    avg_spectralon = np.mean(subimage_spectralon, axis=0)
    
    
    #Check values of spectralon columns
    # plt.figure()
    # for i in range (avg_spectralon.shape[0]):
    #     plt.plot(wv,avg_spectralon[i,:])
    # plt.show()
    
##############################################################################################
##############################################################################################    
    
    #correct image from spectralon,  per column
    avg_spectralon_expanded = np.repeat(avg_spectralon[np.newaxis, :, :], hypercube.shape[0], axis=0)
    hypercube[foreground_mask] /= avg_spectralon_expanded[foreground_mask]
    
    #replace background by spectral zeros and ref by spectral ones
    hypercube[background_mask,:] = np.zeros(hypercube.shape[2], dtype=hypercube.dtype)
    hypercube[reference_mask,:] = np.ones(hypercube.shape[2], dtype=hypercube.dtype)
 
    hypercube = (65535*(hypercube)).astype(np.uint16) 
    
    # save new corrected image in new folder with corresponding header
    base_filename = os.path.splitext(os.path.basename(HSIreader.dataset[idx]['data']))[0]
    save_path = save_folder +f"/{base_filename}_ref.hdr"
    header_path = HSIreader.dataset[idx]['hdr']
    header = envi.read_envi_header(header_path)
    
    envi.save_image(save_path, hypercube,ext='ref', dtype='uint16', force=True, metadata=header) 

    del hypercube
    HSIreader.clear_cache()
    
    
