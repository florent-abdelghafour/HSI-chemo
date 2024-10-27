import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.measure import label

"""
    Automatically correct HSI in reflectance from the reference in the image
    Segment image with PCA projection -> get the reference
    Get an avg reference spectrum per column of the image
    all cols will be corrected accordingly
    
    Save the corrected image and copy hdr
    
    For VNIR img, normalize to 16 bits
"""



# Define the path to the main data folder: code will iterate trough relvant files
main_data_folder = "D:\\VNIR_barley"     

# Initialize the HSI dataset and define file extension: contains all paths of hdr and data files
dataset =HsiDataset(main_data_folder,data_ext='hyspex')
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Define the path to save the corrected hyperspectral images
save_folder = os.path.join(main_data_folder, 'ref_corrected')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize the HSI reader: class containg info about the image and allow to load and operate on the hsi
HSIreader = HsiReader(dataset)


nb_pca_comp =3
pca = PCA(n_components=nb_pca_comp)
slice_step = 1000
min_kernel_size=500

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
    final_pca_scores_img = np.concatenate(pca_scores_imgs, axis=0)
     
    score_pc_ref = final_pca_scores_img[:,:,0]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=3)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    labeled_image = label(segmented)
    binary_image = labeled_image > 0
    filled_binary_image = binary_fill_holes(binary_image)
    labeled_image = label(filled_binary_image)
    labeled_image= label(remove_small_objects(labeled_image > 0, min_size=min_kernel_size))
    
    # color_image = color_labels(labeled_image)
        
    # plt.figure()
    # plt.imshow(color_image)
    # plt.title('Color-Mapped Labeled Image')
    # plt.axis('off')
    # plt.show()
    
    
    #get the reference object compute row average -> 1 ref spectrum per column
    reference_mask = labeled_image == 1
    coordinates = np.argwhere(reference_mask)
    y_min, x_min = coordinates.min(axis=0)  
    y_max, x_max = coordinates.max(axis=0)  
    bbox_spectralon = (x_min, y_min, x_max, y_max)
    spectralon = hsi[y_min:y_max, :, :]
    avg_spectralon = np.sum(spectralon, axis=0)
    
    reference_mask=np.repeat(reference_mask[:, :, np.newaxis], hsi.shape[2], axis=2)
    num_valid_pixels = np.sum(reference_mask, axis=0)
    avg_spectralon=avg_spectralon / num_valid_pixels    
    avg_spectralon[num_valid_pixels == 0]  = np.nan 
    
    hypercube_slices = []
    for start_row in range(0, n_rows, slice_step):
       
        end_row = min(start_row + slice_step, n_rows)
        subcube = hsi[start_row:end_row, :, :]
        subcube =subcube/ avg_spectralon[np.newaxis, :, :]     
        subcube_scaled = (subcube * 65536).astype(np.uint16) 
        hypercube_slices.append(subcube_scaled)
        
    corrected_hypercube = np.concatenate(hypercube_slices, axis=0)
 
   
    # save new corrected image in new folder with corresponding header
    base_filename = os.path.splitext(os.path.basename(HSIreader.dataset[idx]['data']))[0]
    save_path = os.path.join(save_folder, f"{base_filename}_ref.hdr")
    header_path = HSIreader.dataset[idx]['hdr']
    header = envi.read_envi_header(header_path)
    
    envi.save_image(save_path, corrected_hypercube,ext='ref', dtype='uint16', force=True, metadata=header) 


    HSIreader.clear_cache()
    print(f"img_{idx} corrected and saved : {save_path} ")
   
    
    
    