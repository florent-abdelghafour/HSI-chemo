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
main_data_folder = './data/img_test'    

# Initialize the HSI dataset and define file extension: contains all paths of hdr and data files
dataset =HsiDataset(main_data_folder,data_ext='hyspex')

# Define the path to save the corrected hyperspectral images
save_folder = './data/img_test'   +  '/ref_corrected'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize the HSI reader: class containg info about the image and allow to load and operate on the hsi
HSIreader = HsiReader(dataset)

# Loop through each hyperspectral image in the dataset
for idx in range(len(dataset)):
   if idx==0: 
    
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    # Get the hyperspectral data
    hypercube= HSIreader.get_hsi()
    
    #Sample some spectra to determine generic pca laodings
    n_samples = 50000
    x_idx = np.random.randint(0, hypercube.shape[0], size=n_samples)
    y_idx = np.random.randint(0, hypercube.shape[1], size=n_samples)
    
    spectral_samples = hypercube[x_idx, y_idx, :]
    
    nb_pca_comp =3
    pca = PCA(n_components=nb_pca_comp)
    pca_scores = pca.fit_transform(spectral_samples)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    
    # default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # for i in range(np.shape(pca_loadings)[1]):
    #     plt.figure()
    #     plt.plot(wv,pca_loadings[:,i],default_colors[i])
    #     plt.xlabel("Wavelength (nm)")
    #     plt.ylabel("Reflectance")  
    #     lab= 'PC'+str(i+1)
    #     plt.title(lab) 
    #     plt.grid()  
    # plt.show(block=False)
    
   
    #project back the laodings on the entire hsi to get scores
    score_img = HSIreader.project_pca_scores(pca_loadings)
   
    # for s in range(pca_loadings.shape[1]):
    #     plt.figure()
    #     plt.imshow(score_img[:,:,s])
    #     plt.title(f'Score image PC{s+1}')
    #     plt.axis('off')
    #     plt.show(block=False)
    
    # automatic thresholding with Ostu method (histogram based)
    score_pc_ref = score_img[:,:,0]   
    thresholds = threshold_multiotsu(score_pc_ref, classes=3)
    segmented = np.digitize(score_pc_ref, bins=thresholds)
    
    # plt.figure()
    # plt.imshow(segmented)
    # plt.show(block=False)
    
    #get a labelled image 
    labeled_image = label(segmented)
   

    # plt.figure()
    # plt.imshow(labeled_image)
    # plt.show(block=False)

    
    #fill holes in small object
    binary_image = labeled_image > 0
    filled_binary_image = binary_fill_holes(binary_image)
    
    # plt.figure()
    # plt.imshow(filled_binary_image)
    # plt.show(block=False)
    
    #remove artefacts of segmentation
    labeled_image = label(filled_binary_image)
    labeled_image= label(remove_small_objects(labeled_image > 0, min_size=20))
    
    # plt.figure()
    # plt.imshow(labeled_image)
    # plt.show(block=False)
    
    color_image = color_labels(labeled_image)
        
    # plt.figure()
    # plt.imshow(color_image)
    # plt.title('Color-Mapped Labeled Image')
    # plt.axis('off')
    # plt.show(block=False)
    
    
    #get the reference object compute row average -> 1 ref spectrum per column
    reference_mask = labeled_image == 1

    #check if reference is value 1 i.e in yellow and rest is value O i.e. in dark blue
    # plt.figure()
    # plt.imshow(reference_mask)
    # plt.axis('off')
    # plt.show(block=False)

    reference_mask=np.repeat(reference_mask[:, :, np.newaxis], hypercube.shape[2], axis=2)
    spectralon = np.where(reference_mask, hypercube, 0)

    # #check spectralon
    # plt.figure()
    # plt.imshow(spectralon[:,:,50])
    # plt.show()
   
    
    avg_spectralon = np.sum(spectralon, axis=0)
    num_valid_pixels = np.sum(reference_mask, axis=0)
    avg_spectralon /= num_valid_pixels    
    avg_spectralon[num_valid_pixels == 0]  = np.nan 
            
    hypercube = np.divide(hypercube, avg_spectralon[np.newaxis, :, :])     
    hypercube_scaled = (hypercube * 65536).astype(np.uint16)  

    
    # save new corrected image in new folder with corresponding header
    base_filename = os.path.splitext(os.path.basename(HSIreader.dataset[idx]['data']))[0]
    save_path = save_folder +f"/{base_filename}_ref.hdr"
    header_path = HSIreader.dataset[idx]['hdr']
    header = envi.read_envi_header(header_path)
    
    envi.save_image(save_path, hypercube_scaled,ext='ref', dtype='uint16', force=True, metadata=header) 

    
    HSIreader.clear_cache()
    
    
