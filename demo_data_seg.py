import os
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA
from skimage.filters import threshold_multiotsu
from skimage.measure import label
from scipy.ndimage import median_filter

from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects


main_data_folder = 'D:/HSI data/Barley_ground_30cm_SWIR'

save_folder = os.path.join(main_data_folder,'article_figures')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
    
    
    
dataset =HsiDataset(main_data_folder,data_ext='hyspex')
nb_images = len(dataset)
# Initialize the HSI reader: class containg info about the image and allow to load and operate on the hsi
HSIreader = HsiReader(dataset)

# Principal component corresponding to best segmentaion of reference and background
pc_comp_ref=0
# size of median filter, deal with dead pixels and spikes
filter_size=(7, 7, 1)
#if true correct only foreground, ref = 1 & background = 0
mask_foreground=True


idx=0
HSIreader.read_image(idx) #reads without loading! to get metadata
metadata = HSIreader.current_metadata

#define wavelenrghts (for plots mostly)
wv =HSIreader.get_wavelength()
wv = [int(l) for l in wv]

# Get the hyperspectral data
hypercube= HSIreader.get_hsi()
hypercube = median_filter(hypercube, size=filter_size)

n_samples = 50000
x_idx = np.random.randint(0, hypercube.shape[0], size=n_samples)
y_idx = np.random.randint(0, hypercube.shape[1], size=n_samples)

spectral_samples = hypercube[x_idx, y_idx, :]

nb_pca_comp =3
pca = PCA(n_components=nb_pca_comp)
pca_scores = pca.fit_transform(spectral_samples)
pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(np.shape(pca_loadings)[1]):
    plt.figure()
    plt.plot(wv,pca_loadings[:,i],default_colors[i])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")  
    lab= 'PC'+str(i+1)
    plt.title(lab) 
    plt.grid()  
    plt.tight_layout()
    save_file =os.path.join(save_folder,f"{lab}.pdf")
    plt.savefig(save_file)
plt.show()

plt.figure(figsize=(10, 6))
for i in range(np.shape(pca_loadings)[1]):
    plt.plot(wv, pca_loadings[:, i], label=f'PC{i + 1}', color=default_colors[i])

# Add labels, title, legend, and grid
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("PCA Loadings")
plt.legend()
plt.grid()
plt.tight_layout()
save_file = os.path.join(save_folder, "PCA_Loadings.pdf")
plt.savefig(save_file)
plt.show()

score_img = HSIreader.project_pca_scores(pca_loadings)
score_pc_ref = score_img[:,:,pc_comp_ref]   


for i in range(np.shape(pca_loadings)[0]):
    score_map = score_img[:,:,i]   
    plt.figure()
    plt.imshow(score_map)
    plt.axis('off')
    plt.colorbar(label="Score Intensity")
    plt.title(f"Score Map - PC{i + 1}")
    plt.tight_layout()
    
    save_file = os.path.join(save_folder, f"score_map_pc{i+1}.pdf")
    plt.savefig(save_file)
plt.show()

thresholds = threshold_multiotsu(score_pc_ref, classes=3)
segmented = np.digitize(score_pc_ref, bins=thresholds)
labeled_image = label(segmented)

plt.figure(figsize=(10, 6))
plt.hist(score_pc_ref.ravel(), bins=50, color='blue', alpha=0.7, label="Scores")

# Add red vertical lines for the thresholds
for i,threshold in enumerate(thresholds):
    plt.axvline(threshold, color='red', linestyle='--', label=f"T{i+1}")
    
    
    
ylim = plt.gca().get_ylim()
y_pos = ylim[1] * 0.7  # Adjusted y-position slightly lower
plt.text(thresholds[0] / 2, y_pos, "Classe 1", color='black', fontsize=12, ha='center',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
plt.text((thresholds[0] + thresholds[1]) / 2, y_pos, "Classe 2", color='black', fontsize=12, ha='center',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
plt.text(thresholds[1] + (plt.gca().get_xlim()[1] - thresholds[1]) / 2, y_pos, "Classe 3", color='black', fontsize=12, ha='center',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Add labels, legend, and grid
plt.xlabel("Score Intensity")
plt.ylabel("Frequency")
plt.title("Histogram of PCA Scores with Thresholds")
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
save_file = os.path.join(save_folder, "histogram_scores_PC1.pdf")
plt.savefig(save_file)
plt.show()



#fill holes in small object
binary_image = labeled_image > 0
binary_image = labeled_image > 0 #label 0 = background 
labeled_image=label(binary_image)   

#get the reference object compute row average -> 1 ref spectrum per column
reference_mask = labeled_image == 1
background_mask = labeled_image == 0
foreground_mask = np.logical_and(np.logical_not(background_mask), np.logical_not(reference_mask))

masks=[reference_mask,background_mask,foreground_mask]
labels=['Reference','Background','Kernels']


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, mask, label in zip(axes, masks, labels):
    ax.imshow(mask, cmap='gray')
    ax.set_title(label)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

# Adjust layout
plt.title('Segmentation binary masks')
plt.tight_layout()
plt.show()









###################################################





main_data_folder = 'D:/HSI data/VNIR_barley' 
dataset =HsiDataset(main_data_folder,data_ext='ref')
nb_images = len(dataset)
HSIreader = HsiReader(dataset)


nb_pca_comp=3
pca = PCA(n_components=nb_pca_comp)
slice_step = 1000

kernel_comp=1
min_kernel_size = 1000
horizontal_tolerance =100

h_t=100 # horizontal_threshold 
v_t=300 # vertical_threshold

HSIreader.read_image(idx) #reads without loading! to get metadata
metadata = HSIreader.current_metadata

#define wavelenrghts (for plots mostly)
wv =HSIreader.get_wavelength()
wv = [int(l) for l in wv]

hsi=HSIreader.current_image
image_name  = HSIreader.current_name

pca_score_image = project_hsi_VNIR(HSIreader, slice_step, n_samples=50000, nb_pca_comp=3)
score_pc_ref = pca_score_image[:,:,kernel_comp]   
thresholds = threshold_multiotsu(score_pc_ref, classes=2)
segmented = np.digitize(score_pc_ref, bins=thresholds)

pca_score_image = project_hsi_VNIR(HSIreader, slice_step, n_samples=50000, nb_pca_comp=3)
score_pc_ref = pca_score_image[:,:,kernel_comp]   
thresholds = threshold_multiotsu(score_pc_ref, classes=2)
segmented = np.digitize(score_pc_ref, bins=thresholds)

# plt.figure()
# plt.imshow(segmented)
# plt.show()

labeled_image = label(segmented)  #get a labelled image   
filled_binary_image = binary_fill_holes(labeled_image)   #fill holes in small object
labeled_image = label(filled_binary_image)
labeled_image= label(remove_small_objects(labeled_image > 0, min_size=min_kernel_size)) # remove artefacts of segmentation

color_image = color_labels(labeled_image)
plt.figure()
plt.imshow(color_image)
plt.title('Color-Mapped Labeled Image')
plt.axis('off')
plt.show()