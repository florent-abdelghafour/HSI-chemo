import os
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from hsi_utils import *

from sklearn.decomposition import PCA


# Define the path to the main data folder: code will iterate trough relvant files
main_data_folder = 'D:/VNIR_barley' 
# D:/HSI data/Barley_ground_30cm_SWIR
# 'D:/VNIR_barley' 

dataset =HsiDataset(main_data_folder,data_ext='ref')
nb_images = len(dataset)
#check if data path exists !
if os.path.isdir(main_data_folder):
    if nb_images>0:
        print(f"dataset  is valid and contains {nb_images} image(s)")
    else:
        print('empty dataset')
else:
    print('path invalid')


HSIreader = HsiReader(dataset)

# Loop through each hyperspectral image in the dataset
# for idx in range(len(dataset)):
#     HSIreader.read_image(idx)
##############################################################################################
##############################################################################################


#choose an image to process e.g. first img idx=0
idx=0
HSIreader.read_image(idx)
print(f"read image{HSIreader.current_name}")

#define wavelenrghts (for plots mostly)
wv =HSIreader.get_wavelength()
wv = [int(l) for l in wv]
##############################################################################################
##############################################################################################


##############################################################################################
###               Read  N=n_samples  random pixels and plot ssêctral samples               ###
##############################################################################################

hsi=HSIreader.current_image
n_rows, n_cols, n_channels =hsi.shape

n_samples =1000
# draw random samples and load them as a data array
x_idx = np.random.randint(0, n_cols, size=n_samples)
y_idx = np.random.randint(0, n_rows, size=n_samples)
spectral_samples = np.zeros((n_samples, n_channels), dtype=hsi.dtype)
coords = list(zip(y_idx, x_idx))
spectral_samples = np.array(HSIreader.extract_pixels(coords))

plt.figure()
for i in range(n_samples):
    plt.plot(wv, spectral_samples[i, :], label=f'Sample {i+1}' if i < 5 else "", alpha=0.6)
plt.xlabel("Wavelength")
plt.ylabel("Absorbance")
plt.title("Spectral samples")
plt.show()
##############################################################################################
##############################################################################################

# Get the pseudo rgb imahe from hyperspectral data and plot image
# pseudo_rgb= HSIreader.get_rgb()
# plt.figure()
# plt.imshow(pseudo_rgb)
# plt.axis('off')
# plt.show()
##############################################################################################
##############################################################################################


#check spectral data by manually selecting pixel-spectrum samples
HSIreader.get_spectrum();
##############################################################################################
##############################################################################################

