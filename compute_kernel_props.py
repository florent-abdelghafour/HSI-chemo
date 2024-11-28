import os
import matplotlib.pyplot as plt
from hsi_utils import *
import json


main_data_folder = 'D:/HSI data/VNIR_barley' 
# D:/HSI data/Barley_ground_30cm_SWIR
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
    print('path invalid')
    

HSIreader = HsiReader(dataset)

for idx in range(len(dataset)):
    HSIreader.read_image(idx) #reads without loading! to get metadata
    metadata = HSIreader.current_metadata
    
    #define wavelenrghts (for plots mostly)
    wv =HSIreader.get_wavelength()
    wv = [int(l) for l in wv]
    
    hsi=HSIreader.current_image
    image_name  = HSIreader.current_name
    
    kernel_path = os.path.join((main_data_folder+'\kernel_data'),(image_name+'\kernel.json)'))
   
    
    print(kernel_path)
    
   
    
    
    