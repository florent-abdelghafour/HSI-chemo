import scipy.io
import numpy as np
import os

data_root= "D:\\Piment7RGB\\Piment7RGB.mat"
output_base_dir = "D:\\Piment7RGB\\index_maps\\"
formula_txt_path = "D:\\Piment7RGB\\formula_mapping.txt"


dataset = scipy.io.loadmat(data_root)
list_img = [key for key in dataset.keys() if not key.startswith('__')]

organized_data = {
    "recto": {key: dataset[key] for key in list_img if key.endswith('r')},
    "verso": {key: dataset[key] for key in list_img if key.endswith('v')}
}

del dataset



for key, img in organized_data["recto"].items():
    print(f"image: {key}")