import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

Data_path = 'D:/HSI data/barley_swir_20241029.mat'


dataset = scipy.io.loadmat(Data_path)
list_keys = [key for key in dataset.keys()]

dry_set =dataset['spectra_snv_dry']
irr_set =dataset['spectra_snv_irrigation']


X = np.vstack((dry_set, irr_set))  
# Create labels: 0 for dry, 1 for irrigated
y = np.concatenate((np.zeros(dry_set.shape[0]), np.ones(irr_set.shape[0])))

# Perform PCA
n_components = 4  # Change to 3 for 3D visualization
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
fig, axes = plt.subplots(n_components-1, n_components-1, figsize=(12, 10))
for i in range(n_components-1):
    for j in range(i+1, n_components):
        ax = axes[i, j-1]
        ax.scatter(X_pca[y == 0, i], X_pca[y == 0, j], label='Dry', alpha=0.7)
        ax.scatter(X_pca[y == 1, i], X_pca[y == 1, j], label='Irrigated', alpha=0.7)
        ax.set_xlabel(f'PC{i+1}')
        ax.set_ylabel(f'PC{j+1}')
        if i == 0 and j == 1:
            ax.legend()

plt.suptitle("PCA Scores Plot")
plt.tight_layout()
plt.show()

# -------------------------------
# Plot PCA Loadings
# -------------------------------
plt.figure(figsize=(10, 6))
for i in range(n_components):
    plt.plot(pca_loadings[:, i], label=f'PC{i+1}')

plt.xlabel("Wavelength Index")
plt.ylabel("Loading Value")
plt.title("PCA Loadings Plot")
plt.legend()
plt.show()