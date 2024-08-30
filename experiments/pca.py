import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_process(data, dim):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # PCA降维
    pca = PCA(n_components=dim)
    principal_components = pca.fit_transform(scaled_data)
    return principal_components

