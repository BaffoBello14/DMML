from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Carichiamo il dataset
data = pd.read_csv("../Dataset/preprocessed_vehicles.csv")

# Inizializzazione dell'oggetto PCA
pca = PCA()

# Adattamento dell'oggetto PCA al dataset
pca.fit(data)

# calcoliamo la varianza spiegata dalle componenti
explained_variance = pca.explained_variance_ratio_

# scelta automatica del numero di componenti principali
n_components = np.argmax(np.cumsum(explained_variance) > 0.95) + 1

# Inizializzazione dell'oggetto PCA con numero di componenti principali scelto
pca = PCA(n_components=n_components)

# Adattamento dell'oggetto PCA al dataset
pca.fit(data)

# Trasformazione del dataset originale in base alle componenti principali
transformed_data = pca.transform(data)
