import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Carica il dataset
data = pd.read_csv("../Dataset/vehicles_preprocessed.csv")

# Separa la colonna target dalle features
X = data.drop("price", axis=1)
y = data["price"]

# Inizializzazione dell'oggetto PCA
pca = PCA()

# Adattamento dell'oggetto PCA al dataset
pca.fit(X)

# calcoliamo la varianza spiegata dalle componenti
explained_variance = pca.explained_variance_ratio_

# scelta automatica del numero di componenti principali
n_components = np.argmax(np.cumsum(explained_variance) > 0.99) + 1

# Inizializzazione dell'oggetto PCA con numero di componenti principali scelto
pca = PCA(n_components=9)

# Adattamento dell'oggetto PCA al dataset
pca.fit(X)

# Trasformazione del dataset originale in base alle componenti principali
X_transformed = pd.DataFrame(pca.transform(X))

# Aggiungi la colonna target al dataset ridotto
X_transformed["price"] = y

# Salvataggio del dataset trasformato come file CSV
X_transformed.to_csv("../Dataset/vehicles_preprocessed_PCA.csv", index=False, header=True)
