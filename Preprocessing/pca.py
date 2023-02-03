import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

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
pca = PCA(n_components=5)

# Adattamento dell'oggetto PCA al dataset
pca.fit(X)

# Trasformazione del dataset originale in base alle componenti principali
X_transformed = pca.transform(X)

# Dividi i dati trasformati in un set di addestramento e un set di test
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=np.random.seed(0))

# Inizializzazione dell'oggetto XGBoost Regressor
xgb = XGBRegressor(random_state=0)

# Dichiarazione dei parametri da testare
param_grid = {'max_depth': [5, 15],
              'n_estimators': [50, 200]}

# Inizializzazione di GridSearchCV
grid_search = GridSearchCV(xgb, param_grid, cv=5)

# Adattamento del modello al set di addestramento
grid_search.fit(X_train, y_train)

# Previsione del prezzo delle macchine usate sul set di test
y_pred = grid_search.predict(X_test)

# Calcola il valore di R2 e MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(r2)
print(mse)
