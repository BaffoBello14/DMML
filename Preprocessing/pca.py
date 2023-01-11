from sklearn.decomposition import PCA
from pandas import read_csv
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px

X = read_csv('../Dataset/vehicles_preprocessed.csv')
y = X['price']
X.drop('price', axis=1, inplace=True)
X.drop('Unnamed: 0', axis=1, inplace=True)
pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)
print(X_reduced.shape)
print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Paired')
plt.xlabel('Component #1')
plt.ylabel('Component #2')
plt.show()
'''
column_names = X.columns
pca = PCA(n_components=2)
components = pca.fit_transform(X)  # notice that we are coupling fit and transform in a single statement
print(X_reduced.shape)
print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
df = pd.DataFrame(pca.components_, columns=column_names)
print(df)

total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter_3d(
    components, x=0, y=1, color=y,
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2'}
)
fig.show()
'''