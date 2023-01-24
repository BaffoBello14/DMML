import pandas as pd
import numpy as np
from scipy.stats import spearmanr

feature_dropped=[]
df = pd.read_csv('../Dataset/vehicles_preprocessed.csv')
for i in df.columns:
    for j in df.columns:
        if i in feature_dropped:
            continue
        if j in feature_dropped:
            continue
        corr, p_value = spearmanr(df[j], df[i])
        if i == 'Unnamed: 0' or j == 'Unnamed: 0':
            continue
        '''if abs(corr) > 0.8 and i != j:
            print(f'Correlazionte tra {i} e {j}: {corr}')
            corr_i, p_value = spearmanr(df['price'], df[i])
            print(f'Correlazione con price di {i} = {corr_i}')
            corr_j, p_value = spearmanr(df['price'], df[j])
            print(f'Correlazione con price di {j} = {corr_j}')
            if (corr_i > corr_j) :
                df.drop(j, axis=1, inplace=True)
                feature_dropped.append(j)
            else:
                df.drop(i, axis=1, inplace=True)
                feature_dropped.append(i)'''
print('Cerco scarse correlazioni con price')
for i in df.columns:
    if i in feature_dropped:
        continue
    corr, p_value = spearmanr(df['price'], df[i])
    if abs(corr) < 0.1:
        df.drop(i, axis=1, inplace=True)
        feature_dropped.append(i)
print(feature_dropped)
df.to_csv('../Dataset/vehicles_preprocessed.csv')

