import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats as st


def outlierDetector(df):
    for att in df.columns[:-1]:
        z_scores = st.zscore(df.columns[:, att])
        df["z-score"] = z_scores
        print(np.max(df["z-score"]))
        print(np.min(df["z-score"]))
        df = df[(df["z-score"] > -1) & (df["z-score"] < 1)]


def remove_outliers(df):
    # Calcola gli z-score di ogni punto dei dati
    z = st.zscore(df)
    print(z)
    # Soglia per gli z-score al di fuori della quale rimuovere i dati
    threshold = 2 # Seleziona solo i dati il cui z-score è inferiore alla soglia
    df = df[(z < threshold) & (z > -threshold)]

'''
def vecchiaManiera(df):
    # Calcoliamo lo z-score di ogni valore del DataFrame
    selected_columns_df = df[['price', 'year', 'odometer']]
    z_scores = st.zscore(selected_columns_df)

    # Iteriamo su ogni riga del DataFrame
    for i, row in df.iterrows():
        # Se almeno uno z-score della riga è maggiore di 2 o minore di -2, eliminiamo la riga
        if (z_scores.loc[i] > 3).any() or (z_scores.loc[i] < -3).any():
            df.drop([i], inplace=True)
'''

'''
def vecchiaManiera(df):
    # Calcoliamo lo z-score di alcuni valore del DataFrame
    selected_columns_df = df[['price', 'year', 'odometer']]
    z_scores = st.zscore(selected_columns_df)

    # Iteriamo su ogni riga del DataFrame
    for i, row in df.iterrows():
        # elimino riga
        if (z_scores[i, 1] > 0.5) or (z_scores[i, 1] < -0.5):
            df = df.drop([i])
        if (z_scores[i, 2] > 2.5) or (z_scores[i, 2] < -2.5):
            df = df.drop([i])
        if (z_scores[i, 4] > 1.5) or (z_scores[i, 2] < -1.5):
            df = df.drop([i])

    # Restituisce il DataFrame filtrato
    return df
'''
def check_z_scores(df):
    selected_column = df['price']
    z_score = st.zscore(selected_column)
    for i, row in df.iterrows():
        # elimino riga
        if (z_score[i] > 0.5) or (z_score[i] < -0.5):
            df = df.drop([i])
    selected_column = df['odometer']
    z_score = st.zscore(selected_column)
    for i, row in df.iterrows():
        # elimino riga
        if (z_score[i] > 1.5) or (z_score[i] < -1.5):
            df = df.drop([i])
    selected_column = df['year']
    z_score = st.zscore(selected_column)
    for i, row in df.iterrows():
        # elimino riga
        if (z_score[i] > 0.5) or (z_score[i] < -0.5):
            df = df.drop([i])
    return df


def correlationWithPrice(df):
    corr = np.corrcoef(df)



df = pd.read_csv('../Dataset/numerical_data.csv')
df = check_z_scores(df)
for att in df.columns[:-1]:
    plt.figure()
    print(df[att].max())
    print(df[att].min())
    plt.hist(df[att], range=(df[att].min(), df[att].max()))
    plt.ylabel('occurrences')
    plt.xlabel(att)
    plt.title(f'histogram of {att} attribute')
    plt.show()
for att in df.columns[:-1]:
    plt.boxplot(df[att])
    plt.ylabel('values')
    plt.xlabel(att)
    plt.title(f'boxplot of {att} attribute')
    plt.show()





