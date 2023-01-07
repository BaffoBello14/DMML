import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats as st
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import spearmanr, kendalltau

def correlationWithPrice(df):
    print("Spearman")
    for i in df.columns:
        corr, p_value = spearmanr(df['price'], df[i])
        print(f'{i} {corr}')

    print("Pearson")
    for i in df.columns:
        corr = df['price'].corr(df[i])
        print(f'{i} {corr}')

    print("Kendall")
    for i in df.columns:
        corr, p_value = kendalltau(df['price'], df[i])
        print(f'{i} {corr}')

    # Seleziona le colonne del DataFrame da utilizzare come caratteristiche
    X = df.loc[:, df.columns.difference(['price', 'Unnamed: 0'])]
    # Seleziona la colonna del DataFrame da utilizzare come etichetta
    y = df['price']
    # Esegui il splitting del dataset in dataset di addestramento e dataset di test
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    # Inizializza il vettorizzatore
    vectorizer = CountVectorizer()
    # Esegui il fitting del vettorizzatore sul dataset di addestramento
    vectorizer.fit(X_train)
    # Trasforma il dataset di addestramento e il dataset di test utilizzando il vettorizzatore
    X_train_transformed = vectorizer.transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    # Genera i nomi delle caratteristiche per il dataset trasformato
    feature_names = vectorizer.get_feature_names_out()
    # Assegna i nomi delle caratteristiche al dataset trasformato
    X_train_transformed.columns = feature_names
    X_test_transformed.columns = feature_names
    # Crea una nuova istanza di SelectKBest utilizzando f_regression come funzione di punteggio
    fs = SelectKBest(score_func=f_regression, k=5)
    # Esegui il fitting di SelectKBest sul dataset trasformato
    print("fs")
    print(fs)
    print("X_train")
    print(X_train)
    print("y_train")
    print(y_train)
    fs.fit(X_train, y_train)
    # Trasforma il dataset di addestramento e il dataset di test utilizzando SelectKBest
    X_train_fs = fs.transform(X_train_transformed)
    x_test_fs = fs.transform(X_test_transformed)
    # Ottieni i nomi delle caratteristiche selezionate
    feature_names = vectorizer.get_feature_names_out()
    # Stampa i nomi delle caratteristiche selezionate insieme ai loro punteggi
    print("regression test results:")
    for i in range(len(fs.scores_)):
        print('Feature %s: %f' % (feature_names[i], fs.scores_[i]))
        
def check_z_scores(df):
    # Normalizziamo il DataFrame con la funzione zscore()
    df = df.apply(st.zscore)

    # Utilizza la funzione between() per selezionare solo le righe che soddisfano la condizione
    df = df[(df['price'].between(-0.01, 0.01)) &
            (df['odometer'].between(-1.8, 1.8)) &
            (df['year'].between(-2.2, 2.2))]

    # Reimposta gli indici del DataFrame per poterli confrontare con gli indici di df
    df = df.reset_index()
    return df

df = pd.read_csv('../Dataset/numerical_data.csv')
df_z_scores = check_z_scores(df)
df = df[df.index.isin(df_z_scores['index'])]
print(df.info)

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
    
correlationWithPrice(df)
