import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as st
from scipy.stats import spearmanr, kendalltau
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split


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
    feature_names = vectorizer.get_feature_names()
    # Assegna i nomi delle caratteristiche al dataset trasformato
    X_train_transformed.columns = feature_names
    X_test_transformed.columns = feature_names
    # Crea una nuova istanza di SelectKBest utilizzando f_regression come funzione di punteggio
    fs = SelectKBest(score_func=f_regression, k=5)
    # Esegui il fitting di SelectKBest sul dataset trasformato
    fs.fit(X_train, y_train)
    # Trasforma il dataset di addestramento e il dataset di test utilizzando SelectKBest
    X_train_fs = fs.transform(X_train_transformed)
    x_test_fs = fs.transform(X_test_transformed)
    # Ottieni i nomi delle caratteristiche selezionate
    feature_names = vectorizer.get_feature_names()
    # Stampa i nomi delle caratteristiche selezionate insieme ai loro punteggi
    print("regression test results:")
    for i in range(len(fs.scores_)):
        print('Feature %s: %f' % (feature_names[i], fs.scores_[i]))



def numerical_graph(df_numerical):
    for att in df_numerical.columns[:-1]:
        plt.figure()
        print(df_numerical[att].max())
        print(df_numerical[att].min())
        plt.hist(df_numerical[att], range=(df_numerical[att].min(), df_numerical[att].max()))
        plt.ylabel('occurrences')
        plt.xlabel(att)
        plt.title(f'histogram of {att} attribute')
        plt.show()
    for att in df_numerical.columns[:-1]:
        plt.boxplot(df_numerical[att])
        plt.ylabel('values')
        plt.xlabel(att)
        plt.title(f'boxplot of {att} attribute')
        plt.show()


def check_numerical_z_scores(df):
    # Normalizziamo il DataFrame con la funzione zscore()
    df = df.apply(st.zscore)

    # Utilizza la funzione between() per selezionare solo le righe che soddisfano la condizione
    df = df[(df['price'].between(-3.0, 3.0)) &
            (df['odometer'].between(-2.0, 2.0)) &
            (df['age'].between(-2.0, 2.0))] #(df['year'].between(-2.0, 2.0))

    # Reimposta gli indici del DataFrame per poterli confrontare con gli indici di df
    df = df.reset_index()
    return df


def categorical_graph(df):
    for att in df.columns[:-1]:
        plt.figure()
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



def outlier_deleter(df):
    lower_bound = np.quantile(df['price'], q=0.10)
    upper_bound = np.quantile(df['price'], q=0.90)
    print(lower_bound)
    print(upper_bound)
    df = df[(df['price'].between(lower_bound, upper_bound))]
    lower_bound = np.quantile(df['odometer'], q=0.10)
    upper_bound = np.quantile(df['odometer'], q=0.90)
    print(lower_bound)
    print(upper_bound)
    df = df[(df['odometer'].between(lower_bound, upper_bound))]

    #df = df[df['year'].between(1990, 2023)]
    df = df[df['age'].between(0, 33)]
    print(df.info)
    return df



df_numerical = pd.read_csv('../Dataset/numerical_data.csv')
df_numerical = outlier_deleter(df_numerical)
df_z_scores = check_numerical_z_scores(df_numerical)
df_numerical = df_numerical[df_numerical.index.isin(df_z_scores['index'])]
df_categorical = pd.read_csv('../Dataset/categorical_data.csv')
df_categorical = df_categorical[df_categorical.index.isin(df_z_scores['index'])]
df = pd.read_csv('../Dataset/vehicles_preprocessed.csv')
df.to_csv('../Dataset/vehicles_preprocessed2.csv')
df = df[df.index.isin(df_z_scores['index'])]
print(df_numerical.info)
print(df_categorical.info)
print(df.info)

df_numerical.drop('Unnamed: 0', axis=1, inplace=True)
df_numerical.to_csv('../Dataset/numerical_data.csv')
df_categorical.drop('Unnamed: 0', axis=1, inplace=True)
df_categorical.to_csv('../Dataset/categorical_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.dropna(inplace=True)
df.to_csv('../Dataset/vehicles_preprocessed.csv')
numerical_graph(df_numerical)
correlationWithPrice(df)
