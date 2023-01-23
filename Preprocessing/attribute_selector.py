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
    #df = df[(df['price'].between(-0.01, 0.01)) &
    #        (df['odometer'].between(-1.0, 1.0)) &
    #       (df['year'].between(-2.2, 2.2))]
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


def chi2_test_on_categorical_features():
    # this function performs chi2 test on categorical features to find
    # their mutual correlation
    # loading categorical data
    dfc = pd.read_csv('../Dataset/categorical_data.csv', na_filter=False)
    column_names = dfc.columns
    # Assigning column names to row index
    chisqmatrix = pd.DataFrame(dfc, columns=column_names, index=column_names)

    for icol in column_names:  # Outer loop
        for jcol in column_names:  # inner loop
            if icol == 'Unnamed: 0' or jcol == 'Unnamed: 0' or jcol == 'manufacturer' or icol == 'manufacturer':
                continue
            # converting to cross tab as for chi2 test we have to first covert variables into contigency table
            mycrosstab = pd.crosstab(dfc[icol], dfc[jcol])
            # Getting p-value and other usefull information
            stat, p, dof, expected = st.chi2_contingency(mycrosstab)

            # Rounding very small p-values to zero
            chisqmatrix.loc[icol, jcol] = round(p, 5)

            # Expected frequencies should be at
            # least 5 for the majority (80%) of the cells.
            # Here we are checking expected frequency of each group
            cntexpected = expected[expected < 5].size

            # Getting percentage
            perexpected = ((expected.size - cntexpected) / expected.size) * 100
            if perexpected < 20:
                chisqmatrix.loc[icol, jcol] = 2  # Assigning 2

            if icol == jcol:
                chisqmatrix.loc[icol, jcol] = 0.00

    # Saving chi2 results
    chisqmatrix.to_csv('../Dataset/chi2_matrix.csv')

'''
def test_chi2(df, col_a, col_b):
    if col_a == col_b:
        return 0
    # conversione in tabella di frequenze incrociate, poiché per il test di chi-quadrato è necessario prima convertire le variabili in una tabella di frequenze incrociate
    tabella_incrociata = pd.crosstab(df[col_a], df[col_b])
    # Ottenimento del valore p e di altre informazioni utili
    stat, p, dof, attese = st.chi2_contingency(tabella_incrociata)
    print(stat, p, dof, attese)

    # Arrotondamento dei valori p molto piccoli a zero
    #p_arrotondato = round(p, 5)

    # Le frequenze attese dovrebbero essere almeno 5 per la maggior parte (80%) delle celle.
    # Qui si verifica la frequenza attesa di ogni gruppo
    cnt_attese = attese[attese < 5].size

    # Ottenimento del percentuale
    perc_attese = ((attese.size - cnt_attese) / attese.size) * 100
    if perc_attese < 20:
        return 2
    return p




df_categorical = pd.read_csv('../Dataset/numerical_data.csv')
df_senza_Unnamed = df_categorical.loc[:, ~df_categorical.columns.isin(['Unnamed: 0'])]
matrice_chi2 = pd.DataFrame(df_senza_Unnamed, index=df_senza_Unnamed.columns, columns=df_senza_Unnamed.columns)
# Passaggio degli argomenti corretti alla funzione
for col_a in df_categorical.columns:
    for col_b in df_categorical.columns:
        if col_a == 'Unnamed: 0' or col_b == 'Unnamed: 0':
            continue
        print(col_a)
        print(col_b)
        matrice_chi2[col_a][col_b] = test_chi2(df_categorical, col_a, col_b)
matrice_chi2.to_csv('../Dataset/chi2_matrix.csv')
print(matrice_chi2)

chi2_test_on_categorical_features()
#categorical_graph(df_categorical)
'''

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
    print(df.info)
    return df


df_numerical = pd.read_csv('../Dataset/numerical_data.csv')
df_numerical = outlier_deleter(df_numerical)
df_z_scores = check_numerical_z_scores(df_numerical)
df_numerical = df_numerical[df_numerical.index.isin(df_z_scores['index'])]
df_categorical = pd.read_csv('../Dataset/categorical_data.csv')
df_categorical = df_categorical[df_categorical.index.isin(df_z_scores['index'])]
df = pd.read_csv('../Dataset/vehicles_preprocessed.csv')
df = df[df.index.isin(df_z_scores['index'])]
print(df_numerical.info)
print(df_categorical.info)
print(df.info)

df_numerical.drop('Unnamed: 0', axis=1, inplace=True)
df_numerical.to_csv('../Dataset/numerical_data.csv')
df_categorical.drop('Unnamed: 0', axis=1, inplace=True)
df_categorical.to_csv('../Dataset/categorical_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.to_csv('../Dataset/vehicles_preprocessed.csv')
numerical_graph(df_numerical)
correlationWithPrice(df)

