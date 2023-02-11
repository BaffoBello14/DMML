import pandas as pd
import numpy as np
import pickle

categorical_columns = []
numerical_columns = []

def import_dataset():
    df1 = pd.read_csv("../Dataset/vehicles10.csv")
    df1.drop('county', axis=1, inplace=True)
    df1['age'] = 2022 - df1['year']

    df2 = pd.read_csv("../Dataset/vehicles9.csv")
    df2.rename(columns={'county': 'posting_date'}, inplace=True)
    df2 = df2[(df2['price'] >= 0) & (df2['price'] <= 5000) | (df2['price'] > 13000)]
    df2['age'] = 2021 - df2['year']

    df3 = pd.read_csv("../Dataset/vehicles8.csv")
    df3 = df3[df3['price'] > 20000]
    df3.drop('Unnamed: 0', axis=1, inplace=True)
    df3['age'] = 2020 - df3['year']

    df = pd.concat([df1, df2, df3], axis=0, sort=False)
    df.drop("posting_date", axis=1, inplace=True)
    df.drop("year", axis=1, inplace=True)

    return df

# url, region_url, vin, size, county


def delete_useless_columns(df):
    df.drop(['id', 'url', 'region_url', 'VIN', 'size', 'image_url', 'description'], axis=1, inplace=True)
    df["condition"].replace(np.nan, 'good', inplace=True)
    df.dropna(subset=['price', 'odometer', 'age', 'manufacturer', 'cylinders', 'fuel', 'transmission', 'drive', 'type'], inplace=True)

def split_categorical_numerical(df):
    ordinal_label = {}
    for column in df:
        if df[column].dtypes == object:
            label_map = {k: i for i, k in enumerate(df[column].unique(), 0)}
            df[column] = df[column].map(label_map)
            ordinal_label[column] = label_map
            categorical_columns.append(column)
        else:
            numerical_columns.append(column)
    print(categorical_columns)
    print(numerical_columns)
    # Serializzazione della mappatura ordinal_label in un file
    with open("ordinal_label.pkl", "wb") as f:
        pickle.dump(ordinal_label, f)

    df.loc[:, categorical_columns].to_csv('../Dataset/categorical_data.csv')
    df.loc[:, numerical_columns].to_csv('../Dataset/numerical_data.csv')

#Carichiamo i Dataset
df = import_dataset()

#Eliminiamo le colonne non utili alla nostra analisi
delete_useless_columns(df)

#Dividiamo il dataset in due dataset: uno con i valori categori, l'altro con i numerici
split_categorical_numerical(df)
#Trasformiamo i valori categorici in numerici attraverso l'enumerazione

#Salviamo il Dataset
df.to_csv('../Dataset/vehicles_preprocessed.csv')
