import pandas as pd
import numpy as np

categorical_columns = []
numerical_columns = []

def import_dataset():
    df1 = pd.read_csv("../Dataset/vehicles10.csv")
    df1.drop('county', axis=1, inplace=True)

    df2 = pd.read_csv("../Dataset/vehicles8.csv")
    df2.drop('Unnamed: 0', axis=1, inplace=True)

    df = pd.merge(df1, df2, how="outer")

    df['posting_date'] = pd.to_datetime(df['posting_date'])
    df['age'] = df['posting_date'].dt.year - df['year']

    df.drop("posting_date", axis=1, inplace=True)
    df.drop("year", axis=1, inplace=True)
    return df


# url, region_url, vin, size, county


def delete_useless_columns(df):
    df.drop('id', axis=1, inplace=True)
    df.drop('url', axis=1, inplace=True)
    df.drop('region_url', axis=1, inplace=True)
    df.drop('VIN', axis=1, inplace=True)
    df.drop('size', axis=1, inplace=True)
    df.drop('image_url', axis=1, inplace=True)
    df.drop('description', axis=1, inplace=True)
    df["condition"].replace(pd.np.nan, 'good', inplace=True)
    df.dropna(
        subset=['price', 'odometer', 'year', 'manufacturer', 'cylinders', 'fuel', 'transmission', 'drive', 'type'],
        inplace=True)
    # Eliminando tutti i NULL si raggiunge circa lo 0.65 di accuracy (RF)


def split_categorical_numerical(df):
    for column in df:
        if df[column].dtypes == object:
            categorical_columns.append(column)
            ordinal_label = {k: i for i, k in enumerate(df[column].unique(), 0)}
            df[column] = df[column].map(ordinal_label)
        else:
            numerical_columns.append(column)
    print(categorical_columns)
    print(numerical_columns)
    df.loc[:, categorical_columns].to_csv('../Dataset/categorical_data.csv')
    df.loc[:, numerical_columns].to_csv('../Dataset/numerical_data.csv')

#Carichiamo i Dataset
df = import_dataset()

#Eliminiamo le colonne non utili alla nostra analisi
delete_useless_columns(df)

#Dividiamo il dataset in due dataset: uno con i valori categori, l'altro con i numerici
split_categorical_numerical(df)

#Trasformiamo i valori categorici in numerici attraverso l'enumerazione
for column in df:
    if df[column].dtypes == object:
        ordinal_label = {k: i for i, k in enumerate(df[column].unique(), 0)}
        df[column] = df[column].map(ordinal_label)
print(df.info)

#Salviamo il Dataset
df.to_csv('../Dataset/vehicles_preprocessed.csv')
