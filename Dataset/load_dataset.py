import pandas as pd
import numpy as np

categorical_columns = []
numerical_columns = []


def import_dataset():
    # Carica il file CSV in un DataFrame di pandas
    df = pd.read_csv("vehicles.csv")
    #print(df.describe())
    return df


# url, region_url, vin, size, county


def delete_useless_columns(df):
    df.drop('id', axis=1, inplace=True)
    df.drop('url', axis=1, inplace=True)
    df.drop('region_url', axis=1, inplace=True)
    df.drop('VIN', axis=1, inplace=True)
    df.drop('size', axis=1, inplace=True)
    df.drop('image_url', axis=1, inplace=True)
    df.drop('county', axis=1, inplace=True)
    df["condition"].replace('salvage', 1, inplace=True)
    df["condition"].replace('fair', 2, inplace=True)
    df["condition"].replace('good', 3, inplace=True)
    df["condition"].replace('excellent', 4, inplace=True)
    df["condition"].replace('like new', 4, inplace=True)
    df["condition"].replace('new', 5, inplace=True)
    df["condition"].replace(pd.np.nan, 3, inplace=True)
    df["type"].replace(pd.np.nan, 'other', inplace=True)
    df["cylinders"].replace(pd.np.nan, 'other', inplace=True)
    df["drive"].replace(pd.np.nan, 'unknown', inplace=True)
    df["paint_color"].replace(pd.np.nan, 'unknown', inplace=True)
    df.dropna(inplace=True)


def split_categorical_numerical(df):
    for column in df:
        if df[column].dtypes == object:
            categorical_columns.append(column)
            ordinal_label = {k: i for i, k in enumerate(df[column].unique(), 0)}
            df[column] = df[column].map(ordinal_label)
        else:
            numerical_columns.append(column)
    print(df.loc[:, categorical_columns])
    print(df.loc[:, numerical_columns])


df = import_dataset()
delete_useless_columns(df)
print(df.describe())
print(df.info())
print(df.isnull().sum())
print('cylinders')
valori_unici = set(df['cylinders'])
for i in valori_unici:
    print(i)
print('conditions')
valori_unici = set(df['condition'])
for i in valori_unici:
    print(i)
print('drive')
valori_unici = set(df['drive'])
for i in valori_unici:
    print(i)
print('type')
valori_unici = set(df['type'])
for i in valori_unici:
    print(i)
#cylinders, condition, drive, type
print(df.info)
split_categorical_numerical(df)
