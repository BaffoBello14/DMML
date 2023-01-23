import pandas as pd
import numpy as np

categorical_columns = []
numerical_columns = []


def import_dataset():
    # Carica il file CSV in un DataFrame di pandas
    df = pd.read_csv("../Dataset/vehicles.csv")
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
    '''df["type"].replace(pd.np.nan, 'unknown', inplace=True)
    df["cylinders"].replace(pd.np.nan, 'unknown', inplace=True)
    df["drive"].replace(pd.np.nan, 'unknown', inplace=True)
    df["paint_color"].replace(pd.np.nan, 'unknown', inplace=True)'''
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



df2=pd.read_csv("../Dataset/vehicles_8.csv")
df2.drop('Unnamed: 0', axis=1, inplace=True)
df2['age'] = 2020 - df2['year']
'''for i in df2.nrows:
    data = df2[i]['posting_date']
    data = data[0:3]
    
print(df2['posting_date'])'''
df = import_dataset()
df['age'] = 2022 - df['year']
df.drop('county', axis=1, inplace=True)
df_merged = pd.merge(df, df2, how="outer")
delete_useless_columns(df_merged)
split_categorical_numerical(df_merged)
for column in df_merged:
    if df_merged[column].dtypes == object:
        ordinal_label = {k: i for i, k in enumerate(df_merged[column].unique(), 0)}
        df_merged[column] = df_merged[column].map(ordinal_label)
print(df_merged.info)
df_merged.to_csv('../Dataset/vehicles_preprocessed.csv')
