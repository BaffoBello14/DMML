import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt


def import_dataset():
    df1 = pd.read_csv("../Dataset/vehicles10.csv")
    df1.drop('county', axis=1, inplace=True)
    #df1=df1[df1['price']<100000]
    df1['age'] = 2022 - df1['year']

    df2 = pd.read_csv("../Dataset/vehicles9.csv")
    df2.rename(columns={'county': 'posting_date'}, inplace=True)
    #df2 = df2[df2['price'] < 100000]
    #df2 = df2[(df2['price'] >= 0) & (df2['price'] <= 5000) | (df2['price'] > 13000)]
    df2['age'] = 2021 - df2['year']

    df3 = pd.read_csv("../Dataset/vehicles8.csv")
    #df3 = df3[df3['price'] > 20000]
    #df3 = df3[(df3['price'].between(12500, 100000))]
    #df3 = df3[df3['price'] < 100000]
    df3.drop('Unnamed: 0', axis=1, inplace=True)
    df3['age'] = 2020 - df3['year']

    df = pd.concat([df1, df2, df3], axis=0, sort=False)
    df.drop("posting_date", axis=1, inplace=True)
    df.drop("year", axis=1, inplace=True)

    return df


def numerical_graph(df_numerical):
    plt.figure()
    print(df_numerical['price'].max())
    print(df_numerical['price'].min())
    plt.hist(df_numerical['price'], range=(df_numerical['price'].min(), df_numerical['price'].max()))
    plt.ylabel('occurrences')
    plt.xlabel('price')
    plt.title(f'histogram of price attribute')
    plt.show()


df = import_dataset()
#numerical_graph(df)
