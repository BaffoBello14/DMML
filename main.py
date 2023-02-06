import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import joblib as jl
import pickle
import datetime as dt

def delete_useless_columns(df):
    df.drop(['id', 'url', 'region_url', 'VIN', 'size', 'image_url', 'description'], axis=1, inplace=True)
    df["condition"].replace(np.nan, 'good', inplace=True)
    df.dropna(subset=['price', 'odometer', 'year', 'manufacturer', 'cylinders', 'fuel', 'transmission', 'drive', 'type'], inplace=True)
    df['age'] = dt.datetime.now().year - df['year']
    return df

def load_file():
    global data
    global df

    file_path = filedialog.askopenfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')])
    df = pd.read_csv(file_path)

    df = delete_useless_columns(df)

    df = df[['manufacturer', 'cylinders', 'fuel', 'odometer', 'drive', 'type', 'long', 'age', 'price']]

    # Deserialization of the ordinal_label mapping from the file
    with open("./Preprocessing/ordinal_label.pkl", "rb") as f:
        ordinal_label = pickle.load(f)
    # Encoding of the categorical columns in the second script
    for column in df:
        if df[column].dtypes == object:
            df[column] = df[column].map(ordinal_label[column])

    data = df

    for i, row in df.iterrows():
        for j, value in enumerate(row.values[:-1]):
            label = tk.Label(root, text=value)
            label.grid(row=i+1, column=j+1)

    predict_button.grid(row=0, column=len(data.columns))

def predict():
    global data
    global model
    prediction = model.predict(data)

    for i, p in enumerate(prediction):
        label = tk.Label(root, text=p)
        label.grid(row=i+1, column=len(data.columns))

global root, predict_button
root = tk.Tk()

load_button = tk.Button(root, text="Load file", command=load_file)
load_button.grid(row=0, column=0)

predict_button = tk.Button(root, text="Predict", command=predict)

model = jl.load("./Models/random_forest.pkl")

root.mainloop()
