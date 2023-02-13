from tkinter import filedialog
import numpy as np
import pandas as pd
import joblib as jl
import time

def load_dataset():
    df = pd.read_csv(filedialog.askopenfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')]))
    df['age'] = 2023 - df['year']
    df = df[['manufacturer', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'age', 'price']]
    #Mostrare a video nell'app(tranne price)
    print(df)
    # Deserializzazione della mappatura ordinal_label dal file
    ordinal_label = jl.load("Preprocessing/ordinal_label.pkl")
    # Codifica delle colonne categoriche nel secondo script
    for column in df:
        if df[column].dtypes == object:
            df[column] = df[column].map(ordinal_label[column])
    return df

def predict_price(df, model):
    X = df.drop('price', axis=1)
    y = df['price']

    predict_y = model.predict(X)
    return predict_y

df = load_dataset()
df_adjusted = df
price_time = np.zeros(5)
start_time = time.time()
#model = jl.load(filedialog.askopenfilename(defaultextension='.pkl', filetypes=[('PKL files', '*.pkl')]))
model = jl.load('Models/m5_rules.pkl')
for i in range(0, 5):
    df_adjusted['age'] = df['age'] + i
    df_adjusted['odometer'] = (df['odometer']/df['age'])*(df['age'] + i)
    price = predict_price(df_adjusted, model)
    price_time[i] = price
    print("--- %s seconds ---" % (time.time() - start_time))
print(price_time)
