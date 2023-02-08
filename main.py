import tkinter as tk
from tkinter import filedialog
import pandas as pd
import joblib as jl
from tkinter import ttk

def load_csv():
    global df
    file_path = filedialog.askopenfilename()
    df = pd.read_csv(file_path)
    column_names = df.columns.tolist()
    for column in column_names:
        tree.heading(column, text=column)
        tree.column(column, width=100)
    for index, row in df.iterrows():
        tree.insert("", "end", values=row.tolist())

def predict():
    model = jl.load('Models/random_forest.pkl')
    predict_y = model.predict(df.drop("price", axis=1))
    predict_y = [round(y, 2) for y in predict_y]
    text.insert(tk.END, f"\n\nPredicted prices:\n{predict_y}")

root = tk.Tk()
root.title("Price Predictor")

load_button = tk.Button(root, text="LOAD", command=load_csv)
load_button.pack()

predict_button = tk.Button(root, text="PREDICT", command=predict)
predict_button.pack()

text = tk.Text(root)
text.pack()

tree = ttk.Treeview(root)
tree.pack()

root.mainloop()
