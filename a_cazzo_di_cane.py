import shutil
from pathlib import Path
from tkinter import filedialog
import numpy as np
import pandas as pd
import joblib as jl
import PySimpleGUI as sg
import time

def predict_price(df, model):
    X = df.drop('price', axis=1)
    y = df['price']

    predict_y = model.predict(X)
    return predict_y


def main():
    sg.theme("DarkTeal2")
    layout = [[sg.T("")], [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-IN-")],
              [sg.Button("Submit")],
              [sg.Text("Choose a classifier: "), sg.Radio('M5R', 'clf', default=True, key="M5R"),
               sg.Radio('RandomForest', 'clf', key="RandomForest")],
              [sg.ProgressBar(5, orientation='h', size=(50, 20), border_width=4, key="-PROGRESS_BAR-")]]

    window = sg.Window('UsedVehiclesPricePredictor', layout, size=(600, 200))
    rf = jl.load('Models/random_forest.pkl')
    m5r = jl.load('Models/m5_rules.pkl')

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        elif event == "Submit":
            fname = values["-IN-"]

            if not fname:
                sg.popup("Cancel", "No filename supplied")

            if Path(fname).is_file():

                try:
                    if fname.endswith('.csv'):
                        i = 0
                        df = pd.read_csv(filedialog.askopenfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')]))
                        window["-PROGRESS_BAR-"].update(i + 1)
                        i += 1

                        df['age'] = 2023 - df['year']
                        df = df[['manufacturer', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'age', 'price']]

                        window["-PROGRESS_BAR-"].update(i + 1)
                        i += 1
                        # Deserializzazione della mappatura ordinal_label dal file
                        ordinal_label = jl.load("Preprocessing/ordinal_label.pkl")

                        window["-PROGRESS_BAR-"].update(i + 1)
                        i += 1

                        # Codifica delle colonne categoriche nel secondo script
                        for column in df:
                            if df[column].dtypes == object:
                                df[column] = df[column].map(ordinal_label[column])

                        window["-PROGRESS_BAR-"].update(i + 1)
                        i += 1

                        if values["M5R"]:
                            price = predict_price(df, m5r)
                            window["-PROGRESS_BAR-"].update(i + 1)
                            print(price)

                        elif values["RandomForest"]:
                            price = predict_price(df, rf)
                            window["-PROGRESS_BAR-"].update(i + 1)
                            print(price)

                    else:
                        sg.popup("Not a pdf file, please select a valid file")
                except Exception as e:
                    sg.popup("Error: ", e)


if __name__ == '__main__':
    main()
