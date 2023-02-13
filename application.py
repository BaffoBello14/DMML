import tkinter as tk
from tkinter import filedialog
import pandas as pd
import joblib as jl

# Load
random_forest = jl.load("Models/random_forest.pkl")
m5_rules = jl.load("Models/m5_rules.pkl")
ordinal_label = jl.load("Preprocessing/ordinal_label.pkl")

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        # Load button
        self.load_button = tk.Button(self)
        self.load_button["text"] = "LOAD"
        self.load_button["command"] = self.load_csv
        self.load_button.grid(row=0, column=0)

        # Random Forest button
        self.rf_button = tk.Button(self)
        self.rf_button["text"] = "Random Forest"
        self.rf_button["command"] = self.run_random_forest
        self.rf_button.grid(row=1, column=0)

        # M5 Rules button
        self.m5_button = tk.Button(self)
        self.m5_button["text"] = "M5 Rules"
        self.m5_button["command"] = self.run_m5_rules
        self.m5_button.grid(row=2, column=0)

    def load_csv(self):
        # Open a file dialog to select a CSV file
        file_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not file_path:
            return

        # Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(file_path)
        self.data['age'] = 2023 - self.data['year']
        self.data = self.data[['manufacturer', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'age', 'price']]
        self.data_x=self.data[['manufacturer', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'age']]

        # Display the DataFrame in a table
        for j, col in enumerate(self.data_x.columns):
            label = tk.Label(self, text=col, font=("Helvetica", 14, "bold"))
            label.grid(row=3, column=j)
        for i, row in enumerate(self.data_x.values):
            for j, col in enumerate(row):
                label = tk.Label(self, text=col)
                label.grid(row=i + 4, column=j)
            for i in range(3, self.data_x.shape[0] + 4):
                tk.Frame(self, height=2, bd=1, relief="sunken").grid(row=i, column=0, columnspan=self.data_x.shape[1], sticky="ew")
            for j in range(self.data_x.shape[1]):
                tk.Frame(self, width=2, bd=1, relief="sunken").grid(row=3, rowspan=self.data_x.shape[0]+1, column=j, sticky="ns")

    def run_random_forest(self):
        # Run the random forest model on the data
        # ...
        print(1)

    def run_m5_rules(self):
        # Run the M5 rules model on the data
        # ...
        print(2)

root = tk.Tk()
app = Application(master=root)
app.mainloop()
