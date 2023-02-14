import tkinter as tk
from tkinter import filedialog
import pandas as pd
import joblib as jl
from matplotlib import pyplot as plt

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
        self.load_button.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        # Random Forest button
        self.rf_button = tk.Button(self)
        self.rf_button["text"] = "Random Forest"
        self.rf_button["command"] = self.run_random_forest
        self.rf_button.grid(row=0, column=1, padx=10, pady=10, sticky="e")

        # M5 Rules button
        self.m5_button = tk.Button(self)
        self.m5_button["text"] = "M5 Rules"
        self.m5_button["command"] = self.run_m5_rules
        self.m5_button.grid(row=0, column=2, padx=10, pady=10, sticky="e")

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
        for column in self.data:
            if self.data[column].dtypes == object:
                self.data[column] = self.data[column].map(ordinal_label[column])

        # Display the DataFrame in a table
        for j, col in enumerate(self.data_x.columns):
            label = tk.Label(self, text=col, font=("Helvetica", 14, "bold"), padx=5, pady=5)
            label.grid(row=3, column=j)
        for i, row in enumerate(self.data_x.values):
            for j, col in enumerate(row):
                label = tk.Label(self, text=col, padx=5, pady=5)
                label.grid(row=i + 4, column=j)
            for i in range(3, self.data_x.shape[0] + 4):
                tk.Frame(self, height=2, bd=1, relief="sunken").grid(row=i, column=0, columnspan=self.data_x.shape[1], sticky="ew")
            for j in range(self.data_x.shape[1]):
                tk.Frame(self, width=2, bd=1, relief="sunken").grid(row=3, rowspan=self.data_x.shape[0]+1, column=j, sticky="ns")

    def run_random_forest(self):
        X = self.data.drop('price', axis=1)
        y = self.data['price']
        prices = []
        for i in range(0, 5):
            X['age'] = self.data['age'] + i
            X['odometer'] = (self.data['odometer'] / self.data['age']) * (i + self.data['age'])
            predict_y = random_forest.predict(X)
            prices.append(predict_y)
        plt.plot(range(int(self.data.iloc[0]['age']), int(self.data.iloc[0]['age'])+5), prices)
        plt.xlabel('Anni')
        plt.ylabel('Valore predetto')
        plt.show()



    def run_m5_rules(self):
        X = self.data.drop('price', axis=1)
        y = self.data['price']
        prices = []
        for i in range(0, 5):
            X['age'] = self.data['age'] + i
            X['odometer'] = (self.data['odometer'] / self.data['age']) * (i + self.data['age'])
            predict_y = m5_rules.predict(X)
            prices.append(predict_y)
        plt.plot(range(int(self.data.iloc[0]['age']), int(self.data.iloc[0]['age']) + 5), prices)
        plt.xlabel('Anni')
        plt.ylabel('Valore predetto')
        plt.show()


root = tk.Tk()
app = Application(master=root)
app.master.geometry("800x500")
app.mainloop()
