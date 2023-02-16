import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import numpy as np
import joblib as jl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


random_forest = jl.load("Models/random_forest.pkl")
m5_rules = jl.load("Models/m5_rules.pkl")
ordinal_label = jl.load("Preprocessing/ordinal_label.pkl")

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Load button
        self.load_button = tk.Button(self)
        self.load_button["text"] = "LOAD"
        self.load_button["command"] = self.load_csv
        self.load_button.pack()

        # Random Forest button
        self.rf_button = tk.Button(self)
        self.rf_button["text"] = "Random Forest"
        self.rf_button["command"] = self.run_random_forest
        self.rf_button.pack()

        # M5 Rules button
        self.m5_button = tk.Button(self)
        self.m5_button["text"] = "M5 Rules"
        self.m5_button["command"] = self.run_m5_rules
        self.m5_button.pack()

    def load_csv(self):
        # Open a file dialog to select a CSV file
        file_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not file_path:
            return
        for widget in root.winfo_children():
            widget.destroy()
        super().__init__(root)
        self.master = root
        self.pack()
        self.create_widgets()
        # Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(file_path)
        self.data['age'] = 2023 - self.data['year']
        self.data = self.data[['manufacturer', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'age', 'price']]
        self.data_x=self.data[['manufacturer', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'age']]
        for column in self.data:
            if self.data[column].dtypes == object:
                self.data[column] = self.data[column].map(ordinal_label[column])
        table = ttk.Treeview(root)
        table['columns']=('manufacturer', 'cylinders', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'age')
        for col in table['columns']:
            table.heading(col, text=col)
        for col in table['columns']:
            table.column(col, width=150)
        val=[]
        for i in table['columns']:
            val.append(self.data_x.iloc[0][i])
        table.insert("", tk.END, text="Vehicle's details", values=val)
        table.pack()


    def run_random_forest(self):
        X = self.data.drop('price', axis=1)
        y = self.data['price']
        prices = np.zeros(5)
        for i in range(0, 5):
            X['age'] = self.data['age'] + i
            X['odometer'] = (self.data['odometer'] / self.data['age']) * (i + self.data['age'])
            predict_y = random_forest.predict(X)
            prices[i]=predict_y
        fig = plt.figure()
        plt.plot(range(int(self.data.iloc[0]['age']), int(self.data.iloc[0]['age']) + 5), prices)
        plt.title("Random Forest Prediction")
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side='left')
        ext_price = prices[0]
        formatted_price = "${:.2f}".format(ext_price)
        label = tk.Label(self, text="Estimated actual price: " + formatted_price, font=("Helvetica", 9, "bold"), padx=5, pady=5)
        label.place(relx=0.25, rely=1.0, anchor='s')



    def run_m5_rules(self):
        X = self.data.drop('price', axis=1)
        y = self.data['price']
        prices = np.zeros(5)
        for i in range(0, 5):
            X['age'] = self.data['age'] + i
            X['odometer'] = (self.data['odometer'] / self.data['age']) * (i + self.data['age'])
            predict_y = m5_rules.predict(X)
            prices[i] = predict_y
        fig = plt.figure()
        plt.plot(range(int(self.data.iloc[0]['age']), int(self.data.iloc[0]['age']) + 5), prices)
        plt.title("M5Rules Prediction")
        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side='right')
        ext_price = prices[0]
        formatted_price = "${:.2f}".format(ext_price)
        label = tk.Label(self, text="Estimated actual price: " + formatted_price, font=("Helvetica", 9, "bold"), padx=5, pady=5)
        label.place(relx=0.75, rely=1.0, anchor='s')


root = tk.Tk()
app = Application(master=root)
app.master.title("UsedVehiclesPricePredictor")
app.master.attributes('-fullscreen', True)
app.mainloop()
