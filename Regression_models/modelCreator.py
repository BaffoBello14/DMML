import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import joblib as jl
from sklearn.tree import DecisionTreeRegressor


def randomForest(df):
    X = df.drop('price', axis=1)
    y = df['price']
    model = RandomForestRegressor(n_jobs=10, n_estimators=500, max_features=4)
    model.fit(X, y)
    jl.dump(model, '../Models/random_forest.pkl')


def KNN(df):
    X = df.drop('price', axis=1)
    y = df['price']
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)
    jl.dump(model, '../Models/knn.pkl')


def M5R(df):
    X = df.drop('price', axis=1)
    y = df['price']
    model = DecisionTreeRegressor(criterion="friedman_mse", random_state=42)
    model.fit(X, y)
    jl.dump(model, '../Models/m5_rules.pkl')


def adaboost(df):
    X = df.drop('price', axis=1)
    y = df['price']
    model = AdaBoostRegressor(n_estimators=100, learning_rate=0.3, random_state=42)
    model.fit(X, y)
    jl.dump(model, '../Models/ada_boost.pkl')



df = pd.read_csv('../Dataset/vehicles_preprocessed.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
randomForest(df)
KNN(df)
M5R(df)
adaboost(df)
