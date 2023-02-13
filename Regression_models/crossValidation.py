import pandas as pd
import joblib as jl
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


def crossValRandomForest(df):
    X = df.drop("price", axis=1)
    y = df["price"]
    scores = []
    rmses = []
    maes = []

    # Inizializza il classificatore Random Forest
    model = RandomForestRegressor(n_jobs=8, n_estimators=500, max_features=4)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('Results for RF')
    # Inizia la cross-validation
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestra il modello sul set di training
        model.fit(X_train, y_train)

        # Valuta le prestazioni sul set di test
        score = model.score(X_test, y_test)
        predict_y = model.predict(X_test)
        mse = mean_squared_error(y_test, predict_y)
        mae = mean_absolute_error(y_test, predict_y)
        print("Mean Absolute Error:", mae)
        print("Mean square error:", mse)
        print("Accuracy:", score)
        scores.append(score)
        rmses.append(np.sqrt(mse))
        maes.append(mae)
    print("Mean score: ",np.mean(scores))
    print("Mean of means absolute error ", np.mean(maes))
    print("Mean of roots of mean square errors ", np.mean(rmses))


def crossValAdaBoostRegressor(df):
    X = df.loc[:, df.columns != 'price']
    y = df['price']

    scores = []
    rmses = []
    maes = []
    mean_prices=[]

    # Adaboost Regression
    model = AdaBoostRegressor(n_estimators=100, learning_rate=0.3, random_state=42)

    # Use K-Fold cross-validation to evaluate the model performance
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print('Results for Adaboost')
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestra il modello sul set di training
        model.fit(X_train, y_train)

        # Valuta le prestazioni sul set di test
        score = model.score(X_test, y_test)
        predict_y = model.predict(X_test)
        mse = mean_squared_error(y_test, predict_y)
        mae = mean_absolute_error(y_test, predict_y)
        print("Mean Absolute Error:", mae)
        print("Mean square error:", mse)
        print("Accuracy:", score)
        scores.append(score)
        rmses.append(np.sqrt(mse))
        maes.append(mae)
        print(np.mean(y_test))
        mean_prices.append(np.mean(y_test))
    print("Mean score: ", np.mean(scores))
    print("Mean of means absolute error ", np.mean(maes))
    print("Mean of roots of mean square errors ", np.mean(rmses))
    print("Mean of means of prices", np.mean(mean_prices))


def crossValKNN(df):
    X = df.drop("price", axis=1)
    y = df["price"]
    scores = []
    rmses = []
    maes = []
    mean_prices= []

    # Inizializza il classificatore k-NN
    model = KNeighborsRegressor(n_neighbors=5)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('Results for 5-NN')
    # Inizia la cross-validation
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestra il modello sul set di training
        model.fit(X_train, y_train)

        # Valuta le prestazioni sul set di test
        score = model.score(X_test, y_test)
        predict_y = model.predict(X_test)
        mse = mean_squared_error(y_test, predict_y)
        mae = mean_absolute_error(y_test, predict_y)
        print("Mean Absolute Error:", mae)
        print("Mean square error:", mse)
        print("Accuracy:", score)
        scores.append(score)
        rmses.append(np.sqrt(mse))
        maes.append(mae)
        print("Mean of prices ", np.mean(y_test))
        mean_prices.append(np.mean(y_test))
    print("Mean score: ", np.mean(scores))
    print("Mean of means absolute error ", np.mean(maes))
    print("Mean of roots of mean square errors ", np.mean(rmses))
    print("Mean of means of prices ", np.mean(mean_prices))


def crossValM5Rules(df):
    X = df.drop("price", axis=1)
    y = df["price"]
    scores = []
    rmses = []
    maes = []

    # Inizializza il classificatore Decision Tree
    # "squared_error", "friedman_mse", "absolute_error
    model = DecisionTreeRegressor(criterion="friedman_mse", random_state=42)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('Results for M5rules')
    # Inizia la cross-validation
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestra il modello sul set di training
        model.fit(X_train, y_train)

        # Valuta le prestazioni sul set di test
        score = model.score(X_test, y_test)
        predict_y = model.predict(X_test)
        mse = mean_squared_error(y_test, predict_y)
        mae = mean_absolute_error(y_test, predict_y)
        print("Mean Absolute Error:", mae)
        print("Mean square error:", mse)
        print("Accuracy:", score)
        scores.append(score)
        rmses.append(np.sqrt(mse))
        maes.append(mae)
    print("Mean score: ", np.mean(scores))
    print("Mean of means absolute error ", np.mean(maes))
    print("Mean of roots of mean square errors ", np.mean(rmses))


df = pd.read_csv("../Dataset/vehicles_preprocessed.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)
print(np.mean(df['price']))
crossValAdaBoostRegressor(df)  #0.56
#crossValElasticNet(df)         #0.6
#crossValLassoAlg(df)           #0.6
crossValM5Rules(df)            #0.785
crossValKNN(df)                #0.247
crossValRandomForest(df)       #0.89
#crossValGBR(df)                #0.81
#crossValXGBR(df)                #0.85

