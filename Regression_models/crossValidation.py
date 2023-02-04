import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error



def crossValRandomForest(df):
    X = df.drop("price", axis=1)
    y = df["price"]
    # Inizializza il classificatore Random Forest
    model = RandomForestRegressor(n_jobs=10, n_estimators=500, max_features=4)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('results for RF')
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
        print("Mean square error:", mse)
        print("Accuracy:", score)


def crossValLassoAlg(df):
    X = df.loc[:, df.columns != 'price']
    y = df['price']

    # Use K-Fold cross-validation to evaluate the model performance
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print('Results for Lasso')
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Linear Regression model based on Lasso algorithm
        lasso = Lasso(alpha=0.1, max_iter=5000, random_state=42)
        lasso.fit(X_train, y_train)
        predict_y = lasso.predict(X_test)
        score = lasso.score(X_test, y_test)
        mse = mean_squared_error(y_test, predict_y)
        print("Mean square error:", mse)
        print("Accuracy:", score)


def crossValAdaBoostRegressor(df):
    X = df.loc[:, df.columns != 'price']
    y = df['price']

    # Use K-Fold cross-validation to evaluate the model performance
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print('Results for Adaboost')
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Adaboost Regression
        ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.3 ,random_state=42)
        ada.fit(X_train, y_train)
        predict_y = ada.predict(X_test)
        score = ada.score(X_test, y_test)
        mse = mean_squared_error(y_test, predict_y)
        print("Mean square error:", mse)
        print("Accuracy:", score)


def crossValElasticNet(df):
    X = df.loc[:, df.columns != 'price']
    y = df['price']

    # Use K-Fold cross-validation to evaluate the model performance
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print('Results for ElasticNet')
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = ElasticNet(alpha=0.1, max_iter=10000)
        model.fit(X_train, y_train)
        predict_y = model.predict(X_test)
        score = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, predict_y)
        print("Mean square error:", mse)
        print("Accuracy:", score)


def crossValKNN(df):
    X = df.drop("price", axis=1)
    y = df["price"]
    # Inizializza il classificatore k-NN
    model = KNeighborsRegressor(n_neighbors=5)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('results for 5-NN')
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
        print("Mean square error:", mse)
        print("Accuracy:", score)



def crossValM5Rules(df):
    X = df.drop("price", axis=1)
    y = df["price"]
    # Inizializza il classificatore Decision Tree
    model = DecisionTreeRegressor(criterion="mse", random_state=42)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('results for M5rules')
    # Inizia la cross-validation
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestra il modello sul set di training
        model.fit(X_train, y_train)

        # Valuta le prestazioni sul set di test
        predict_y = model.predict(X_test)
        mse = mean_squared_error(y_test, predict_y)
        print("Mean square error:", mse)
        print("Accuracy:", model.score(X_test, y_test))




df = pd.read_csv("../Dataset/vehicles_preprocessed.csv")
#crossValAdaBoostRegressor(df)  #circa 0.47
#crossValElasticNet(df)  #circa 0.5
#crossValLassoAlg(df)    #circa 0.5
#crossValM5Rules(df)     #circa 0.61
#crossValKNN(df)         #circa 0.35
crossValRandomForest(df)
