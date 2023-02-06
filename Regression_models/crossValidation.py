import pandas as pd
import joblib as jl
from sklearn.metrics import mean_squared_error
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

    best_score = 0

    # Inizializza il classificatore Random Forest
    model = RandomForestRegressor(n_jobs=10, n_estimators=500, max_features=4)

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
        print("Mean square error:", mse)
        print("Accuracy:", score)

        # Salvataggio del modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/random_forest.pkl')


def crossValLassoAlg(df):
    X = df.loc[:, df.columns != 'price']
    y = df['price']

    best_score = 0

    # Linear Regression model based on Lasso algorithm
    model = Lasso(alpha=0.1, max_iter=5000, random_state=42)

    # Use K-Fold cross-validation to evaluate the model performance
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print('Results for Lasso')
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestra il modello sul set di training
        model.fit(X_train, y_train)

        # Valuta le prestazioni sul set di test
        predict_y = model.predict(X_test)
        score = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, predict_y)
        print("Mean square error:", mse)
        print("Accuracy:", score)

        # Salvataggio del modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/lasso.pkl')


def crossValAdaBoostRegressor(df):
    X = df.loc[:, df.columns != 'price']
    y = df['price']

    best_score = 0

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
        predict_y = model.predict(X_test)
        score = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, predict_y)
        print("Mean square error:", mse)
        print("Accuracy:", score)

        # Salvataggio del Modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/ada_boost.pkl')


def crossValElasticNet(df):
    X = df.loc[:, df.columns != 'price']
    y = df['price']

    best_score = 0

    #Elastic Net Regressor
    model = ElasticNet(alpha=0.1, max_iter=10000)

    # Use K-Fold cross-validation to evaluate the model performance
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print('Results for ElasticNet')
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestra il modello sul set di training
        model.fit(X_train, y_train)

        # Valuta le prestazioni sul set di test
        predict_y = model.predict(X_test)
        score = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, predict_y)
        print("Mean square error:", mse)
        print("Accuracy:", score)

        # Salvataggio del Modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/elastic_net.pkl')


def crossValKNN(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    best_score = 0

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
        print("Mean square error:", mse)
        print("Accuracy:", score)

        # Salvataggio del Modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/knn.pkl')


def crossValM5Rules(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    best_score = 0

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
        print("Mean square error:", mse)
        print("Accuracy:", score)

        #Salvataggio del Modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/m5_rules.pkl')


def crossValGBR(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    best_score = 0

    # Inizializza il classificatore Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=500, max_features=4)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('Results for GBR')
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

        # Salvataggio del modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/gbr.pkl')


def crossValSVR(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    best_score = 0

    # Inizializza l'algoritmo Support Vector Regression (SVR)
    model = SVR(kernel='linear', C=1, epsilon=0.1)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('Results for SVR')
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

        # Salvataggio del modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/svr.pkl')


def crossValXGBR(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    best_score = 0

    # Inizializza l'algoritmo XGBoost Regressor
    model = XGBRegressor(n_estimators=500, learning_rate=0.05)

    # Inizializza la cross-validation k-fold con 10 divisioni
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    print('Results for XGBR')
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

        # Salvataggio del modello
        if score > best_score:
            best_score = score
            jl.dump(model, '../Models/xgbr.pkl')


df = pd.read_csv("../Dataset/vehicles_preprocessed.csv")
#crossVAlAdaBoostRegressor(df)  #0.56
#crossVAlElasticNet(df)         #0.6
#crossVAlLassoAlg(df)           #0.6
#crossVAlM5Rules(df)            #0.785
#crossVAlKNN(df)                #0.247
crossValRandomForest(df)       #0.89
#crossValGBR(df)                #0.81
#crossValXGBR(df)                #0.85
