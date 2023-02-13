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



def linearRegressionAlg(df):
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size=0.2,
                                                        train_size=0.8, random_state=np.random.seed(0))
    # Regression Model based Multiple Linear Regression Algorithm
    lm = linear_model.LinearRegression()
    # fitting model
    lm.fit(X_train, y_train)
    scores = cross_validate(lm, X_test, y_test, scoring=('r2', 'explained_variance'), cv=10)
    # making predictions
    predict_y = lm.predict(X_test)
    mse = mean_squared_error(y_test, predict_y)
    print("Result for linear regression")
    print(mse)
    print(np.mean(scores['test_r2']))
    print(np.mean(scores['test_explained_variance']))


def KNN(df):
    X = df.drop('price', axis=1)
    y = df['price']
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)
    jl.dump(model, '../Models/knn.pkl')



def elasticNet(df):
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size=0.3,
                                                        train_size=0.7, random_state=np.random.seed(0))
    X_trainn = df.loc[:, df.columns != 'price']
    y_trainn = df['price']
    # Regression model based on Elastic Net algorithm
    en = ElasticNet(alpha=0.01, max_iter=1000)
    scores = cross_validate(en, X_trainn, y_trainn, scoring=('r2', 'explained_variance'), cv=10)
    # fitting model
    en.fit(X_train, y_train)
    # making predictions
    predict_y = en.predict(X_test)
    mse = mean_squared_error(y_test, predict_y)
    print("results for Elastic Net")
    print(mse)
    print(np.mean(scores['test_r2']))
    print(np.mean(scores['test_explained_variance']))


def lassoAlg(df):
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size=0.3,
                                                        train_size=0.7, random_state=np.random.seed(0))
    X_trainn = df.loc[:, df.columns != 'price']
    y_trainn = df['price']
    # Linear  Regression model based on Lasso algorithm
    lasso = Lasso(alpha=0.01, max_iter=1000)
    scores = cross_validate(lasso, X_trainn, y_trainn, scoring=('r2', 'explained_variance'), cv=10)
    # fitting model
    lasso.fit(X_train, y_train)
    # making predictions
    predict_y = lasso.predict(X_test)
    mse = mean_squared_error(y_test, predict_y)
    print("results for Lasso")
    print(mse)
    print(np.mean(scores['test_r2']))
    print(np.mean(scores['test_explained_variance']))


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
#linearRegressionAlg(df)
KNN(df)
M5R(df)
adaboost(df)
#elasticNet(df)
#lassoAlg(df)
