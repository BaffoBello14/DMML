import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsRegressor


def randomForestEvaluation(df):
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size=0.2,
                                                        train_size=0.8, random_state=np.random.seed(0))
    X_trainn = df.loc[:, df.columns != 'price']
    y_trainn = df['price']
    # Regression Model based on Random Forest Algorithm
    ran_forest = RandomForestRegressor(max_depth=6, random_state=0)

    scores = cross_validate(ran_forest, X_trainn, y_trainn, scoring=('r2', 'explained_variance'), cv=10)
    print("results for ran_forest")
    print(np.mean(scores['test_r2']))
    print(np.mean(scores['test_explained_variance']))

    # fitting model
    ran_forest.fit(X_train, y_train)

    # making predictions
    predict_y = ran_forest.predict(X_test)


def linearRegressionAlg(df):
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size=0.2,
                                                        train_size=0.8, random_state=np.random.seed(0))
    X_trainn = df.loc[:, df.columns != 'price']
    y_trainn = df['price']
    # Regression Model based Multiple Linear Regression Algorithm
    lm = linear_model.LinearRegression()
    scores = cross_validate(lm, X_trainn, y_trainn, scoring=('r2', 'explained_variance'), cv=10)
    print(np.mean(scores['test_r2']))
    print(np.mean(scores['test_explained_variance']))
    # fitting model
    lm.fit(X_train, y_train)
    # making predictions
    predict_y = lm.predict(X_test)


def kNearestNeighbours(df):
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size=0.2,
                                                        train_size=0.8, random_state=np.random.seed(0))
    X_trainn = df.loc[:, df.columns != 'price']
    y_trainn = df['price']
    # Regression based on k-nearest neighbors
    neigh = KNeighborsRegressor(n_neighbors=200)
    scores = cross_validate(neigh, X_trainn, y_trainn, scoring=('r2', 'explained_variance'), cv=10)
    print("results for k-nearest neighbors")
    print(np.mean(scores['test_r2']))
    print(np.mean(scores['test_explained_variance']))

    # fitting model
    neigh.fit(X_train, y_train)

    # making predictions
    predict_y = neigh.predict(X_test)


def elasticNet(df):
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size=0.3,
                                                        train_size=0.7, random_state=np.random.seed(0))
    X_trainn = df.loc[:, df.columns != 'price']
    y_trainn = df['price']
    # Regression model based on Elastic Net algorithm
    en = ElasticNet(alpha=0.01, max_iter=1000)
    scores = cross_validate(en, X_trainn, y_trainn, scoring=('r2',  'explained_variance'), cv=10)
    print("results for Elastic Net")
    print(np.mean(scores['test_r2']))
    print(np.mean(scores['test_explained_variance']))

    # fitting model
    en.fit(X_train, y_train)

    # making predictions
    predict_y = en.predict(X_test)


def lassoAlg(df):
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'price'], df['price'], test_size=0.3,
                                                        train_size=0.7, random_state=np.random.seed(0))
    X_trainn = df.loc[:, df.columns != 'price']
    y_trainn = df['price']
    # Linear  Regression model based on Lasso algorithm
    lasso = Lasso(alpha=0.01, max_iter=1000)
    scores = cross_validate(lasso, X_trainn, y_trainn, scoring=('r2', 'explained_variance'), cv=10)
    print("results for Lasso")
    print(np.mean(scores['test_r2']))
    print(np.mean(scores['test_explained_variance']))
    # fitting model
    lasso.fit(X_train, y_train)
    # making predictions
    predict_y = lasso.predict(X_test)


df = pd.read_csv('../Dataset/vehicles_preprocessed.csv')
#df.drop('lat', inplace=True, axis=1)
#df.drop('long', inplace=True, axis=1)
df.dropna(inplace=True)
randomForestEvaluation(df)
linearRegressionAlg(df)
#kNearestNeighbours(df)
elasticNet(df)
lassoAlg(df)

