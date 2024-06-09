import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def linear_model(X, y):
    # import libraries
    from sklearn.linear_model import LinearRegression

    # create model
    model = LinearRegression()

    # split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit model using the training data
    model.fit(X_train, y_train)

    # make predictions using the testing data
    y_pred = model.predict(X_test)

    # calculate the Mean Squared Error (MSE) and R-squared (R2) score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return r2, mse


def svr_model(X, y):
    # import libraries
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # create model
    svr_model = SVR(kernel='rbf') 
    svr_model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))

    #split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scores = cross_val_score(svr_model, X, y, cv=5, scoring='r2').mean()

    #apply scaling and reshaping 
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.transform(X_test)
    y_train_scaled = sc_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
    y_test_scaled = sc_y.transform(y_test.values.reshape(-1,1)).ravel()

    # fit model using the training data
    svr_model.fit(X_train_scaled, y_train_scaled)

    # make predictions using the testing data
    y_pred = svr_model.predict(X_test_scaled)
    y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1)).ravel()

    # calculate the Mean Squared Error (MSE) and R-squared (R2) score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return r2, mse


def decision_tree_model(X, y):
    # import libraries
    from sklearn.tree import DecisionTreeRegressor

    # create model
    dtr_model = DecisionTreeRegressor() 

    # split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit model using the training data
    dtr_model.fit(X_train, y_train)

    # make predictions using the testing data
    y_pred = dtr_model.predict(X_test)

    # calculate the Mean Squared Error (MSE) and R-squared (R2) score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return r2, mse


def model_comparison(df, ticker, features):
    #models = {'Linear':linear_model(), 'Support Vector (SVR)': svr_model(), 'Decision Tree':decision_tree_model()}
    models = {'Linear':LinearRegression(), 'Support Vector (SVR)': make_pipeline(StandardScaler(), SVR(kernel='rbf')), 'Decision Tree':DecisionTreeRegressor()}
    
    # set up dataframe of results
    df_results = pd.DataFrame(columns=models.keys(), index=features)

    # loop through each community & topic combination 
    for feature in features:
        for model in models:
            
            # split the data into train and test sets
            X = df[[feature]]
            y = df[ticker]

            m = models[model]
            r2 = cross_val_score(m, X, y, cv=5, scoring='r2')

            # store R2 mean in dataframe 
            df_results.loc[feature, model] = r2.mean()
    
    return df_results.astype(float)
