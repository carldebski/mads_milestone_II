import pandas as pd
import numpy as np
import pickle
import itertools
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def model_comparison(df, ticker, features):
    """
    Provides the r2 scores for a Linear Regression, Support Vector Regression, and Decision Tree Regression model given a list 
    of feature combinations

    Parameters
        > df (pd.DataFrame): dataframe that includes individual feature columns and target variable column 
        > ticker (str): name of the target variable and its respective column in the df input
        > features (list of lists): list of feature groupings for models

    Returns
        > r2_crossval_results (pd.DataFrame): results of the three models when trained using the provided features. 
            columns
            - model: the model used
            - feature: the combination of features used
            - cv: the cross validation step
            - r2: model result 
    """

    
    #models = {'Linear':linear_model, 'Support Vector (SVR)': svr_model, 'Decision Tree':decision_tree_model}
    models = {'Linear':LinearRegression(), 'Support Vector (SVR)': SVR(kernel='rbf'), 'Decision Tree':DecisionTreeRegressor()}
    
    # set up dataframe of results
    combinations = list(itertools.product(models, features, range(1, 6)))
    r2_crossval_results = pd.DataFrame(combinations, columns=['model', 'features', 'cv'])
    r2_crossval_results['features'] = r2_crossval_results['features'].astype(str)
    r2_crossval_results['mse'] = None

    # model and collect results for each group of features
    for feature in features:
            
        # extract features and predicted variable
        X = df[feature]
        y = df[ticker]

        # split the data into train and test sets
        tscv = TimeSeriesSplit(n_splits=5)
        cv = 0
        for train_index, test_index in tscv.split(X):
            cv += 1
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            scaler_x = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_x.fit_transform(X_train)
            X_test_scaled = scaler_x.transform(X_test)
            #y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            #y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

            # evaluate each cross validation split with each of the identified models
            for model in models:
                
                # select model
                m = models[model]

                # fit model using the training data
                m.fit(X_train_scaled, y_train)

                # make predictions using the testing data
                y_pred = m.predict(X_test_scaled)

                # calculate the r2 score of predictions against held out test data
                mse = mean_squared_error(y_test, y_pred)
                condition = (r2_crossval_results['model']==model) & (r2_crossval_results['features']==str(feature)) & (r2_crossval_results['cv']==cv)
                r2_crossval_results.loc[condition, 'mse'] = mse

                # save model in pickle file
                filename = '{} Model.sav'.format(model)
                pickle.dump(m, open(filename, 'wb'))
    
    return r2_crossval_results