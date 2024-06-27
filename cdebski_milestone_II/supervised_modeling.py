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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def model_comparison(df, ticker, features):
    """
    Provides the mse scores for a Linear Regression, Support Vector Regression, and Decision Tree Regression model given a list 
    using selected features. Models are individually tuned using gridsearchcv. 

    Parameters
        > df (pd.DataFrame): dataframe that includes individual feature columns and target variable column 
        > ticker (str): name of the target variable and its respective column in the df input
        > features (list of lists): list of feature groupings for models

    Returns
        > results (pd.DataFrame): results of the three models when trained using the provided features. 
            columns
            - model: the model used
            - feature: the combination of features used
            - best_params: best parameters of tuned model
            - best_mse: model result 
            - best_std: std of the cross validation results
    """


    param_lin = {'Linear__fit_intercept': [False, True]} 
    param_svr = {'Support Vector (SVR)__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'Support Vector (SVR)__C': [0.1, 1, 10, 100, 1000], 'Support Vector (SVR)__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    param_dt = {'Decision Tree__max_depth': [None, 10, 20, 30, 40, 50], 'Decision Tree__criterion': ['squared_error']}
    models = {'Linear':[LinearRegression(), param_lin], 'Support Vector (SVR)': [SVR(), param_svr], 'Decision Tree':[DecisionTreeRegressor(), param_dt]}

    
    # set up dataframe of results
    features_str = []
    features_str.append(', '.join(map(str, features)))
    combinations = list(itertools.product(models, features_str))
    results = pd.DataFrame(combinations, columns=['model', 'features'])
    results['features'] = results['features'].astype(str)
    results['best_params'] = None
    results['best_mse'] = None
    results['best_std'] = None

    # extract features and predicted variable
    X = df[features]
    y = df[ticker]

    # evaluate each cross validation split with each of the identified models
    for model in models:
                
        # select model
        m = models[model][0]

        # Create a pipeline with a scaler and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model, m)])

        # split the data into train and test sets
        tscv = TimeSeriesSplit(n_splits=5)

        # perform GridSearchCV
        grid_search = GridSearchCV(estimator=pipeline, param_grid=models[model][1], cv=tscv, scoring='neg_mean_squared_error')

        # fit GridSearchCV
        grid_search.fit(X, y)

        # get the best parameters
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        best_index = grid_search.best_index_
        best_cross_val = grid_search.cv_results_['std_test_score'][best_index]

        # best model
        best_model = grid_search.best_estimator_

        # save results
        condition = (results['model']==model) & (results['features']==features_str[0])
        results.loc[condition, 'best_mse'] = best_score
        results.loc[condition, 'best_params'] = str(best_params)
        results.loc[condition, 'best_std'] = str(best_cross_val)
        
        # save model in pickle file
        filename = '{} Model.sav'.format(model)
        pickle.dump(best_model, open(filename, 'wb'))   

    results.to_csv('results.csv', index=False)        
    
    return results