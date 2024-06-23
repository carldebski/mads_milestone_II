import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def svr_model_feature_ablation(df, ticker, features, param_svr):
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


    # set up dataframe of results
    results = pd.DataFrame(columns = ['features', 'mse'])
    
    for i in range(len(features)):
        # extract features and predicted variable
        X = df[features]
        y = df[ticker]

        # initialize model
        model = SVR()

        # Create a pipeline with a scaler and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('SVR', model)])

        # split the data into train and test sets
        tscv = TimeSeriesSplit(n_splits=5)

        # perform GridSearchCV
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_svr, cv=tscv, scoring='neg_mean_squared_error')

        # fit GridSearchCV
        grid_search.fit(X, y)

        # get score
        best_score = -grid_search.best_score_

        # get model
        best_model = grid_search.best_estimator_

        # save results
        results.loc[i, 'mse'] = best_score
        results.loc[i, 'features'] = str(features)        

        # remove one features
        features.pop()

    return results