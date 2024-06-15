

def svr_model(X, y):
    # import libraries
    from sklearn.svm import SVR

    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # create model
        svr_model = SVR(kernel='rbf') 

        # fit model using the training data
        svr_model.fit(X_train, y_train)

        # make predictions using the testing data
        y_pred = svr_model.predict(X_test)

        # calculate the Mean Squared Error (MSE) and R-squared (R2) score
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    print(r2_scores)
    return np.mean(r2)




def linear_model(X, y):
    # import libraries
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Initialize the scaler
        #scaler = StandardScaler()
        
        # apply scaling 
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)
        # X_test_scaled = scaler.transform(X_test)
        # X_test_scaled = scaler.transform(X_test)

        # create and fit model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # make predictions using the testing data
        y_pred = model.predict(X_test)

        # Debugging: Print actual vs predicted values for the first fold
        if len(r2_scores) == 0:
            print("First fold actual vs predicted values:")
            print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(20))
        
        # Calculate r2 score
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
    
    print(r2_scores)

    return np.mean(r2_scores)

def decision_tree_model(X, y):
    # import libraries
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        # create model
        dtr_model = DecisionTreeRegressor() 

        # split train and test sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # fit model using the training data
        dtr_model.fit(X_train, y_train)

        # make predictions using the testing data
        y_pred = dtr_model.predict(X_test)

        # Debugging: Print actual vs predicted values for the first fold
        if len(r2_scores) == 0:
            print("First fold actual vs predicted values:")
            print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(20))

        # calculate the Mean Squared Error (MSE) and R-squared (R2) score
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)



    print(r2_scores)

    return np.mean(r2_scores)


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
    # import libraries
    import pandas as pd
    import numpy as np
    import itertools
    from sklearn.metrics import r2_score
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import TimeSeriesSplit
    
    #models = {'Linear':linear_model, 'Support Vector (SVR)': svr_model, 'Decision Tree':decision_tree_model}
    models = {'Linear':LinearRegression(), 'Support Vector (SVR)': SVR(kernel='rbf'), 'Decision Tree':DecisionTreeRegressor()}
    
    # set up dataframe of results
    combinations = list(itertools.product(models, features, range(1, 6)))
    r2_crossval_results = pd.DataFrame(combinations, columns=['model', 'features', 'cv'])
    r2_crossval_results['features'] = r2_crossval_results['features'].astype(str)
    r2_crossval_results['r2'] = None

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

            # evaluate each cross validation split with each of the identified models
            for model in models:

                # select model
                m = models[model]

                # fit model using the training data
                m.fit(X_train, y_train)

                # make predictions using the testing data
                y_pred = m.predict(X_test)

                # calculate the r2 score of predictions against held out test data
                r2 = r2_score(y_test, y_pred)
                condition = (r2_crossval_results['model']==model) & (r2_crossval_results['features']==str(feature)) & (r2_crossval_results['cv']==cv)
                r2_crossval_results.loc[condition, 'r2'] = r2
    
    return r2_crossval_results


import pandas as pd

#df = pd.read_csv('cdebski_milestone_II/combined_data.csv')
#df['all']= df[['all']].shift(1)
#df['all'] = df['all'].rolling(window=3).mean()
#df['GME'] = df['GME'].diff()
#df.dropna(inplace=True)
#print(df.head())
#r2 = svr_model(df[['all']], df['GME'])
#print(r2)

#results = model_comparison(df[['GME', '0.0:Crime/Nefarious Use of Stock Market', '0.0:Diamond Hands, Apes Vs. Hedge Funds ', 'all']], 'GME', ['0.0:Diamond Hands, Apes Vs. Hedge Funds ', 'all'])
#print(results)