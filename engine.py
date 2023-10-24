from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from statsmodels.regression.linear_model import OLS
from sklearn import linear_model
from sklearn.metrics import accuracy_score, f1_score, r2_score
import matplotlib.pyplot as plt

from helper_functions import plot_ConfusionMatrix

def model_eval(X, y, model_type = "OLS"):
    if model_type == "OLS":
        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits

        # Initinalize empty lists to score scores
        mae_values = list()
        r2_values = list()
        f1_values = list()
        accuracy_values = list()

        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits

        # Initinalize empty lists to score scores
        mae_values = list()
        r2_values = list()
        f1_values = list()
        accuracy_values = list()

        for train_index, val_index in cv.split(X): # Loop over the different training-test splits

            # Define X and y data
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # If X_train doesn't contain any news features (this happens for earlier dates) we drop news columns from both X_train and X_test
            X_train = X_train.dropna(axis=1, how='all').copy()
            X_test = X_test[X_train.columns]
            
            #Interpolate training data to generate more training points
            X_train = X_train.groupby('district', as_index=False).apply(lambda group: group.interpolate())
            X_train.reset_index(level=0, drop=True, inplace=True)
            y_train = y_train.groupby('district', as_index=False).apply(lambda group: group.interpolate())
            y_train.reset_index(level=0, drop=True, inplace=True)

            model = OLS(y_train, X_train, missing="drop") # Initialize OLS model on training data
            model = model.fit() # Get model results on training data
            print(model.summary()) # Print model summary

            y_pred = model.predict(X_test) # Run model on test data
            
            # Append results to respective lists
            mae_values.append((y_pred - y_val['ipc']).abs().mean())
            r2_values.append(model.rsquared)

            y_pred = np.round(y_pred[y_val['ipc'].notnull()])
            y_val = np.round(y_val[y_val['ipc'].notnull()])

            accuracy_values.append(accuracy_score(y_pred=y_pred,
                            y_true=y_val['ipc']))

            f1_values.append(f1_score(y_pred=y_pred,
                            y_true=y_val['ipc'], average='weighted'))

            plot_ConfusionMatrix(predicted_labels = y_pred, true_labels = y_val['ipc']) # Plot confusion matrix
            
        print(f"Mean MAE: {np.mean(mae_values):.2f}") # Print MAE
        print(f"Mean R2: {np.mean(r2_values):.2f}") # Print R2
        print(f"Mean Accuracy: {100*np.mean(accuracy_values):.2f}%") # Print Mean Accuracy
        print(f"Mean Weighted F1: {np.mean(f1_values):.3f}") # Print Mean Weighted F1
    
    elif model_type == "Ridge":

        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits

        # Initinalize empty lists to score scores
        mae_values = list()
        r2_values = list()
        accuracy_values = list()
        f1_values = list()

        for train_index, val_index in cv.split(X): # Loop over the different training-test splits

            # Define X and y data
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # If X_train doesn't contain any news features (this happens for earlier dates) we drop news columns from both X_train and X_test
            X_train = X_train.dropna(axis=1, how='all').copy()
            X_test = X_test[X_train.columns]
            
            #Interpolate training data to generate more training points
            X_train = X_train.groupby('district', as_index=False).apply(lambda group: group.interpolate())
            X_train.reset_index(level=0, drop=True, inplace=True)
            y_train = y_train.groupby('district', as_index=False).apply(lambda group: group.interpolate())
            y_train.reset_index(level=0, drop=True, inplace=True)
            
            # model = OLS(y_train, X_train, missing="drop") # Initialize OLS model on training data
            model = linear_model.Ridge(alpha=3)

            model.fit(X_train, y_train) # Get model results on training data

            y_pred = np.round(model.predict(X_test).T[0]) # Run model on test data

            # Append results to respective lists
            mae_values.append((y_pred - y_val['ipc']).abs().mean())

            cur_r2 = r2_score(y_pred=model.predict(X_train),
                            y_true=y_train['ipc'])
            r2_values.append(cur_r2)

            accuracy_values.append(accuracy_score(y_pred=y_pred,
                            y_true=y_val['ipc']))

            f1_values.append(f1_score(y_pred=y_pred,
                            y_true=y_val['ipc'], average='weighted'))

            plot_ConfusionMatrix(predicted_labels = y_pred, true_labels = y_val['ipc']) # Plot confusion matrix
            
        print(f"Mean MAE: {np.mean(mae_values):.2f}") # Print MAE
        print(f"Mean R2: {np.mean(r2_values):.2f}") # Print R2
        print(f"Mean Accuracy: {100*np.mean(accuracy_values):.2f}%") # Print Mean Accuracy
        print(f"Mean Weighted F1: {np.mean(f1_values):.3f}") # Print Mean Weighted F1
    return model