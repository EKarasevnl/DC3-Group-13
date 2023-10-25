# Standard imports
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from statsmodels.regression.linear_model import OLS
from sklearn import linear_model
from sklearn.metrics import accuracy_score, f1_score, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import List
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm_notebook

# Custom imports
from helper_functions import plot_ConfusionMatrix

def model_eval(X, y, model_type = "OLS"):
    ''' Function to train and evaluate a given model
    
    Inputs:
        X - dataframe with predictor variables
        y - dataframe with ipc scores
        model_type - "OLS"/"Ridge"/"XGboost"/"NN" choice of the model to evaluate
    Output:
        model - model trained on X and y
        if "NN" is selected:
            true_labels - true IPC labels for test dataset
            predicted_probabilities - confidence for all IPC scores prediction for each data point
            predicted_labels - the label predicted by NN.'''
    
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

            plot_ConfusionMatrix(predicted_labels = y_pred, true_labels = y_val['ipc'], cm_title = "Template model") # Plot confusion matrix
            
        print(f"Mean MAE: {np.mean(mae_values):.2f}") # Print MAE
        print(f"Mean R2: {np.mean(r2_values):.2f}") # Print R2
        print(f"Mean Accuracy: {100*np.mean(accuracy_values):.2f}%") # Print Mean Accuracy
        print(f"Mean Weighted F1: {np.mean(f1_values):.3f}") # Print Mean Weighted F1
    
    elif model_type == "Ridge":

        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits

        # Initinalize empty lists to store scores
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

            plot_ConfusionMatrix(predicted_labels = y_pred, true_labels = y_val['ipc'], cm_title = "Ridge Regression") # Plot confusion matrix
            
        print(f"Mean MAE: {np.mean(mae_values):.2f}") # Print MAE
        print(f"Mean R2: {np.mean(r2_values):.2f}") # Print R2
        print(f"Mean Accuracy: {100*np.mean(accuracy_values):.2f}%") # Print Mean Accuracy
        print(f"Mean Weighted F1: {np.mean(f1_values):.3f}") # Print Mean Weighted F1
    
    elif model_type == "XGboost":

        cv = TimeSeriesSplit(n_splits=5) # Define TimeSeriesSplit with 5 splits
        # Initinalize empty lists to store scores
        mae_values = list()
        f1_values = list()
        accuracy_values = list()

        for train_index, val_index in cv.split(X): # Loop over the different training-test splits

            # Define X and y data
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # If X_train doesn't contain any news features (this happens for earlier dates) we drop news columns from both X_train and X_test
            X_train = X_train.dropna(axis=1, how='all').copy()
            X_test = X_test[X_train.columns]

            # Create a DMatrix for XGBoost (XGBoost uses its own data structure)
            dtrain = xgb.DMatrix(X_train, label=y_train)

            # Specify the XGBoost parameters (you can adjust these as needed)
            params = {
                "objective": "reg:squarederror",  # Regression task
                "max_depth": 4,  # Maximum depth of the tree
                "eta": 0.02,  # Learning rate
                "subsample": 0.7,  # Fraction of data to randomly sample for each tree
                "colsample_bytree": 0.7,  # Fraction of features to randomly sample for each tree
            }

            # Train the XGBoost model
            num_round = 100  # Number of boosting rounds (you can adjust this)
            model = xgb.train(params, dtrain, num_round)

            # Create a DMatrix for X_test
            dtest = xgb.DMatrix(X_test)

            # Make predictions on the test data
            y_pred = np.round(model.predict(dtest))

            # Append results to respective lists
            mae_values.append((y_pred - y_val['ipc']).abs().mean())

            accuracy_values.append(accuracy_score(y_pred=y_pred,
                            y_true=y_val['ipc']))

            f1_values.append(f1_score(y_pred=y_pred,
                            y_true=y_val['ipc'], average='weighted'))

            plot_ConfusionMatrix(predicted_labels = y_pred, true_labels = y_val['ipc'], cm_title = "XGboost") # Plot confusion matrix
            
        print(f"Mean MAE: {np.mean(mae_values):.3f}") # Print MAE
        print(f"Mean accuracy: {100*np.mean(accuracy_values):.2f}%") # Print R2
        print(f"Mean weighted f1: {np.mean(f1_values):.3f}") # Print R2

    elif model_type == "NN":

        # Removes rows with null values for the latest lag there is
        if "ipc_lag_3" in X.columns:
            y = y[X['ipc_lag_3'].notnull()]
            X = X[X['ipc_lag_3'].notnull()]
        else:
            y = y[X['ipc_lag_1'].notnull()]
            X = X[X['ipc_lag_1'].notnull()]     

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=69)
        
        # Normalize your data
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train.values)
        X_test_model = scaler.transform(X_test.values)

        # Convert data to PyTorch tensors
        X_train_model = torch.FloatTensor(X_train_model)
        X_test_model = torch.FloatTensor(X_test_model)
        y_train_model = torch.LongTensor(y_train.values)
        y_test_model = torch.LongTensor(y_test.values)

        # Subtract 1 from the target labels to make them range from 0 to 4
        y_train_model = torch.LongTensor([label - 1 for label in y_train_model]).unsqueeze(dim=1)
        y_test_model = torch.LongTensor([label - 1 for label in y_test_model]).unsqueeze(dim=1)

        # Create DataLoader for training and testing
        train_dataset = TensorDataset(X_train_model, y_train_model)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dataset = TensorDataset(X_test_model, y_test_model)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # NN architecture
        class HungerModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super(HungerModel, self).__init__()
                self.linear_layer_1 = nn.Linear(input_size, 64)
                self.relu = nn.ReLU()
                self.linear_layer_2 = nn.Linear(64, num_classes)

            def forward(self, x):
                x = self.linear_layer_1(x)
                x = self.relu(x)
                x = self.linear_layer_2(x)
                return x

        # Define class weights based on ordinal distance
        class_weights = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0])

        input_size = X_train.shape[1]
        num_classes = 5

        model = HungerModel(input_size, num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)


        num_epochs = 1200
        best_loss = 10000
        mean_losses_train: List[torch.Tensor] = []
        mean_losses_test: List[torch.Tensor] = []


        for epoch in tqdm_notebook(range(num_epochs), desc="Model Training"):
            model.train()

            # Save the model before training
            if best_loss == 10000:
                torch.save(model.state_dict(), 'best_model.pth')

            curr_losses_train: List[torch.Tensor] = []
            curr_losses_test: List[torch.Tensor] = []
            for batch in train_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.squeeze()
                train_loss = criterion(outputs, labels)
                curr_losses_train.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
            mean_losses_train.append(sum(curr_losses_train) / len(curr_losses_train))

            # Evaluation on the test data
            with torch.no_grad():
                for batch in test_loader:
                    inputs, labels = batch
                    outputs = model(inputs)
                    labels = labels.squeeze()
                    test_loss = criterion(outputs, labels)
                    curr_losses_test.append(test_loss.item())
                mean_losses_test.append(sum(curr_losses_test) / len(curr_losses_test))

            # Save the model if the current test loss is better than the previous best
            if mean_losses_test[-1] < best_loss:
                best_loss = mean_losses_test[-1]
                # Save the model checkpoint
                torch.save(model.state_dict(), 'best_model.pth')
                
        model = HungerModel(input_size, num_classes)
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()  # Set the model to evaluation model

        sns.lineplot(x=range(len(mean_losses_train)), y=mean_losses_train, label="train")
        sns.lineplot(x=range(len(mean_losses_test)), y=mean_losses_test, label="test")

        # Set plot labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Test Loss")

        # Add a legend
        plt.legend()

        min_test_loss_epoch = mean_losses_test.index(min(mean_losses_test))
        plt.axvline(x=min_test_loss_epoch, color='green', linestyle='--', label="Lowest Test Loss")
        
        # Show the plot
        plt.show()

        # Lists to store true labels and predicted labels
        true_labels = []
        predicted_labels = []
        predicted_probabilities = []  # Added for MSE and MAE

        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                outputs = model(inputs)
                # Apply softmax and get predicted classes using argmax
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                true_labels.extend(labels.numpy())
                predicted_labels.extend(predicted_classes.numpy())

                predicted_probabilities.extend(probabilities.numpy())  # Added for MSE and MAE

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Calculate F1 score
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        plot_ConfusionMatrix(predicted_labels = predicted_labels, true_labels = true_labels, cm_title = "NN Model") # Plot confusion matrix

        # Convert true_labels to one-hot encoded format
        num_classes = 5  # Assuming you have 5 classes
        true_labels_onehot = np.zeros((len(true_labels), num_classes))
        true_labels_onehot[np.arange(len(true_labels)), true_labels] = 1

        mae = np.mean(np.abs(np.array(predicted_labels) - [i[0] for i in true_labels]))

        print("Mean Absolute Error (MAE):", mae)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)

        return model, true_labels, predicted_probabilities, predicted_labels

    return model