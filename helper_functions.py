from bertopic import BERTopic
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# We create a function to calculate a list of the top n topics related to (a) given keyword(s)

def get_relevant_topics(bertopic_model, keywords, top_n):
    '''
    Retrieve a list of the top n number of relevant topics to the provided (list of) keyword(s)
    
    
    Parameters:
        bertopic_model: a (fitted) BERTopic model object
        
        keywords:   a string containing one or multiple keywords to match against,
                    
                    This can also be a list in the form of ['keyword(s)', keyword(s), ...]
                    
                    In this case a maximum of top_n topics will be found per list element 
                    and subsetted to the top_n most relevant topics.
                    
                    !!!
                    Take care that this method only considers the relevancy per inputted keyword(s) 
                    and not the relevancy to the combined list of keywords.
                    
                    In other words, topics that appear in the output might be significantly related to a 
                    particular element in the list of keywords but not so to any other element, 
                    
                    while topics that do not appear in the output might be significantly related to the 
                    combined list of keywords but not much to any of the keyword(s) in particular.
                    !!!
                    
        top_n: an integer indicating the number of desired relevant topics to be retrieved
        
        
        Return: a list of the top_n (or less) topics most relevant to the (list of) provided keyword(s)
    '''
    
    if type(keywords) is str: keywords = [keywords] # If a single string is provided convert it to list type
    
    relevant_topics = list() # Initilize an empty list of relevant topics
    
    for keyword in keywords: # Iterate through list of keywords
        
        # Find the top n number of topics related to the current keyword(s)
        topics = bertopic_model.find_topics(keyword, top_n = top_n)
        
        # Add the topics to the list of relevant topics in the form of (topic_id, relevancy)
        relevant_topics.extend(
            zip(topics[0], topics[1]) # topics[0] = topic_id, topics[1] = relevancy
        )
    
    
    relevant_topics.sort(key=lambda x: x[1]) # Sort the list of topics on ASCENDING ORDER of relevancy
    
    # Get a list of the set of unique topics (with greates relevancy in case of duplicate topics)
    relevant_topics = list(dict(relevant_topics).items())
    
    
    relevant_topics.sort(key=lambda x: x[1], reverse=True) # Now sort the list of topics on DESCENDING ORDER of relevancy
    
    return relevant_topics[:10] # Return a list of the top_n unique relevant topics

def create_lag_df(df, columns, lag, difference=False, rolling=None, dropna=False):
    '''
    Function to add lagged colums to dataframe
    
    Here we define a function that lags input variables. There are options for creating a rolling mean,
    taking the difference between subsequent rows, and dropping NaNs.
    Feature engineering can of course be extended much further than this.

    Inputs:
        df - Dataframe
        columns - List of columns to create lags from
        lag - The number of timesteps (in months for the default data) to lag the variable by
        difference - Whether to take the difference between each observation as new column
        rolling - The size of the rolling mean window, input None type to not use a rolling variable
        dropna - Whether to drop NaN values
        
    Output:
        df - Dataframe with the lagged columns added
    '''
    
    for column in columns:
        col = df[column].unstack()
        if rolling:
            col = col.rolling(rolling).mean()
        if difference:
            col = col.diff()
        if dropna:
            col = col.dropna(how='any')
        df[f"{column}_lag_{lag}"] = col.shift(lag).stack()
    return df

def plot_ConfusionMatrix(prediction, true, binary=False):
    '''
    Function to plot a confusion matrix as a heatmap from a prediction and true values.
    
    Here we define a function that plots a confusion matrix given a prediction
    and the true values, it can be used both for binary and categorical variables.

    Inputs:
        prediction - The predicted values
        true - the true values
        binary - whether the variable is binary or not
        
    Output:
        confusion_matrix - The calculated confusion matrix based on the prediction and true values.
        
        Also plots the confusion matrix as heatmap in an interactive environment such as Jupyter Notebook.
    '''
    
    y_pred = prediction
    
    if not binary:
        # Round prediction to nearest integer (i.e. the nearest phase)
        y_pred = y_pred.round() 
        y_pred = np.minimum(y_pred, 5) # Cap maximum prediction at 5 (maximum phase)
        y_pred = np.maximum(y_pred, 1) # Cap minimum prediction at 1 (minimum phase)

    # Initialize confusion matrix
    confusion_matrix = pd.crosstab(
        true, y_pred, rownames=["Actual"], colnames=["Predicted"]
    )

    # Plot confusion matrix as heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt="g")
    plt.show()
    plt.clf()
    
    return confusion_matrix