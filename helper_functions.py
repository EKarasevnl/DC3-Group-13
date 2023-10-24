import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import numpy as np


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


def plot_ConfusionMatrix(predicted_labels, true_labels, num_classes = 5, binary=False):
    '''
    Function to plot a confusion matrix as a heatmap from a prediction and true values.
    
    Inputs:
        prediction - The predicted values
        true - the true values
        binary - whether the variable is binary or not
        
    Output:
        confusion_matrix - The calculated confusion matrix based on the prediction and true values.
        
        Also plots the confusion matrix as heatmap in an interactive environment such as Jupyter Notebook.
    '''
    
    predicted_labels = predicted_labels[true_labels.notnull()]
    true_labels = true_labels[true_labels.notnull()]
    
    # Calculate confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)
    confusion = confusion[:num_classes - 1, :num_classes - 1]
    
    # Create a confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size if needed
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=[str(i+1) for i in range(num_classes - 1)],
                yticklabels=[str(i+1) for i in range(num_classes - 1)])
    plt.xlabel('Predicted IPC scores')
    plt.ylabel('True IPC scores')
    plt.title('Confusion Matrix - NN model')
    plt.show()

def create_news_features(columns, news_df):
    cols = []
    for column in columns:
        col = news_df.groupby(["date"])[column].mean()
        col = col.fillna(0)
        col = col.rolling(3).mean()
        col = col.shift(3)
        cols.append(col)
    return pd.concat(cols, axis=1)

def articles_per_ipc(y, data_dir = "data/"):

    news_df = pd.read_csv(data_dir + "df_news_districted.csv") # Read news data into DataFrame

    # Create date column
    news_df["date"] = pd.to_datetime(
        pd.to_datetime(news_df["date"], format="%Y-%m-%d").dt.strftime("%Y-%m"),
        format="%Y-%m",
    )
    news_df = news_df.drop(columns=['Unnamed: 0'])
    combined = ( pd.DataFrame(y['ipc'])
    .join(news_df.groupby(["date"])["hunger"].mean())
    .join(news_df.groupby(["date"])["refugees"].mean())
    .join(news_df.groupby(["date"])["Diplomacy"].mean())
    .join(news_df.groupby(["date"])["Crisis"].mean())
    .join(news_df.groupby(["date"])["Security"].mean())
    .join(news_df.groupby(["date"])["Education"].mean())
    .join(news_df.groupby(["date"])["International"].mean())
    .join(news_df.groupby(["date"])["Media"].mean())
    .join(news_df.groupby(["date"])["Leadership"].mean()))

    # Plot the mean share of articles per ipc value for the different topics
    combined.groupby("ipc")[combined.columns[1:]].mean().plot(
        kind="bar", ylabel="Share of total articles", title = "Mean Share of Articles per IPC"
    );

def articles_per_region(data_dir = "data/"):

    df_news = pd.read_csv("data/articles_topics_original.csv", parse_dates=["date"])
    df_food_crisis = pd.read_csv("data/food_crises_cleaned.csv", parse_dates=['date'])

    districts = list(region for region in df_food_crisis["district"].unique())
    for i in range(len(districts)):
        districts[i] = districts[i].replace("Center", "").replace("South", "").replace("North", "").replace("East", "").replace("West", "").strip()
    districts = list(dict.fromkeys(districts))
    df_news = df_news.drop(columns=['summary', 'lat', 'lng'])
    df_news['district'] = np.nan
    df_news['district'] = df_news['district'].astype('object')
    df_news_districted = pd.DataFrame(columns=df_news.columns)
    # Initialize an empty list to collect rows
    matched_rows = []

    # Iterate through the rows of df_news
    for i in tqdm(range(len(df_news)), desc="Preparing data for the Pie Chart"):
        text = df_news.iloc[i, 1].lower()
        matched = False

        for district in districts:
            if (district.lower() in text) or (text in district.lower()):
                matched_row = df_news.iloc[i].copy()  # Create a copy of the matching row
                matched_row.iloc[-1] = district
                matched_rows.append(matched_row)
                matched = True

        if not matched:
            no_match_row = df_news.iloc[i].copy()  # Create a copy of the original row
            no_match_row.iloc[-1] = "Not assigned"
            matched_rows.append(no_match_row)



    df_news_districted = pd.DataFrame(matched_rows, columns = df_news.columns).reset_index(drop=True)
    monthly_avg_art = df_news_districted[['district']].groupby(['district']).size().reset_index(name='count').sort_values(by='count', ascending=False).iloc[0:10,]

    # Set the threshold for grouping into "other"
    threshold = sum(monthly_avg_art['count']/ 100*2.3)  # Less than 2% will be grouped into "other"

    # Identify the groups to group into "other"
    small_groups = monthly_avg_art[monthly_avg_art['count'] < threshold]
    small_groups_total = small_groups['count'].sum()

    # Combine small groups into "other"
    monthly_avg_art.loc[monthly_avg_art['count'] < threshold, 'district'] = 'Other regions'
    # monthly_avg_art.loc[monthly_avg_art['district'] == 'Other', 'count'] += small_groups_total

    monthly_avg_art = monthly_avg_art.groupby(['district']).sum()

    # Create a pie chart using Seaborn
    plt.figure(figsize=(6, 6))
    sns.set_palette('Set3')


    # Plot the pie chart
    plt.pie(monthly_avg_art['count'], labels=monthly_avg_art.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
    plt.title('Distribution of News Articles', fontsize=16, fontweight='bold')

    plt.show()