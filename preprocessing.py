import pandas as pd
import os

from helper_functions import create_lag_df, get_relevant_topics, create_news_features
from bertopic import BERTopic
import statsmodels.api as sm

def prepare_articles_features(data_dir, features):

        if features == "Standard":
            # Check if the csv for processed template articles features exists, if not generate it
            if not os.path.exists(data_dir + "articles_topics_template.csv"):
                # Read the data and perform preprocessing
                df = pd.read_csv("data/articles_summary_cleaned.csv", parse_dates=["date"]) # Read data into 'df' dataframe

                docs = df["summary"].tolist() # Create a list containing all article summaries

                df.head() # Show first 5 dataframe entries

                bertopic = BERTopic(language="english", calculate_probabilities=True, verbose=True) # Initialize the BERTopic model

                bertopic.fit_transform(docs) # Fit the model to the list of article summaries
                bertopic.save("southsudan_model") # Save the trained model as "southsudan_model"

                # Get the top 10 topics related to the keywords 'hunger' and 'food insecurity'
                relevant_topics = get_relevant_topics(bertopic_model = bertopic, keywords=['hunger', 'food insecurity'], top_n=10)
                topic_ids = [el[0] for el in relevant_topics] # Create seperate list of topic IDs
                df["hunger"] = [t in topic_ids for t in bertopic.topics_] # Add boolean column to df if topic in list of relevant topics

                # Get the top 10 topics related to the keywords 'refugees' and 'displaced'
                relevant_topics = get_relevant_topics(bertopic_model = bertopic, keywords=['refugees', 'displaced'], top_n=10)
                topic_ids = [el[0] for el in relevant_topics] # Create seperate list of topic IDs
                df["refugees"] = [t in topic_ids for t in bertopic.topics_] # Add boolean column to df if topic in list of relevant topics

                # Get the top 10 topics related to the keyword 'humanitarian'
                relevant_topics = get_relevant_topics(bertopic_model = bertopic, keywords=['humanitarian'], top_n=10)
                topic_ids = [el[0] for el in relevant_topics] # Create seperate list of topic IDs
                df["humanitarian"] = [t in topic_ids for t in bertopic.topics_] # Add boolean column to df if topic in list of relevant topics

                # Get the top 10 topics related to the keywords 'conflict', 'fighting', and 'murder'
                relevant_topics = get_relevant_topics(bertopic_model = bertopic, keywords=['conflict', 'fighting', 'murder'], top_n=10)
                topic_ids = [el[0] for el in relevant_topics] # Create seperate list of topic IDs
                df["conflict"] = [t in topic_ids for t in bertopic.topics_] # Add boolean column to df if topic in list of relevant topics

                original_df = pd.read_csv("data/articles_summary_cleaned.csv", parse_dates=["date"])

                # Combine article summaries with the newly created features
                df = original_df.merge(
                    df[["summary", "hunger", "refugees", "humanitarian", "conflict"]],
                    how="left",
                    left_on="summary",
                    right_on="summary",
                )
                df.to_csv(data_dir + "articles_topics_template.csv", index=False) # Save DataFrame to articles_topics.csv



def prepare_dataset(data_dir = "data/", features = "Standard"):

    # Preprocessing for standard tabular and article features
    if features == "Standard":

        df = pd.read_csv(data_dir + "food_crises_cleaned.csv") # Read data into DataFrame
        df["date"] = pd.to_datetime(df["year_month"], format="%Y_%m") # Create date column
        df.set_index(["date", "district"], inplace=True) # Set index

        # Create several lagged columns to use as explanatory variables for the model

        df = create_lag_df(df, ['count_violence', 'ndvi_anom'], 3, rolling=6) # 3-month-lagged rolling mean window of size 6
        df = create_lag_df(df, ['food_price_idx'], 3, difference=True, rolling=6) # difference of the 3-month-lagged rolling mean window of size 6
        df = create_lag_df(df, ['ipc'], 1, dropna=True) # 1-month-lag
        df = create_lag_df(df, ['ipc'], 2, dropna=True) # 2-month-lag
        df = create_lag_df(df, ['ipc'], 3, dropna=True) # 3-month-lag

        df.sort_index(level=0, inplace=True) # Sort DataFrame by date
        df = df.iloc[df['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column

        prepare_articles_features(data_dir, features) # generate articles_topics_template.csv

        news_df = pd.read_csv(data_dir + "articles_topics_template.csv") # Read news data into DataFrame

        # Create date column
        news_df["date"] = pd.to_datetime(
            pd.to_datetime(news_df["date"], format="%Y-%m-%d").dt.strftime("%Y-%m"),
            format="%Y-%m")
        
        # create news articles features
        news_features = create_news_features(["hunger", 'refugees', 'conflict', 'humanitarian'], news_df)        

        df.sort_index(level=0, inplace=True) # Sort DataFrame by date
        df = df.iloc[df['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column
        df = df.join(news_features, how="left") # Join df with created news features

        X = df.iloc[:, -10:] # Define explanatory variables
        X = sm.add_constant(X) # Add constant column of 1s for intercept
        y = df[["ipc"]] # Define target data

    return X, y
    