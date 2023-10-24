# Standard imports
import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm

# Custom imports
from helper_functions import create_lag_df, get_relevant_topics, create_news_features
from bertopic import BERTopic
import statsmodels.api as sm

def article_mapping(data_dir = "data/"):
    df_news = pd.read_csv(data_dir + "articles_topics_improved.csv", parse_dates=["date"])
    df_food_crisis = pd.read_csv(data_dir + "food_crises_cleaned.csv", parse_dates=['date'])

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
    for i in tqdm(range(len(df_news)), desc= "Article Mapping"):
        text = df_news.iloc[i, 1].lower()
        matched = False

        for district in districts:
            if (district.lower() in text) or (text in district.lower()):
                matched_row = df_news.iloc[i].copy()  # Create a copy of the matching row
                matched_row.iloc[-1] = district
                matched_rows.append(matched_row)
                matched = True

        if not matched:
            for district in districts:
                no_match_row = df_news.iloc[i].copy()  # Create a copy of the original row
                no_match_row.iloc[-1] = district
                matched_rows.append(no_match_row)


    df_news_districted = pd.DataFrame(matched_rows, columns = df_news.columns).reset_index(drop=True)
    df_news_districted.to_csv(data_dir + "df_news_districted_improved.csv")

def prepare_articles_features(data_dir, features):

    # Read the data and perform preprocessing
    df = pd.read_csv(data_dir + "articles_summary_cleaned.csv", parse_dates=["date"]) # Read data into 'df' dataframe

    docs = df["summary"].tolist() # Create a list containing all article summaries

    df.head() # Show first 5 dataframe entries

    if os.path.exists('southsudan_model'):
        bertopic = BERTopic.load('southsudan_model')
    else:
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

    original_df = pd.read_csv(data_dir + "articles_summary_cleaned.csv", parse_dates=["date"])
    if features == "Standard":

        # Combine article summaries with the newly created features
        df = original_df.merge(
            df[["summary", "hunger", "refugees", "humanitarian", "conflict"]],
            how="left",
            left_on="summary",
            right_on="summary")

        df.to_csv(data_dir + "articles_topics_template.csv", index=False) # Save DataFrame to articles_topics.csv    

    elif features == "Improved":

        # New keywords
        final_keywords = ['Governance', 'Diplomacy', 'Crisis', 'Security', 'Society', 'Health', 'Development',
                        'Education', 'Survival', 'International', 'Opposition', 'Welfare', 'Media', 'Leadership']


        for keyword in final_keywords:
            # Get relevant topics for the current keyword
            relevant_topics = get_relevant_topics(bertopic_model=bertopic, keywords=[keyword], top_n=10)
            # Extract topic IDs
            topic_ids = [el[0] for el in relevant_topics]
            # Add a boolean column to df for the current keyword
            df[keyword] = [t in topic_ids for t in bertopic.topics_]

        df = original_df.merge(
        df[["summary", "hunger", "refugees", "humanitarian", "conflict"] + final_keywords],
        how="left",
        left_on="summary",
        right_on="summary")

        df.to_csv(data_dir + "articles_topics_improved.csv", index=False) # Save DataFrame to articles_topics.csv

        if not os.path.exists(data_dir + "df_news_districted_improved.csv"):
            article_mapping(data_dir)



def prepare_dataset(data_dir = "data/", features = "Standard"):

    df = pd.read_csv(data_dir + "food_crises_cleaned.csv") # Read data into DataFrame
    df["date"] = pd.to_datetime(df["year_month"], format="%Y_%m") # Create date column
    df.set_index(["date", "district"], inplace=True) # Set index

    # Preprocessing for standard tabular and article features
    if features == "Standard":

        # Create several lagged columns to use as explanatory variables for the model

        df = create_lag_df(df, ['count_violence', 'ndvi_anom'], 3, rolling=6) # 3-month-lagged rolling mean window of size 6
        df = create_lag_df(df, ['food_price_idx'], 3, difference=True, rolling=6) # difference of the 3-month-lagged rolling mean window of size 6
        df = create_lag_df(df, ['ipc'], 1, dropna=True) # 1-month-lag
        df = create_lag_df(df, ['ipc'], 2, dropna=True) # 2-month-lag
        df = create_lag_df(df, ['ipc'], 3, dropna=True) # 3-month-lag

        df.sort_index(level=0, inplace=True) # Sort DataFrame by date
        df = df.iloc[df['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column


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

        for i in ["hunger", 'refugees', 'conflict', 'humanitarian']:
            df[i] = df[i].fillna(0)

        X = df.iloc[:, -10:] # Define explanatory variables
        X = sm.add_constant(X) # Add constant column of 1s for intercept
        y = df[["ipc"]] # Define target data

    elif features == "Improved":
        # Create several lagged columns to use as explanatory variables for the model

        df = create_lag_df(df, ['count_violence', 'ndvi_anom',
                                'ruggedness_mean','pop',
                                'cropland_pct', 'sum_fatalities',
                                'et_anom', 'rain_anom'], 3, rolling=6) # 3-month-lagged rolling mean window of size 6
        df = create_lag_df(df, ['food_price_idx'], 3, difference=True, rolling=6) # difference of the 3-month-lagged rolling mean window of size 6
        df = create_lag_df(df, ['ipc'], 1, dropna=True) # 1-month-l ag

        df.sort_index(level=0, inplace=True) # Sort DataFrame by date
        df = df.iloc[df['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column

        prepare_articles_features(data_dir, features) # generate articles_topics_template.csv

        news_df = pd.read_csv(data_dir + "df_news_districted_improved.csv") # Read news data into DataFrame

        # Create date column
        news_df["date"] = pd.to_datetime(
            pd.to_datetime(news_df["date"], format="%Y-%m-%d").dt.strftime("%Y-%m"),
            format="%Y-%m")
        news_df = news_df.drop(columns=['Unnamed: 0'])

        # country wide news features
        news_features = create_news_features(news_df.columns[2:-1], news_df)
        df = df.join(news_features, how="left") # Join df with created news features

        for i in news_features.columns:
            df[i] = df[i].fillna(0)
        
        df.sort_index(level=0, inplace=True) # Sort DataFrame by date
        df = df.iloc[df['ipc'].notnull().argmax():].copy() # Drop rows until first notna value in ipc column

        X = df.iloc[:, -28:] # Define explanatory variables
        X = sm.add_constant(X) # Add constant column of 1s for intercept
        y = df[["ipc"]] # Define target data

        y = y[X['ipc_lag_1'].notnull()]
        X = X[X['ipc_lag_1'].notnull()]

    return X, y
    