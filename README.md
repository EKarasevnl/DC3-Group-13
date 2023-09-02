# South Sudan Project 
## Code description
You are provided with two notebooks and the necessary data to help you start the South Sudan project. First, run _topic_modelling.ipynb_. In this notebook, a BERTopic model is fit on _articles_summary_cleaned.csv_, then four categories/keywords (hunger, refugees, humanitarian, and conflict) are defined. These categories are used for categorising articles (or none if the article doesn’t match any of the categories) and thus creating features from the news articles. Secondly, run the _predictions.ipynb_ notebook for some very basic data exploration, along with fitting several linear models on the data, with- and without the news features. The notebook uses the _food_crises_cleaned.csv_ dataset and the csv file obtained from the  _topic_modelling.ipynb_ notebook.

## Requirements
To install the requirements open Terminal (macOS)/Command Prompt (Windows) and run pip install -r requirements.txt. If you create a new environment in PyCharm, an icon should appear to install requirements. The code runs with Python 3.9.16.

Required libraries:
- bertopic == 0.15.0 
- pandas == 1.4.4 
- geopandas == 0.13.2 
- matplotlib == 3.7.2 
- seaborn == 0.12.2
- statsmodels == 0.14.0 

## Troubleshooting

If you encounter any issues while running the notebooks, try the following:
- check that you have all the necessary libraries installed and the correct versions of them
- check your Python version. In principle, the code should work with any Python versions higher than 3.9.16. If this is not the case, create a virtual environment that uses Python 3.9.16.
