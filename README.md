# South Sudan Project 
## Code description
There are 3 .py files, 1 jupyter notebook and the necessary data (`data/all_africa_southsudan.csv`, `data/articles_summary_cleaned.csv`, `data/food_crises_cleaned.csv`) to execute the analysis.

`helper_functions.py` - contains functions for repeated subroutines and generation of graphs sections 2 and 3 of the jupyter notebook

`preprocessing.py` - contains functions to generate template (features = "Standard") and the improved dataset (features = "Improved") for IPC prediction

`engine.py` - contains the function to train, test and evaluate the models (model_type = "OLS"/"Ridge"/"XGboost"/"NN")

In order to run the analysis, first make sure that all the libraries in `requirements.txt` are installed, `DC3-Group-13` is selected as CWD, and then execute `src.ipynb`.

## Requirements
To install the requirements open Terminal (macOS)/Command Prompt (Windows) and run `pip install -r requirements.txt`. If you create a new environment in PyCharm, an icon should appear to install requirements. The code runs with Python 3.10.12.

Required libraries:
- bertopic == 0.15.0
- pandas == 1.4.4
- geopandas == 0.13.2
- matplotlib == 3.7.2
- seaborn == 0.12.2
- numpy == 1.24.3
- statsmodels == 0.14.0
- scikit-learn == 1.3.0
- matplotlib == 3.7.2
- xgboost == 2.0.0
- torch == 2.1.0
- seaborn == 0.12.2
- tqdm == 4.65.0
- ipywidgets == 8.1.1

## Troubleshooting

If you encounter any issues while running the notebooks, try the following:
- check that you have all the necessary libraries installed and the correct versions of them
- check your Python version. In principle, the code should work with any Python versions higher than 3.10.12. If this is not the case, create a virtual environment that uses Python 3.10.12.
