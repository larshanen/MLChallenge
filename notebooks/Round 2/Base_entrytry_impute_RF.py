#%%
import pandas as pd
import logging
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor  # Add this import
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from fancyimpute import IterativeImputer

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train_df = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test_df = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")

    # Adding a feature for abstract length
    train_df['abstract_length'] = train_df['abstract'].apply(lambda x: len(str(x)))
    test_df['abstract_length'] = test_df['abstract'].apply(lambda x: len(str(x)))

    # Adding a feature for editor count
    train_df['editor_count'] = train_df['editor'].apply(lambda x: len(x) if x is not None else np.nan)
    test_df['editor_count'] = test_df['editor'].apply(lambda x: len(x) if x is not None else np.nan)

    # Adding a feature for author count
    train_df['author_count'] = train_df['author'].apply(lambda x: len(x) if x is not None else np.nan)
    test_df['author_count'] = test_df['author'].apply(lambda x: len(x) if x is not None else np.nan)

    # Adding a feature for publisher count
    train_df['publisher_count'] = train_df['publisher'].apply(lambda x: len(x) if x is not None else np.nan)
    test_df['publisher_count'] = test_df['publisher'].apply(lambda x: len(x) if x is not None else np.nan)

    # Specify the columns to be converted to numeric
    change_to_be_numeric = ["author_count", "editor_count", "publisher_count"]

    # Convert specified columns to numeric and fill missing values with medians
    train_df[change_to_be_numeric] = train_df[change_to_be_numeric].apply(pd.to_numeric, errors='coerce')
    train_df[change_to_be_numeric] = train_df[change_to_be_numeric].fillna(train_df[change_to_be_numeric].median())

    # Splitting validation
    train_df, val = train_test_split(train_df, stratify=train_df['year'], test_size=0.2, random_state=123)

    # List of columns to impute (exclude the columns you want to skip)
    columns_to_impute = ['author_count', 'editor_count', 'publisher_count']

    # Separate the columns to impute from the rest of the DataFrame
    data_to_impute = train_df[columns_to_impute]

    # Define the imputer
    imputer = IterativeImputer(max_iter=10, random_state=0)

    # Fit and transform the data
    imputed_data = imputer.fit_transform(data_to_impute)

    # Update the original DataFrame with the imputed values
    train_df[columns_to_impute] = imputed_data

    categorical_cols = ['ENTRYTYPE']
    numerical_cols = ['abstract_length', 'author_count', 'editor_count', 'publisher_count']

    featurizer = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'title'),
            ('num', StandardScaler(), numerical_cols),
            ('cat', FunctionTransformer(lambda x: pd.get_dummies(x, columns=categorical_cols, drop_first=True)),
             categorical_cols)
        ],
        remainder='drop'
    )
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    rf_regressor = make_pipeline(featurizer, RandomForestRegressor())  # Change here
    logging.info("Fitting models")
    dummy.fit(train_df.drop('year', axis=1), train_df['year'].values)
    rf_regressor.fit(train_df.drop('year', axis=1), train_df['year'].values)  # Change here
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    err = mean_absolute_error(val['year'].values, rf_regressor.predict(val.drop('year', axis=1)))  # Change here
    logging.info(f"Random Forest regressor MAE: {err}")  # Change here
    logging.info("Predicting on test")
    pred = rf_regressor.predict(test_df)  # Change here
    test_df['year'] = pred
    logging.info("Writing prediction file")
    test_df.to_json("predicted.json", orient='records', indent=2)

main()

# %%
#INFO:root:Loading training/test data
#INFO:root:Fitting models
#INFO:root:Evaluating on validation data
#INFO:root:Mean baseline MAE: 7.81040589188589
#INFO:root:Random Forest regressor MAE: 3.3524981320060077
#INFO:root:Predicting on test
#INFO:root:Writing prediction file