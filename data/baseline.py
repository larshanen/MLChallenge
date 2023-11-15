import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)
    featurizer = ColumnTransformer(
        transformers=[("title", CountVectorizer(), "title")],
        remainder='drop')
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    ridge = make_pipeline(featurizer, Ridge())
    logging.info("Fitting models")
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    ridge.fit(train.drop('year', axis=1), train['year'].values)
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    err = mean_absolute_error(val['year'].values, ridge.predict(val.drop('year', axis=1)))
    logging.info(f"Ridge regress MAE: {err}")
    logging.info(f"Predicting on test")
    pred = ridge.predict(test)
    test['year'] = pred
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)
    
main()

