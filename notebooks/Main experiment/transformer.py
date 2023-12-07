import pandas as pd
import numpy as np
import logging
import json
from sklearn.neighbors import KNeighborsRegressor
from sentence_transformers import SentenceTransformer

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading train/test data")

    train = pd.DataFrame.from_records(json.load(open('../../data/train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('../../data/test.json'))).fillna("")

    full_set = pd.concat([train, test])

    # Step 1: Concatenate 'title' and 'abstract'
    logging.info("Combining title and abstract columns")
    full_set['combined_text'] = full_set['title'] + ' ' + full_set['abstract']

    # Step 2: Obtain sentence embeddings for the combined text
    logging.info("Loading SentenceTransformer all-mpnet-base-v2 model")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Extract 768 dimensional sentence embeddings for train/test data")
    embeddings = model.encode(full_set['combined_text'].tolist(), convert_to_tensor=True)

    # Step 3: Create a new DataFrame with the sentence embeddings and 'year' as the target variable
    logging.info("Convert vectors to train dataframe")
    embedding_columns = [f'dim_{i+1}' for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings.numpy(), columns=embedding_columns)

    x = len(train)
    train_df = embedding_df.iloc[:x, :]
    test_df = embedding_df.iloc[x:, :]

    # Step 4: Get nearest neighbors
    logging.info("Predict years")
    kNN = KNeighborsRegressor(n_neighbors=8, metric='cosine', weights= 'distance')
    kNN.fit(train_df, pd.to_numeric(train['year'].values))
    pred = kNN.predict(test_df)
    test['year'] = pred
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)

main()