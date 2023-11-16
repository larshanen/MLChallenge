#%%
import json
import pandas as pd

# Load training data
#with open('train.json', 'r') as f:
#    train_data = json.load(f)

# Load test data
#with open('test.json', 'r') as f:
#    test_data = json.load(f)
data = pd.read_json('train.json')
print(data.head())

# Convert to pandas DataFrame for easier manipulation
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Explore the data
print(train_df.head())
print(train_df.info())
print(train_df.tail())
print(train_df[["year","abstract"]])

middle_rows = len(train_df) // 2
print(train_df.iloc[middle_rows - 5: middle_rows + 5][["year", "abstract"]])

#check year 2017 - 2022
start_year = 2017
end_year = 2023

# Convert "year" column to integers
train_df["year"] = train_df["year"].astype(int)

# Filter based on the specified range
filtered_df = train_df[(train_df["year"] >= start_year) & (train_df["year"] <= end_year)]

# Print the desired columns
print(filtered_df[["year", "abstract"]])

#check the max of 'year'
train_df["year"].max()
#%%
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

with open('train.json', 'r') as f:
    train_data = json.load(f)

train_df = pd.DataFrame(train_data)

print(train_df.columns)

# %%
#look like some link with the
#turn abstract to be integer
train_df['abstract_length'] = train_df['abstract'].apply(lambda x: len(str(x)))
print(train_df['abstract_length'])
# Handle missing values (if any)
train_df.fillna(0, inplace=True)
