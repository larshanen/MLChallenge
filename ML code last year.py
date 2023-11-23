# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:06:54 2022

@author: mirja
"""

#importing all the necessary things 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

#downloading all the necessary things 
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# step1:  Loading our datasets
train = "train.json"
test = "test.json"
train = pd.read_json(train)
test = pd.read_json(test)


#subset the data
train = train[0:2000]


# Step 2: Preprocessing train data 
#step 2.1: Tokenization
#Tokenization means cleaning data in such a way that we only keep a plain text
#Source:(Dwivedi, 2020) https://analyticsindiamag.com/how-to-predict-authors-features-from-his-writings/

train["abstract"] = train["abstract"].str.replace('[^A-Za-z]',' ') #Everything not in the alphabet is converted to a space
train["abstract"] = train["abstract"].str.lower() #Make everything lower case
train["abstract"] = train["abstract"].str.strip() #Remove all leading and trailing spaces (the spaces used at the beginning and the end)
train["abstract"] = train["abstract"].str.split() #Makes it a list with indivual words as list elements using a space as a separator


#step 2.2: Stopwords
#Stopwords are words without an intrinsic value. They are deleted (i.e. the, and etc.)
#We imported the stopwords list from corpus from nltk, set to english to use the english version for this document.
#If the word is not in stopwords, it gets joined as a sentence again. The sentence is a sentence without stopwords.
#Source: (Dwivedi, 2020) https://analyticsindiamag.com/how-to-predict-authors-features-from-his-writings/
stopwordsenglish = set(stopwords.words('english')) #(Dwivedi, 2020)
train.abstract = train.abstract.apply(lambda ourlist: ' '.join([word for word in ourlist if word not in stopwordsenglish])) ##(Dwivedi, 2020)


#step 2.3: lemmatization
#words are transferred to their root word. e.g. from children to child.
#Source: https://www.appsloveworld.com/pandas/100/296/lemmatization-pandas-python
lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')

def pos_tag_per_word(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatizer_function(sentence):
    #tokenize the sentence, use the POS-tag function to find the POS_tag per word
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], pos_tag_per_word(x[1])), nltk_tagged)
    lemmatized_row = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no tag, keep the word as is.
            lemmatized_row.append(word)
        else:
            #if there is a tag, lemmatize according to the pos_tag
            lemmatized_row.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_row)

train['abstract'] = train['abstract'].apply(lambda x: lemmatizer_function(x))
print(train["abstract"].head())

#step 3: pre-processing test data. We apply the three preprocessing steps (tokenization, stopwords and lemmatization) also to the test data
#Tokenization:
test["abstract"] = test["abstract"].str.replace('[^A-Za-z]',' ') #Everything not in the alphabet is converted to a space
test["abstract"] = test["abstract"].str.lower() #Make everything lower case
test["abstract"] = test["abstract"].str.strip() #Remove all leading and trailing spaces (the spaces used at the beginning and the end)
test["abstract"] = test["abstract"].str.split() #Makes it a list with indivual words as list elements using a space as a separator

#stopwords:
stopwordsenglish = set(stopwords.words('english')) #(Dwivedi, 2020)
test.abstract = test.abstract.apply(lambda ourlist: ' '.join([word for word in ourlist if word not in stopwordsenglish])) ##(Dwivedi, 2020)

#Lemmatization
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

test['abstract'] = test['abstract'].apply(lambda x: lemmatize_sentence(x))
print(test["abstract"].head())


#step 4: splitting the data

# step 4.1 Defining our y variable
ylabels = train["authorId"]

#  step 4.2 Splitting the training data into training and validation
#Source: https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/
#Therefore, we use random state = 42 to make sure we have the same split in the data each time.
x_train, x_val, y_train, y_validation = train_test_split(train, ylabels, test_size=0.33,random_state=42)


# Defining our x variables
x_abstract_train = x_train["abstract"]
x_abstract_val = x_val["abstract"]
x_abstract_test = test["abstract"]

# #We extract word counts from text
#Source: tutorial 5, ML course. https://tilburguniversity.instructure.com/courses/11878/modules
#A countvectorizer counts all words/wordsequences of length 1, 2 and 3.
vec = CountVectorizer(analyzer='word', ngram_range=(1,3), lowercase=True, max_features=300000)

# Filling the vocabulary
vec.fit(x_abstract_train)
vec.fit(x_abstract_val)
vec.fit(x_abstract_test)

vocab = vec.get_feature_names()
print(vocab)

vec2 = CountVectorizer(analyzer='word', ngram_range=(1,3), lowercase=True, vocabulary=vocab)

x_features_train = vec2.fit_transform(x_abstract_train)
x_features_val = vec2.fit_transform(x_abstract_val)
x_features_test = vec2.fit_transform(x_abstract_test)


# Fit model without hyperparametertuning
model = SGDClassifier(loss='log', random_state=123) #check for different loss functions and why random state 123
model.fit(x_features_train, y_train) #epochs checken
y_pred = model.predict(x_features_val)
print("{:.3}".format(accuracy_score(y_validation, y_pred)))

# step 5 Hyperparameter tuning
#Source: https://www.appsloveworld.com/pandas/100/296/lemmatization-pandas-python
params = {
    "loss" : ["log"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1,1],
    "penalty" : ["l2", "l1", "elasticnet", "none"],
}

clf = SGDClassifier(max_iter=1000)
grid = GridSearchCV(clf, param_grid=params, cv=3)

grid.fit(x_features_train, y_train)

print(grid.best_params_)


#We fit the model based on the outcomes of the hyperparameter tuning process.
#Source: tutorial 5, ML course. https://tilburguniversity.instructure.com/courses/11878/modules
model = SGDClassifier(loss='log', random_state = 123, alpha = 0.1, penalty = 'l2')
model.fit(x_features_train, y_train)
y_pred = model.predict(x_features_val)
print("{:.3}".format(accuracy_score(y_validation, y_pred)))


# Step 6: Use the model to predict the authorId in the test set
#help from tutorial 
#notebooks from data processing as source (notebook 11: working with text files)
y_pred_test = model.predict(x_features_test)
print(y_pred_test)

# Linking the corresponding paperId to the predicted authorId
paperId = test["paperId"]
df = pd.DataFrame(y_pred_test, columns = ["authorId"])
df_pred_paper_auth = df.assign(paperId =paperId)

# Creating a file in which the predictions can be found
pred_path = pd.DataFrame(df_pred_paper_auth)
pred_path = pred_path.iloc[:, [1,0]]


#turn the authorId into string so it has " " around it 
pred_path['authorId'] =pred_path['authorId'].apply(str)

#turn the prediction into a json file 
pred_path_new = pred_path.to_json(orient='records')

fp = open(r'predicted.json', 'w' )
fp.write(pred_path_new)
fp.close()