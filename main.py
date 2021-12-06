import random
from collections import defaultdict

import inflect
import nltk
from nltk.corpus import PlaintextCorpusReader
import re, string, unicodedata
import re
import numpy as np
import pandas as pd
nltk.download('punkt')

#importing things for evaluation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


#get data from train and test files
train = pd.read_json("train.json")
train.head()
test = pd.read_json("test.json")
test.head()

#get some baseline numbers and set up cuisines and ingredients
allIngredients = set()
for ingredients in train['ingredients']:
    allIngredients = allIngredients | set(ingredients)
len(allIngredients)

allCuisines = set(train['cuisine'])
len(allCuisines)

numberIngredients = 0
for ingredients in train['ingredients']:
    numberIngredients += len(ingredients)

    
#ingredient vocab
vocab = {}
for ingredient, ing in zip(allIngredients, range(len(allIngredients))):
    vocab[ingredient] = ing



#training

cuisineClassifier = Pipeline([
    ('vect', CountVectorizer(vocabulary=vocab)),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# method to turn data into string format
def stringifyIngredients(data):
    return [' '.join(ingredients) for ingredients in data['ingredients']]


#predict cuisine training
strTrainIngredients = stringifyIngredients(train)
cuisineClassifier.fit(strTrainIngredients, train['cuisine'])
cuisineClassifier.score(strTrainIngredients, train['cuisine'])

trainPrediction = cuisineClassifier.predict(strTrainIngredients)

#get some metrics from training to do some feature extraction
from sklearn import metrics
print(metrics.classification_report(train['cuisine'], trainPrediction, target_names = allCuisines))

#here we would put in parameter selection, but it kept breaking our code and we didn't have time to fix it
#so we took it out to ensure our code would compile


#predicting
prediction = cuisineClassifier.predict(stringifyIngredients(test))
test['cuisine'] = prediction
test.head()

#create file with all the predictions for our test set
test[['id', 'cuisine']].to_csv('predictions.csv', index=False)
