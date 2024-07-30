import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Activation, Dropout
from keras._tf_keras.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['!', '?', '.', ',']
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.tokenize(pattern)
        words.extend(word)
        documents.append(intent['tag'])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
