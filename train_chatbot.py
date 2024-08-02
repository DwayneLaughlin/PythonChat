import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Activation, Dropout
from keras._tf_keras.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle 
with open("Classes.pkl", "wb") as classes:
    classes_file = pickle.load(classes)
with open("Words.pkl", "wb") as words:
    words_file = pickle.load(words)



lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_letters = ['!', '?', '.', ',']
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# goes through each object in intents.json to find the list of patterns. The words in each pattern string are broken up into substrings(via tokenize) and added to the words list. The tag attribute is appended to the documents list and added to the classes list if it's not already there. Returns the list of documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.tokenize(pattern)
        words.extend(word)
        documents.append(intent['tag'])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(documents)

# lemmatizer = finds the root word it is passed
# words list is set to the lemmatized version of every word in words not including punctuation marks. words and classes lists are sorted. 
words = [lemmatizer.lemmatize(w.lower() for w in words if w not in ignore_letters)]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# dump saves the python object in a file
pickle.dump(words, words_file)
pickle.dump(classes, classes_file)
