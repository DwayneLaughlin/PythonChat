import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Activation, Dropout
from keras._tf_keras.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle 
with open("Classes.pkl", "wb") as classesFile:
    classes_file = pickle.load(classesFile)
with open("Words.pkl", "wb") as wordsFile:
    words_file = pickle.load(wordsFile)



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

# creating training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    lemmatizer.lemmatize(word.lower() for word in pattern_words)
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append(bag, output_row)

    random.shuffle(training)
    training = np.array(training)

    train_patterns = list(training[:,0])
    train_intents = list(training[:,1])
    print("Training data created")

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_patterns[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_intents[0]),activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_patterns), np.array(train_intents), epochs=200, batch_size=5, verbose=1)

    model.save('chatbot_model.h5', hist)

    print("model created")

