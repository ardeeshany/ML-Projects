# Definitions:
## corpus = a collection of texts
## token separating a text into smaller units
## id of token = transfer tokens of a text into seq numbers
## vocabulary = list of unique tokens (e.g., words)
## dictionary = list of vocabulary with indexes

# Steps:
## 1. creates tokens from the corpus
## 2. generates vocabulary
## 3. creates ids of tokens
## 4. defines a dictionary of vocabulary and ids (key: value)
## 5. each doc of corpus to list of indexes
## 6. Pre-processing: padding to make the lists equal size


# NLP main application

## Many to one: 
### 1. Snetiment Analysis 
### 2. Classification -> e.g., in recom system

## Many to Many: 
### 1. Text generation -> e.g., in recom system 
### 2. Machine Translation -> Two engins : Encoder & Decoder
### 3. Language model -> computes the probability of the whole sentence

# RNN: recurrent neural network -> inputs + keep states + share weights -> smaller number of parameters

# Lnaguage model -> prob of each word in a sent to be appeared in the specific order
## models:
### 1. unigram
### 2. N-gram
### 3. skip gram
### 4. Neural Network with `softmax` function


# --------------------------------------------------

import keras.datasets

# Retrieve the training sequences.
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(seed=113)
# Retrieve the word index file mapping words to indices
word_index = keras.datasets.imdb.get_word_index()
# Reverse the word index to obtain a dict mapping indices to words
inverted_word_index = dict((i, word) for (word, i) in word_index.items())
# Decode the first sequence in the dataset
" ".join(inverted_word_index[i] for i in x_train[0])

import matplotlib.pyplot as plt
plt.hist(list(map(len, x_train)))
plt.show()

from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=500, padding='pre')
x_test = pad_sequences(x_test, maxlen=500, padding='pre')

# Toy model -> flip coin
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim = 500))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
y_pred = list(map(lambda x: 0 if(x<0.5) else 1, model.predict(x_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# one step better model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN

# Sime RNN ---------------------------------------------


vocab_ind_max = max(list(map(max, x_train))) # = 88,586
model = Sequential()
model.add(Embedding(input_dim=vocab_ind_max + 1, output_dim=4, input_length=500))
# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
# model.add(LSTM(32, return_sequences=True))
# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(SimpleRNN(4))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

y_pred = list(map(lambda x: 0 if(x<0.5) else 1, model.predict(x_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Tokenize yourself with keras --------------------------------------
import nltk
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
book = " ".join(list(emma))
# import re
# book = re.sub(pattern="\s\.", repl=".", string=book)
# book = re.sub(pattern="\s\,", repl=",", string=book)
# corpus = re.split(pattern="\.", string=book)
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
corpus = sent_tokenize(book)
corpus = corpus[0:100]

from gensim.corpora.dictionary import Dictionary



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)  # first fit to find the indexes for each character
sents_of_index = tokenizer.texts_to_sequences(corpus) # give each chr the corresponding index
print(pad_sequences(sequences=sents_of_index,maxlen=115))

# Tokenize yourself with nltk ---------------------------------------





