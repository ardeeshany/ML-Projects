# Definitions:
## corpus = a collection of docs (texts)
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

# Pre-processing
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
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)
y_pred = list(map(lambda x: 0 if(x<0.5) else 1, model.predict(x_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# one step better model ================================
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN

# Sime RNN ---------------------------------------------


vocab_ind_max = max(list(map(max, x_train))) # = 88,586
model = Sequential()
# Embedding layer enables us to convert each word into a fixed length vector of defined size! We use it to reduce dimension
model.add(Embedding(input_dim=vocab_ind_max + 1, output_dim=4, input_length=500))
# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
# model.add(LSTM(32, return_sequences=True))
# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(SimpleRNN(8))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

y_pred = list(map(lambda x: 0 if(x<0.5) else 1, model.predict(x_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Tokenize yourself with nltk and gensim  --------------------------------------
import nltk
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
book = " ".join(list(emma))
# import re
# book = re.sub(pattern="\s\.", repl=".", string=book)
# book = re.sub(pattern="\s\,", repl=",", string=book)
# corpus = re.split(pattern="\.", string=book)

# 1. create the corpus of docs
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
corpus = sent_tokenize(book)
corpus = corpus[0:100]

# 2. convert each docs to a list of tokens
tokenize_docs = [word_tokenize(doc.lower()) for doc in corpus]


# 3 from gensim.corpora.dictionary import Dictionary
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(tokenize_docs)
dictionary.id2token
dictionary.token2id.get('dearest')

# 4. convert the docs of tokens to corpus of ids
ids_docs = [dictionary.doc2idx(l) for l in tokenize_docs]
ids_docs


# Tokenize yourself with keras  --------------------------------------
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)  # first fit to find the indexes for each character
docs_of_inds = tokenizer.texts_to_sequences(corpus) # give each chr the corresponding index



print(pad_sequences(sequences=docs_of_inds,maxlen=115))


# ===================================================

# Exploiding or Vanishing problems
## Using more complex cells:
### 1. GRU: Gated recurrent unit
### 2. LSTM: Long short term model
## applying gradient clipping technique
### .compile(optimizer = SGD(lr = 0.01, clipvalue = 0.3))

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Embedding

vocab_ind_max = max(list(map(max, x_train)))

model = Sequential()
model.add(Embedding(input_dim=vocab_ind_max + 1, output_dim=128, input_length = x_train.shape[1]))
model.add(LSTM(200, dropout = 0.2, return_sequences=False))#model2.add(LSTM(units=128, return_sequences=False))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')
model.fit(x=x_train,y=y_train, epochs=1)




