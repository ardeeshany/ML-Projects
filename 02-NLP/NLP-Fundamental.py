# ====================

# sample text from waiting for goddot
wfg = "Estragon: We always find something, eh Didi, to give us the impression we exist? Vladimir: Yes, yes, we're magicians."

## simple regex

import pandas as pd
import matplotlib.pyplot as plt
import re
re.split(pattern=" ", string=wfg)
re.findall(pattern="\w+s\s.*?", string=wfg) # non-greedy by crazy
re.sub(pattern="(yes|Yes)", repl="no", string=wfg) # or
f"Hi {wfg[0:7]!r}, I am me at {pd.to_datetime('2022-12-01'):%Y-%B}"  # f string with quotes; pd.Timestamp.now(); D['col'].dt.year


## Token
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize, TweetTokenizer
my_words = word_tokenize(wfg)
my_words = [w.lower() for w in my_words]
plt.hist([len(w) for w in my_words]); plt.show()
regexp_tokenize(text=wfg, pattern="(\w+s)\s") # token on pattern

# ====================

## bag of words
from collections import Counter
Counter(my_words).most_common(2)  # truncated on the second


## preprocessing
from nltk.corpus import stopwords
stopwords.words('english')
[w for w in my_words if w not in stopwords.words('english')]

from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
lemm.lemmatize('worst', pos = "a") # remove polural s; change to "a"

## dictionary with gensim
### a DL algorith to create word vectors -> corpora/dictionary
monol = """Sometimes I feel it coming all the same. Then I go all queer. (He takes off his
hat, peers inside it, feels about inside it, shakes it, puts it on again.) How shall I
say? Relieved and at the same time . . . (he searches for the word) . . . appalled.
(With emphasis.) AP-PALLED. (He takes off his hat again, peers inside it.) Funny.
(He knocks on the crown as though to dislodge a foreign body, peers into it again,
puts it on again.) Nothing to be done."""

my_tokens = [w.lower() for w in word_tokenize(monol)]

from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(my_sents)

dictionary.token2id              # gives an id to each unique word
dictionary.token2id.get('and')
dictionary.get(32)

dictionary.doc2bow(word_tokenize("I made it done done though"))  # compare a new doc with the dictionary ids to create its bow
my_sents = [word_tokenize(s.lower()) for s in sent_tokenize(monol)]

corpus = [dictionary.doc2bow(s) for s in my_sents]
corpus # a list shows bow based on id for each document


## tf-idf with gensim
### term frequency - inverse document frequency
from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus)
tfidf[corpus[1]]
dictionary.get(10)


# ======================
