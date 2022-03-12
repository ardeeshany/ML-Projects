# ====================
# import sys
# sys.prefix
# 
# import site
# site.getsitepackages()

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
my_sents_tokens = [word_tokenize(s.lower()) for s in sent_tokenize(monol)]
dictionary = Dictionary(my_sents_tokens)

dictionary.token2id              # gives an id to each unique word
dictionary.token2id.get('and')
dictionary.get(32)

dictionary.doc2bow(word_tokenize("I made it done done though"))  # compare a new doc with the dictionary ids to create its bow

corpus = [dictionary.doc2bow(s) for s in my_sents_tokens]
corpus # a list shows bow based on id for each document


## tf-idf with gensim
### term frequency - inverse document frequency
from gensim.models.tfidfmodel import TfidfModel
tfidf = TfidfModel(corpus)
tfidf[corpus[1]]
dictionary.get(10)

# ======================

## NER (Named-Entity Recognition) with nltk
### Stanford CoreNLP library -> Needs JAVA
my_tokens = word_tokenize(monol)

import nltk
my_chunks = nltk.pos_tag(my_tokens)
print(nltk.ne_chunk(nltk.pos_tag(my_tokens)))
my_chunks = nltk.ne_chunk(nltk.pos_tag(my_tokens), binary=True)

PRP_list = []
for c in my_chunks:
  if(c[1] == 'PRP'):
    PRP_list.append(c[0])

list(set(PRP_list)) # unique values
list(set(list(map(lambda x: x.lower(), list(set(PRP_list))))))


## NER with SpaCy
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

chelsea = """ Chelsea Football Club, often referred to as Chelsea, 
is an English professional football club based in Fulham, West London. 
Founded in 1905, the club competes in the Premier League, the top division
of English football. Chelsea are among England's most successful clubs,
having won over thirty competitive honours, including six League titles
and nine international trophies. Their home ground is Stamford Bridge.[4]"""

doc = nlp(chelsea)
doc.ents

doc.ents[12].text
doc.ents[12].label_  # GPE = GeoPolitical Entity

## polyglot: Multilingual NER

from polyglot.text import Text

poet = """یکی از نثر نویسان آغاز کار قاجاریه میرزا رضی منشی الممالک، فرزند میرزا شفیع آذربایجانی، است. میرزا رضی از خانواده‌ای برخاسته، که از عقاید صوفیگری پیروی می‌کرده‌اند، ولی این امر او و پدر وی را از رسیدن به مقامات عالیه دولتی بازنداشته‌است. میرزا شفیع مستوفی نادرشاه بود و میرزا رضی پس از فوت پدر در دولت کریم خان زند و پس از او دربار آغا محمد خان قاجار منصب استیفا داشت و رسائل و نامه‌ها و فرامین عمده را به عربی و فارسی و ترکی و جغتایی تحریر می‌کرد و چون نوبت پادشاهی به فتحعلی شاه رسید در دربار او عزت و احترام بیشتری یافت و مشهور است که به هنگام سلام هم لوله قرطاس و هم خنجر الماس به کمر می‌زده است.

میرزا رضی به ترکی و گاهی به فارسی و عربی شعر می‌سرود و " بنده " تخلص می‌کرد. رضا قلی خان هدایت دو قصیده فارسی او را در مجمع الفصحاء ضبط کرده‌است.  """

txt = Text(poet)

txt.entities
txt.entities[0].tag



# =========================

## Text to supervised learning by bow as features + lable

## Count Vectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

## Run sklearn MultinomialNB()

