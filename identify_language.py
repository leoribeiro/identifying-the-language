#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import europarl_raw
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

nltk.download('europarl_raw')


def clean_tokens(tokens):
    return [token.lower() for token in tokens if token.isalpha()]

#languages =  ['english','finnish','french','german','portuguese','swedish']
languages =  ['portuguese','english','german','french']

corpus_data = []
corpus_label = []

def chunks(l,n):
	for i in range(0,len(l),n):
		yield l[i:i+n]

for l in languages:
	words = clean_tokens(europarl_raw.__getattribute__(l).words())
	for chunk in chunks(words,1000):
		corpus_data.append(' '.join(chunk).encode('utf-8'))
		corpus_label.append(languages.index(l))


x_train, x_test, y_train, y_test = train_test_split(corpus_data, corpus_label, test_size = 0.30)

print "size of train data",len(x_train)
print "size of test data",len(x_test)


classifier = Pipeline([
    ('vec', CountVectorizer(analyzer='char',ngram_range=(3, 3))),
    ('tfidf', TfidfTransformer(use_idf=False)),
    #('clf', LogisticRegression()),
    ('clf', LinearSVC()),
])


classifier.fit(x_train, y_train)
y_predicted = classifier.predict(x_test)

acc = accuracy_score(y_test,y_predicted)
print "test acurracy:",acc

# Predict the result on new sentences:
test_sentences = [
    u'A casa da minha tia fica no alto do morro.', # portuguese
    u'The fundamental concepts and techniques are explained in detail.', # english
    u'Die Kiefern waren mit langsam wandernden Raupen bedeckt.', # german
    u'Ou est-ce qu’on peut trouver des restaurants, s’il vous plaît?' # french
]
predicted = classifier.predict(test_sentences)

#predicted_prob = classifier.predict_proba(sentences)

for s, p in zip(test_sentences, predicted):
    print u'Sentence: "%s" Predicted Language: "%s"' % (s, languages[p])



