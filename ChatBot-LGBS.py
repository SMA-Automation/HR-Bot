# -*- coding: utf-8 -*-
"""
Dr. Sönke Magnussen

This is a prototype for chatbot for HR chat.
"""
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

#my input messages
messages = []

messages += ["Hi everybody, I would like to get a copy of my last payroll"]
messages += ["Good morning, lost my last payroll, can you send it to me again?"]
messages += ["Dear Fred, it seems that I can't find payroll of mine for last month. Could you please send a copy?"]

#
#Start of preprocessing the messages
# Initialize stemmer and vectorizer
stemmer = SnowballStemmer("english")
vectorizer = CountVectorizer()

#Defining stopwords
stop_words = set(stopwords.words("english"))
stop_words = stop_words | {'hi','good', 'dear','morn'}

#
# process messages in a loop
#
input_list = []

i=0
while  i<len(messages):
    words = word_tokenize(messages[i])
    sentence = ""
    for w in words:
        stemw = stemmer.stem(w)
        if stemw not in stop_words:
            sentence += stemw+" "
    print(sentence)
    input_list += [sentence]
    i += 1

#
# Apply bag of words principle
#
bag_of_words = vectorizer.fit(input_list)
bag_of_words = vectorizer.transform(input_list)

print (bag_of_words)

print(vectorizer.vocabulary_.get("payrol"))
print(vectorizer.vocabulary_.get("send"))
print(vectorizer.vocabulary_.get("get"))

