# -*- coding: utf-8 -*-
"""
Dr. Sönke Magnussen

This is a prototype for chatbot for HR chat.
Version 0.6 02.04.2017

"""

debug = False

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn import svm

#
# One of the machine learning classification methods
#
#clf = svm.SVC()
clf = tree.DecisionTreeClassifier()

#
# my input messages
#
messages = []

messages += ["Hi everybody, I would like to get a copy of my last payroll"]
messages += ["Good morning, lost my last payroll, can you send it to me again?"]
messages += ["Dear Fred, it seems that I can't find payroll of mine for last month. Could you please send a copy?"]

messages += ["Dear team, please find attached my new health insurance certificate"]
messages += ["Hi everybode, please apply my new attached my health insurance certificate for next payrol job"]

#labels: Labels are the classifications of the different HRService classes
YLabels = [1,1,1,2,2]

#
# Testfälle
#
messageChatbot =[]
messageChatbot += ["Hi there, please send my last payrol again"]
messageChatbot += ["Hi there, please find attached my new health insurance certificate"]


#
# Start of preprocessing the messages
# Initialize stemmer and vectorizer
stemmer = SnowballStemmer("english")
vectorizer = CountVectorizer()

#Defining stopwords
stop_words = set(stopwords.words("english"))
stop_words = stop_words | {'hi','good', 'dear','morn'}

#
# Function vectorize Words for pre-processsing of input text
# ##  process messages in a loop
# ## for stemming and stopwords
#
def vectorizeWords(messages) :
    input_list = []
    i=0
    while  i<len(messages):
        words = word_tokenize(messages[i])
        sentence = ""
        for w in words:
            stemw = stemmer.stem(w)
            if stemw not in stop_words:
                sentence += stemw+" "
        if debug: print(sentence)
        input_list += [sentence]
        i += 1
    return input_list


def dummyAnswer(inputText, vectorizedWords):
    return "I do not understand!"

def handlePayrolRequest(inputText,vectorizedWords):
    return "Will send payrol back"
    
def handleHealthInsuranceCert(inputText,vectorizedWords):
    return "Will handle Health Insurance certificate"


# 
# Prepare machine learning 
# 1. vectorize words
# 
input_list = vectorizeWords(messages)

#
# 2. Apply bag of words principle as preparation for fitting
#
bag_of_words = vectorizer.fit(input_list)
bag_of_words = vectorizer.transform(input_list)


#
# Einige interessante Logs Anweisungen
#
if debug:
    print(vectorizer.vocabulary_)
    print (bag_of_words)
    print()
    print(vectorizer.vocabulary_)
    print()
    print(vectorizer.vocabulary_.get("payrol"))
    print(vectorizer.vocabulary_.get("send"))
    print(vectorizer.vocabulary_.get("get"))



#
# 3. Fitting: Use classifier for machine learning
#
clf.fit(bag_of_words,YLabels)

#
#Test Durchführung
#
if debug:
    vw = vectorizeWords(messageChatbot)
    vcb = vectorizer.transform(vw)
    print(vcb)
    p = clf.predict(vcb)
    print("classification -->" + str(p))


#
# Und hier der einfache Chatbot
#
while True:    # infinite loop
     message = input("you: ")
     messageChatbot = [message]
     vw = vectorizeWords(messageChatbot)
     vcb = vectorizer.transform(vw)
     p = clf.predict(vcb)
     if debug: print("classification -->" + str(p))
     if message == "exit":
        break
     elif p[0] == 1:
        print(handlePayrolRequest(message,vw))
     elif p[0] == 2:
        print(handleHealthInsuranceCert(message,vw))
     else:
        dummyAnswer(message,vw)

