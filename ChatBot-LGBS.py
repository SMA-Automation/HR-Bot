# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
vectorizer = CountVectorizer()

string_1 = "Hi everybody, I would like to geht a copy of my last payroll"
string_2 = "Good morning, lost my last payroll, can you send it to me again?"
string_3 = "Dear Fred, it seems that I can't found payroll of mine for last month. Could you please send a copy?"

input_list = [string_1,string_2,string_3]

bag_of_words = vectorizer.fit(input_list)
bag_of_words = vectorizer.transform(input_list)

print (bag_of_words)

vectorizer.vocabulary_.get("payroll")


