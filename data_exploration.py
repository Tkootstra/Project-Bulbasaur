#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:50:19 2019

@author: timo
"""

import pandas as pd
import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
import numpy as np
import time

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


    

df = pd.read_csv('Hotel_Reviews.csv')

english_countries = [' United Kingdom ', ' United States of America ',
                     ' Australia ',' Ireland ',' New Zealand ',
                     ' Canada ']



df['English Speaking'] = df['Reviewer_Nationality'].apply(lambda x: True if x in english_countries else False)

#TODO: review text extracten. done
#TODO: uit review text adverbs + adjective extracten. done
#TODO: die bij elkaar voegen. done


#TODO: semantics van de adj + adv clusteren
#TODO: features -> word embeddings
#TODO: beslissing maken over feature selection
#TODO: feature selection -> worden deze features anders gebruikt voor native vs non native


#adjective == bijvoegelijk naamwoord
#adverb == bijwoord

native_list = []
nonnative_list = []
native_labels = []
nonnative_labels = []


total_list = []
for i in range(len(df)):
    
    
    pos_review = df.iloc[i]['Negative_Review']
    pos_review = pos_review.split()
    neg_review = df.iloc[i]['Positive_Review']
    neg_review = neg_review.split()
    label = df.iloc[i]['Reviewer_Score']
    
    if len(neg_review) > 3 and len(pos_review) > 3:
        total = pos_review + neg_review
        
        if df.iloc[i]['English Speaking']:
            native_list.append(total)
            native_labels.append(label)
        else:
            nonnative_list.append(total)
            nonnative_labels.append(label)



native_words = []
native_tags = []
nonnative_words = []
nonnative_tags = []

from nltk.corpus import wordnet as wn
for review in native_list:
    tagged = nltk.pos_tag(review)
    
    words = [x[0] for x in tagged if  x[1] == 'JJ' or x[1] == 'RB']
    tags = [x[1] for x in tagged if  x[1] == 'JJ' or x[1] == 'RB']
    
    native_words.append(words)
    native_tags.append(tags)
    
for review in nonnative_list:
    tagged = nltk.pos_tag(review)
    
    words = [x[0] for x in tagged if  x[1] == 'JJ' or x[1] == 'RB']
    tags = [x[1] for x in tagged if  x[1] == 'JJ' or x[1] == 'RB']
    
    nonnative_words.append(words)
    nonnative_tags.append(tags)

nat_df = dict()
nat_df['words'] = native_words
nat_df['tags'] = native_tags
nat_df['label'] = native_labels

nonnat_df = dict()
nonnat_df['words'] = nonnative_words
nonnat_df['tags'] = nonnative_tags
nonnat_df['label'] = nonnative_labels

nat_df = pd.DataFrame(nat_df)
nonnat_df = pd.DataFrame(nonnat_df)



    







#native_data = dict()
#nonnative_data = dict()
#native_data['review'] = native_list
#native_data['label'] = native_labels
#
#nonnative_data['review'] = nonnative_list
#nonnative_data['label'] = nonnative_labels
#
#native_df = pd.DataFrame(native_data)
#non_native_df = pd.DataFrame(nonnative_data)
#
#
#test = list(native_df['review'].head(1))
#
#yolo = nltk.pos_tag(test[0])
#




            
        
        
        
    
    














































#TODO:



#native_reviews_negative = (df.loc[df['English Speaking'] == True]['Negative_Review'])
#native_labels_negative = (df.loc[df['English Speaking'] == True]['Reviewer_Score'])
#native_reviews_positive = (df.loc[df['English Speaking'] == True]['Positive_Review'])
#
#nonnative_reviews_negative = (df.loc[df['English Speaking'] == False]['Negative_Review'])
#nonnative_labels_negative = (df.loc[df['English Speaking'] == False]['Reviewer_Score'])
#nonnative_reviews_positive = (df.loc[df['English Speaking'] == False]['Positive_Review'])


#native_negative = dict()
#
#from sklearn.feature_extraction.text import CountVectorizer
#
#X = nonnative_reviews_negative
#y = nonnative_labels_negative
#bow_transformer = CountVectorizer(analyzer=process_text).fit(X)
#
#X = bow_transformer.transform(X)
#
#from sklearn.model_selection import train_test_split
#
#Xtrain,Xtest,ytrain,ytest = train_test_split(X,y, test_size=0.4, random_state=101)
#
#from sklearn.ensemble import RandomForestRegressor
#now = time.time()
#model = RandomForestRegressor(n_estimators=10, n_jobs=6)
#model.fit(Xtrain, ytrain)
##
#from sklearn.metrics import explained_variance_score
#preds = model.predict(Xtest)
#print(explained_variance_score(ytest, preds))
#print((time.time()-now)/60)
#




    





