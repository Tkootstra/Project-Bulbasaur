# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:23:47 2019

@author: 'Timo Kootstra - t.m.kootstra@uu.nl
"""

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







df['English Speaking'] = df['Reviewer_Nationality'].apply(lambda x: 'Native' if x in english_countries else 'Non_native')


pos_reviews = list(df['Positive_Review'])


ADJ_len = []
ADV_len = []


from nltk.corpus import wordnet as wn
#for review in pos_reviews:
#    tagged = nltk.pos_tag(review)
#  
#    words = [x[0] for x in tagged if  x[1] == 'JJ' or x[1] == 'RB']
#    tags = [x[1] for x in tagged if  x[1] == 'JJ' or x[1] == 'RB']
#    
#    ADJ_len.append(len([x for x in tags if 'JJ' in x]))
#    ADV_len.append(len([x for x in tags if 'RB'in x]))


def extract_features_concurrent(review):
    
    tagged = nltk.pos_tag(review)
  
    words = [x[0] for x in tagged if  x[1] == 'JJ' or x[1] == 'RB']
    tags = [x[1] for x in tagged if  x[1] == 'JJ' or x[1] == 'RB']
    
    ADJ_len = (len([x for x in tags if 'JJ' in x]))
    ADV_len = (len([x for x in tags if 'RB'in x]))
    print('heuken')
    return ADJ_len, ADV_len


from multiprocessing import Pool, freeze_support

if __name__ == "__main__": 
    pool = Pool(6)
    freeze_support()
    
    adjlen, advlen = pool.map(extract_features_concurrent,pos_reviews)
    pool.join()
    pool.close()

        
#        
#df_new = df['English Speaking']
#df_new['ADJ_len'] = ADJ_len
#df_new['ADV_len'] = ADV_len
#
#from sklearn.model_selection import train_test_split
#
#input_features = df_new.drop("English Speaking", axis=1).columns
#output_feature = 'English Speaking'
#train_data, test_data = train_test_split(df_new, train_size=0.6)
#
#from sklearn.linear_model import LogisticRegression
#
#Xtrain = train_data[input_features]
#ytrain = train_data[output_feature]
#
#Xtest = test_data[input_features]
#ytest = test_data[output_feature]
#
#model = LogisticRegression(verbose=True, n_jobs=6)
#
#model.fit(Xtrain,ytrain)
#
#preds = model.predict(Xtest)
#
#from sklearn.metrics import classification_report
#print(classification_report(ytest, preds))
#
#    
#
#
#
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




    





