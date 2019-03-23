#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:49:39 2019

@author: timo
"""
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import pandas as pd
import time
from nltk.corpus import brown
import logging
import numpy as np

#native_df = pd.read_csv('native_data.csv')
#
#wordlist = list(native_df['words'])
#words = []
#
##<<<<<<< Updated upstream
#for line in wordlist:
#    line = line.replace(']', '')
#    line = line.replace('[', '')
#    line = line.replace("'", '')
#    line = line.replace(',', '')
#    sep = line.split(" ")
#    for word in sep:
#        words.append(word)
#
##print('starting embedding')
##now = time.time()
##word2vec = Word2Vec(words, min_count = 2, workers=4)
##=======
#        
#for review in wordlist:
#    [words.append(word) for word in review]


words = [word for word in brown.words()]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = brown.sents()
model = Word2Vec(sentences, min_count=1)
model.save('/tmp/brown_model')

native_df = pd.read_csv('native_data.csv')

wordlist = list(native_df['words'])
new_words = []
for line in wordlist:
    line = line.replace(']', '')
    line = line.replace('[', '')
    line = line.replace("'", '')
    line = line.replace(',', '')
    sep = line.split(" ")
    features = []
    new_words.append(sep)
    
features = []

n_words_to_sample = 20
for sentence in new_words:
    sentence = [word for word in sentence if len(word)>1]
    if len(sentence) > 1:
        feature_list = []
    #    sample random words
        if len(sentence) < n_words_to_sample:
            for x in range(len(sentence)):
                
                word = sentence.pop(int(np.random.randint(0,len(sentence), 1)))
                try:
                    word_features = model.wv[word]
                except KeyError as e:
                    padding = np.random.uniform(0,0,100)
                    word_features = padding
                feature_list.append(word_features)
            
            for x in range(len(feature_list), n_words_to_sample):
                padding = np.random.uniform(0,0,100)
                feature_list.append(padding)
                
                
        else:
            for x in range(n_words_to_sample):
                word = sentence.pop(int(np.random.randint(0,len(sentence), 1)))
                try:
                    word_features = model.wv[word]
                except KeyError as e:
                    padding = np.random.uniform(0,0,100)
                    word_features = padding
                
                feature_list.append(word_features)
    else:
        feature_list = None
    
    features.append(feature_list)
        




#yolo = native_df.head(100)
    
def concat_features(arraylist):
    if arraylist == None:
        return None
    total = []
    for item in arraylist:
        for getal in item:
            total.append(getal)
    return total

ekte_features = []
for index in features:
    ekte_features.append(concat_features(index))
    
native_df['features'] = ekte_features
    
native_df.to_csv('native_features_labels.csv')

        
#cutoff == 10 woorden


#https://stackabuse.com/implementing-word2vec-with-gensim-library-in-python/
#
#heeft alleen nog corpus nodig