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

native_df = pd.read_csv('native_data.csv')

wordlist = list(native_df['words'])
words = []

for review in wordlist:
    [words.append(word) for word in review]

print('starting embedding')
now = time.time()
word2vec = Word2Vec(words, min_count = 2, workers=6)
vocab = word2vec.wv.vocab
print('elapsed' + str(time.time() - now))


#https://stackabuse.com/implementing-word2vec-with-gensim-library-in-python/
#
#heeft alleen nog corpus nodig