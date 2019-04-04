"""
Project-Bulbasaur
Copyright (C) 2019 Timo Kootstra, Floris van den Esschert, Alex Hoogerbrugge

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pandas as pd
import nltk
import numpy as np
import os
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from skopt.space import Real, Categorical
from skopt import gp_minimize
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
import seaborn as sns

# Set all to true if this is the first time running the script. If features.csv
# is already in your working directory, set feature extraction to False.
RUN_FEATURE_EXTRACTION = False
RUN_HYPERPARAM_TUNING = False
RUN_CV = True

N_CORES = os.cpu_count() - 1
TUNING_CALLS = 50


if RUN_FEATURE_EXTRACTION:
    # Download file at: 
    # https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe/downloads/515k-hotel-reviews-data-in-europe.zip/1
    df = pd.read_csv('Hotel_Reviews.csv')
    
    english_countries = [' United Kingdom ', ' United States of America ',
                         ' Australia ',' Ireland ',' New Zealand ',
                         ' Canada ']
    
    
    # Add a Native/Non-native and review length column at the end of the dataframe
    df['English Speaking'] = df['Reviewer_Nationality'].apply(lambda x: 'Native' if x in english_countries else 'Non_native')
    df['review length'] = df['Positive_Review'].apply(lambda col: len([word for word in col.split(' ')]))

    pos_reviews = list(df['Positive_Review'])

    def extract_features_concurrent(review):
        tagged = nltk.pos_tag(review)
      
        adjectives = [x[0] for x in tagged if  x[1] == 'JJ']
        adverbs = [x[0] for x in tagged if  x[1] == 'RB']
        verbs  = [x[0] for x in tagged if 'VB' in  x[1]]
        prepositions = [x[0] for x in tagged if  x[1] == 'IN']
        conjunctions = [x[0] for x in tagged if  x[1] == 'CC']
            
        adj_ratio = len(adjectives) - len(set(adjectives))
        adv_ratio = len(adverbs) - len(set(adverbs))
        verb_ratio = len(verbs) - len(set(verbs))
        prep_ratio = len(prepositions) - len(set(prepositions))
        conj_ratio = len(conjunctions) - len(set(conjunctions))
        
        
        return adj_ratio, adv_ratio, verb_ratio, prep_ratio, conj_ratio
    

    # Run feature extraction in parellel with n_cores (default = total_cores - 2)
    results = Parallel(n_jobs=N_CORES, verbose=True)(delayed(extract_features_concurrent)(review) for review in pos_reviews)
    
    adj_ratio, adv_ratio, verb_ratio, prep_ratio, conj_ratio = (zip(*results))
          
    labels = list(df['English Speaking'])
    
    # Save features to dataframe
    data = dict()
    data['adjectives'] = adj_ratio
    data['adverbs'] = adv_ratio
    data['verbs'] = verb_ratio
    data['prepositions'] = prep_ratio
    data['conjunctions'] = conj_ratio
    data['label'] = labels
    
    df_2 = pd.DataFrame(data)
    df_2.to_csv('features.csv')

else:
    df_2 = pd.read_csv('features.csv')


# Prepare the dataset to be inserted into the model
input_features = ['adjectives','adverbs', 'verbs', 'prepositions', 'conjunctions']
output_feature = 'label'
scaled = StandardScaler().fit_transform(df_2[input_features])
df_2[input_features] = scaled
train_data, test_data = train_test_split(df_2, train_size=0.6)


# This section will approximate the best parameters for the regression model
if RUN_HYPERPARAM_TUNING:
    model = LogisticRegression()
    X = df_2[input_features]
    y = df_2[output_feature]
    
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X=X, y=y)
    
    space = [Categorical(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], name= 'solver'),
             Real(0.000001, 10000, name= 'C'),
             
             Categorical([True, False], name= 'warm_start')]
        	
    @use_named_args(space)
    def objective(**params):
        model.set_params(**params)
        
        return -np.mean(cross_val_score(model,train_data[input_features],train_data[output_feature], cv=10, n_jobs=N_CORES, scoring='roc_auc'))
    
    
    estimator_gaussian_process = gp_minimize(objective, space, n_calls=TUNING_CALLS, verbose=True)
    
    print("Best score=%.4f" % estimator_gaussian_process.fun)

    
# This section will run 10-fold leave-one-out cross-validation
if RUN_CV:
    coef1 = []
    coef2 = []
    coef3 = []
    coef4 = []
    coef5 = []

    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X=train_data[input_features], y=train_data[output_feature])
    
    aucs = []
    colors = ['r','b','g','y','c','m','k']
    counter = 0
    
    fprs = []
    tprs = []
    aucs = []
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure()
    best_params = list(estimator_gaussian_process.x)
    model = LogisticRegression(solver=best_params[0], 
                               C=best_params[1], 
                               warm_start=best_params[2])
    
    coefs = []
    plt.figure(figsize=(10,10))
    for train_index, test_index in skf.split(train_data[input_features], train_data[output_feature]):
        X = df_2[input_features]
        y = df_2[output_feature]
        
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        model.fit(X_train, y_train)
        preds_proba = model.predict_proba(X_test)
        coefs.append(model.coef_)
        
        for i in range(2):
            yscores = model.predict_proba(X_test)[:,i]
            fpr[i], tpr[i], _ = roc_curve(LabelBinarizer().fit_transform(y_test), yscores)
            roc_auc[i] = auc(fpr[i], tpr[i])
        fprs.append(fpr[1])
        tprs.append(tpr[1])
        
        
        coef1.append(model.coef_[0][0])
        coef2.append(model.coef_[0][1])
        coef3.append(model.coef_[0][2])
        coef4.append(model.coef_[0][3])
        coef5.append(model.coef_[0][4])
        
        print(roc_auc_score(y_test, yscores))
        plt.plot(fpr[1], tpr[1], lw=1, alpha=0.4, color=np.random.choice(colors), 
                 label = 'CV_'+str(counter+1)+'_AUC: '+str(round(roc_auc[1],2)))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        counter += 1
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
    plt.show()
    coef1 = np.mean(coef1)
    coef2 = np.mean(coef2)
    coef3 = np.mean(coef3)
    coef4 = np.mean(coef4)
    coef5 = np.mean(coef5)

    plt.figure(figsize=(10,10))
    sns.barplot(x=input_features, y = [coef1, coef2, coef3, coef4, coef5])
    plt.show()
    
    
else:
    Xtrain = train_data[input_features]
    ytrain = train_data[output_feature]
    
    Xtest = test_data[input_features]
    ytest = test_data[output_feature]
    
    model = LogisticRegression(solver=best_params[0], 
                               C=best_params[1], 
                               warm_start=best_params[2])
    
    model.fit(Xtrain,ytrain)
    
    preds = model.predict(Xtest)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        yscores = model.predict_proba(Xtest)[:,i]
        fpr[i], tpr[i], _ = roc_curve(LabelBinarizer().fit_transform(ytest), yscores)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    print(roc_auc_score(ytest, yscores))
    plt.figure()
    plt.plot(fpr[1], tpr[1], lw=2, color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc="lower right")
    plt.show()

    best_params = list(estimator_gaussian_process.x)
    model = LogisticRegression(solver=best_params[0], 
                               C=best_params[1], 
                               warm_start=best_params[2])
    
    print(cross_val_score(model,X,y,scoring='roc_auc', n_jobs=N_CORES, cv=10))

