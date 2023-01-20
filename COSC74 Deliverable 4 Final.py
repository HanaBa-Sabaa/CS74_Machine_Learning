#!/usr/bin/env python
# coding: utf-8

# In[214]:


import pandas as pd
import numpy as np
import nltk


# In[215]:


### LOADING TRAINING AND TESTING SETS ###
train_load = pd.read_csv('Train.csv')
test_load = pd.read_csv('Test.csv')


# In[216]:


### PREPROCESSING FEATURES ###

# "is awesome" anonymous function
is_awesome = lambda x: 1 if np.mean(x) > 4.5 else 0

# converting helpful metric from string to number
from ast import literal_eval
convert_helpful = lambda c: 0 if literal_eval(c)[1] == 0 else literal_eval(c)[0]/literal_eval(c)[1]
train_load['helpfulMetric'] = train_load['helpful'].apply(convert_helpful)
test_load['helpfulMetric'] = test_load['helpful'].apply(convert_helpful)


# In[217]:


# Review text preprocessing

# converting the review text column to str
train_load['reviewText'] = train_load['reviewText'].astype('str')
test_load['reviewText'] = test_load['reviewText'].astype('str')


# In[218]:


# now using TextBlob for sentiment polarization
from textblob import TextBlob
polarity = lambda x: TextBlob(x).sentiment.polarity
train_load['polarity'] = train_load['reviewText'].apply(polarity)
test_load['polarity'] = test_load['reviewText'].apply(polarity)


# In[219]:


# Tried using a separate sentiment analyzer for review summary
# was found to be ineffective

    # nltk.download('vader_lexicon')
    # from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # sid = SentimentIntensityAnalyzer()

    # polarityIntensity = lambda x: sid.polarity_scores(x)

# again using TextBlob for summary text as well
train_load['summary'] = train_load['summary'].astype('str')
test_load['summary'] = test_load['summary'].astype('str')

train_load['summaryPolarity'] = train_load['summary'].apply(polarity)
test_load['summaryPolarity'] = test_load['summary'].apply(polarity)


# In[220]:


# new helpful metric
# multiply_helpful = lambda c: 0 if literal_eval(c)[1] == 0 else literal_eval(c)[0]*(literal_eval(c)[0]/literal_eval(c)[1])
# train_load['helpfulWeight'] = train_load['helpful'].apply(multiply_helpful)
# test_load['helpfulWeight'] = test_load['helpful'].apply(multiply_helpful)


# In[221]:


# log(salesrank)
# import math
# log_sales = lambda c: math.log(c)
# train_load['logSales'] = train_load['salesRank'].apply(log_sales)
# test_load['logSales'] = test_load['salesRank'].apply(log_sales)


# In[222]:


#### BinaryPolarity was found to be ineffective/worse for the model so we decided to 
#### not include it for either review text or summaries

    # testing BinaryPolarity for review text as a new feature
    # binaryPolarity = lambda b: 1 if b >= 0 else 0
    # train_load['binaryPolarity'] = train_load['polarity'].apply(binaryPolarity)

    # testing BinaryPolarity for summary text as a new feature
    # train_load['summarybinaryPolarity'] = train_load['summaryPolarity'].apply(binaryPolarity)


# In[223]:


#### Found no matching reviewer ID within the data set so we decided not to use this metric

    # preprocessing reviewer weight
    # train_load['reviewerWeight'] = train_load.groupby('reviewerID').agg({'helpfulMetric': 'mean'})
    # train_load.groupby('reviewerID').head()


# In[224]:


# last cleaning droping all unused columns
train_load = train_load.drop(columns=['helpful','unixReviewTime', 'reviewTime', 'summary', 'price', 'categories', 'root-genre', 'title', 'label', 'first-release-year', 'songs', 'related'])
test_load = test_load.drop(columns=['helpful','unixReviewTime', 'reviewTime', 'summary', 'price', 'categories', 'root-genre', 'title', 'label', 'first-release-year', 'songs', 'related'])


# In[225]:


#### Failed features variations
# note: not using salesrank/artist had no effect; kept the features in for richness of features/robustness
    # train_data = train_load.groupby('amazon-id').agg({'polarity': 'mean', 'summaryPolarity': 'mean','helpfulMetric': 'mean','salesRank': 'mean', 'overall': is_awesome})
    # train_data = train_load.groupby('amazon-id').agg({'polarity': 'mean', 'summaryPolarity': 'mean','helpfulMetric': 'mean', 'artist': 'mean','overall': is_awesome})
    # train_data = train_load.groupby('amazon-id').agg({'polarity': 'min', 'summaryPolarity': 'mean','helpfulMetric': 'mean','summarybinaryPolarity': 'mean','overall': is_awesome})
    # train_data = train_load.groupby('amazon-id').agg({'polarity': 'mean','helpfulMetric': 'mean','overall': is_awesome})


# aggregating chosen features by amazon-id for train data
# train_data = train_load.groupby('amazon-id').agg({'polarity': 'mean', 'summaryPolarity': 'mean','helpfulMetric': 'mean','artist': 'mean', 'salesRank': 'mean', 'helpfulWeight': 'mean', 'logSales': 'mean','overall': is_awesome})
# train_data = train_load.groupby('amazon-id').agg({'polarity': 'mean', 'summaryPolarity': 'mean','helpfulMetric': 'mean','artist': 'mean', 'salesRank': 'mean', 'helpfulWeight': 'mean','overall': is_awesome})
train_data = train_load.groupby('amazon-id').agg({'polarity': 'mean', 'summaryPolarity': 'mean','helpfulMetric': 'mean','artist': 'mean', 'salesRank': 'mean','overall': is_awesome})
train_data = train_data.reset_index()

# aggregating chosen features by amazon-id for test data
# test_data = test_load.groupby('amazon-id').agg({'polarity': 'mean','summaryPolarity': 'mean', 'helpfulMetric': 'mean', 'artist': 'mean', 'helpfulWeight': 'mean', 'salesRank': 'mean','logSales': 'mean'})
# test_data = test_load.groupby('amazon-id').agg({'polarity': 'mean','summaryPolarity': 'mean', 'helpfulMetric': 'mean', 'artist': 'mean', 'salesRank': 'mean', 'helpfulWeight': 'mean'})
test_data = test_load.groupby('amazon-id').agg({'polarity': 'mean','summaryPolarity': 'mean', 'helpfulMetric': 'mean', 'artist': 'mean', 'salesRank': 'mean'})
test_data = test_data.reset_index()


# In[226]:


# separately preprocessing data for TFIDF vectorizer
# creating a dataframe with reviewText combined by amazon-id for training data
# in other words, reviews for the same product were combined into one large string for every amazon-id
TempNL_train_data = train_load.groupby('amazon-id')
listNL_train_data = TempNL_train_data['reviewText'].agg(lambda column: " ".join(column))
listNL_train_data = listNL_train_data.reset_index(name="reviewText")

# adding the new column of combined reviews sorted by amazon-id into the main training data dataframe
train_data = listNL_train_data.set_index('amazon-id').join(train_data.set_index('amazon-id'))

# creating a dataframe with reviewText combined by amazon-id for testing data
TempNL_test_data = test_load.groupby('amazon-id')
listNL_test_data = TempNL_test_data['reviewText'].agg(lambda column: " ".join(column))
listNL_test_data = listNL_test_data.reset_index(name="reviewText")

# adding the new column of combined reviews sorted by amazon-id into the main training data dataframe
test_data = listNL_test_data.set_index('amazon-id').join(test_data.set_index('amazon-id'))


# In[227]:


# TFIDF vectorizor on training data
from sklearn.feature_extraction.text import TfidfVectorizer

# initializing vectorizer
# parameters were determined by manual trial and error on a separate file
cv = TfidfVectorizer(min_df=200, max_df=3000, ngram_range=(1,2), token_pattern=r'(?u)\b[A-Za-z]+\b', max_features = 10000)

# assigning the result of vectorizer to X_train
X_train = cv.fit_transform(train_data['reviewText'])

# dropping reviewText column because no longer needed
train_data = train_data.drop(columns='reviewText')
train_data = train_data.reset_index()


# In[228]:


# TFIDF vectorizor on testing data

# assigning the result of vectorizer to X_train
X_test = cv.transform(test_data['reviewText'])

# dropping reviewText column because no longer needed
test_data = test_data.drop(columns='reviewText')
test_data = test_data.reset_index()


# In[229]:


# creating a dataframe from the features the vectorizer generated for training data
feature_names = cv.get_feature_names()
dense = X_train.todense()
denselist = dense.tolist()
df_train = pd.DataFrame(denselist, columns=feature_names)


# In[230]:


# creating a dataframe from the features the vectorizer generated for testing data
dense = X_test.todense()
denselist = dense.tolist()
df_test = pd.DataFrame(denselist, columns=feature_names)


# In[231]:


# fixing a bug where the word 'overall' was a feature word found from the texts to avoid duplicate column names
df_train = df_train.drop(columns=['overall'])
df_test = df_test.drop(columns=['overall'])


# In[232]:


# finalizing the training dataframe by adding the features from the TFIDF vectorizer to the rest of the chosen features
final_train_data = pd.concat([df_train.reset_index(drop=True), train_data.reset_index(drop=True)], axis=1)
final_train_data = pd.concat([train_data['amazon-id'].reset_index(drop=True), final_train_data.reset_index(drop=True)], axis=1)

final_train_data = final_train_data.loc[:, ~final_train_data.columns.duplicated()]
final_train_data = final_train_data.set_index('amazon-id')

# finalizing the testing dataframe by adding the features from the TFIDF vectorizer to the rest of the chosen features
final_test_data = pd.concat([df_test.reset_index(drop=True), test_data.reset_index(drop=True)], axis=1)
final_test_data = pd.concat([test_data['amazon-id'].reset_index(drop=True), final_test_data.reset_index(drop=True)], axis=1)

final_test_data = final_test_data.loc[:, ~final_test_data.columns.duplicated()]
final_test_data = final_test_data.set_index('amazon-id')


# In[235]:


#up sampling
from sklearn.utils import resample

# #combine them back for resampling
# train_data = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
not_awesome = final_train_data[final_train_data.overall==0]
awesome = final_train_data[final_train_data.overall==1]

# upsample minority
not_awesome_upsampled = resample(not_awesome,
 replace=True, # sample with replacement
 n_samples=len(awesome), # match number in majority class
 random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([awesome, not_awesome_upsampled])

# check new class counts
upsampled.overall.value_counts()


# In[236]:


# We tried down sampling, but there was information loss

# # downsample majority
# awesome_downsampled = resample(awesome,
#  replace=True, # sample with replacement
#  n_samples=len(not_awesome), # match number in minority class
#  random_state=27) # reproducible results

# # combine minority and downsampled majority
# downsampled = pd.concat([not_awesome, awesome_downsampled])

# # check new class counts
# downsampled.overall.value_counts()


# In[237]:


# upsampling
final_train_data = upsampled

# downsampling
# final_train_data = downsampled


# In[238]:


X_training = final_train_data.drop('overall', axis=1)
y_training = final_train_data['overall']


# In[239]:


#### Tried using selectKBest and RFE (see later below), but did not affect scores ####

# from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
# selectK = SelectKBest(f_classif, k=2000)
# selectK.fit(X_train, y_train)
# selectK_mask=selectK.get_support()
# reduced_df = X_train[X_train.columns[selectK_mask]]


# In[240]:


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# Choosing a model for classification

    # model = DecisionTreeClassifier()
    # model = KNeighborsClassifier(3)
    # model = GaussianNB()
    # model = SVC(kernel="linear", C=0.025)
    # model = SVC(gamma=5, C=3)
    # model = AdaBoostClassifier(n_estimators=300, random_state=0)
    # model = LogisticRegression(random_state=0).fit(X_train, y_train)

model=RandomForestClassifier(criterion = 'entropy', max_features = 'auto', class_weight = 'balanced', min_samples_split= 5, min_weight_fraction_leaf= 0.0, n_estimators= 300)


# In[241]:


##### tried feature selection with RFE as well, but also did not affect scores ####

# selectRFE = RFE(estimator=model, n_features_to_select=2000, step=0.10)
# # selectK=selectRFE
    
# selectRFE.fit(X_train,y_train)
# selectRFE_mask=selectRFE.get_support()

# reduced_df = X_train[X_train.columns[selectRFE_mask]]
# X_test = X_test[X_train.columns[selectRFE_mask]]


# In[242]:


####### pipelining and ensembling attempts ###########

# # model=RandomForestClassifier(criterion = 'entropy', max_features = 'auto', class_weight = 'balanced', min_samples_split= 5, min_weight_fraction_leaf= 0.0, n_estimators= 200)
# ABC = AdaBoostClassifier()

# from sklearn.pipeline import Pipeline 
# kbest = SelectKBest(f_classif) 
# selectRFE = RFE(estimator=ABC)
# pipeline = Pipeline([('kbest', kbest), ('rfe', selectRFE),('Adaboost', ABC)])

# from sklearn.model_selection import GridSearchCV

# tuned_parameters = {
#     'kbest__k':[500, 1000, 2000],
#     'Adaboost__n_estimators' : [100, 200, 300],
#     'Adaboost__random_state': [0],
#     'rfe__n_features_to_select': [2000, 3000],
#     'rfe__step': [0.15, 0.20]
# }
# # scores = ['f1_weighted']

# print("# Tuning hyper-parameters for %s" % score)

# adaboost_gs = GridSearchCV(estimator = pipeline, param_grid=tuned_parameters, scoring='f1_weighted', cv=10)
# adaboost_gs.fit(X_train, y_train)

# #save best model
# adaboost_best = adaboost_gs.best_estimator_
# #check best n_neigbors value
# print(adaboost_gs.best_params_)
# print('adaboost: {}'.format(adaboost_best.score(X_test, y_test)))

##### Similar code for RandomForest pipelining omitted for concision ########

##### Ensembling Code #######
# from sklearn.ensemble import VotingClassifier
# #create a dictionary of our models
# estimators=[('Adaboost', adaboost_best), ('randomforest', randomforest_best)]
# #create our voting classifier, inputting our models
# ensemble = VotingClassifier(estimators, voting='hard')

# #fit model to training data
# ensemble.fit(X_train, y_train)
# #test our model on the test data
# ensemble.score(X_test, y_test)


# In[243]:


#### Code from hyperparameter optimization ####

# instance of optimizing parameters for decisiontree
# from sklearn.model_selection import GridSearchCV
# tuned_parameters = {
#          'criterion': ['gini', 'entropy'],
#          'splitter':['best','random'],
#          'min_samples_split': [2,5],
#      }
# model=DecisionTreeClassifier(criterion = 'gini',splitter='best',min_samples_split='5')
# scores = ['f1_weighted']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)

#     clf = GridSearchCV(
#                  DecisionTreeClassifier(), tuned_parameters, scoring=score
#              )
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print(clf.best_params_)
#     print("Grid scores on development set:")

#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#         % (mean, std * 2, params))

#     print("Detailed classification report:")
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")

#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))    

#### End of hyperparameter optimization code ######


# In[244]:


# Calculate the accuracy
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import cross_validate

# initializing kfold with 10 folds
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True)

# saving cross validation results as a dictionary with keys: 
# fit_time, score_time, estimator, test_accuracy, test_precision, test_recall, test_f1_weighted

results = cross_validate(model, X=X_training, y=y_training, cv=kfold,
                                          scoring=['f1_weighted', 'precision', 'recall'], return_estimator=True)


# In[245]:


# printing all the metrics for each fold
print(results)


# In[246]:


# printing averages of the 10 splits for each respective metric 
mean_f1 = np.mean(results['test_f1_weighted'])
mean_precision = np.mean(results['test_precision'])
mean_recall = np.mean(results['test_recall'])

print('precision after 10 fold cross validation: '+str(mean_precision))
print('recall after 10 fold cross validation: '+str(mean_recall))
print('f1 score after 10 fold cross validation: '+str(mean_f1))


# In[247]:


# since we must predict on the test data, we chose the estimator that happened to perform the best out of all the kfold splits
# although we could choose the one that performed the worst — theoretically it should not make a difference
max_index = np.unravel_index(np.argmax(results['test_f1_weighted'], axis=None), results['test_f1_weighted'].shape)
fitted_model = results['estimator'][max_index[0]]


# In[249]:


# using fitted model to produce predictions for test data
preds_no_labels = fitted_model.predict(final_test_data)
# from produce_predictions
output = pd.DataFrame({'amazon-id': final_test_data.index, 'Awesome': preds_no_labels})
output.to_csv('./Predictions.csv')


# In[ ]:




