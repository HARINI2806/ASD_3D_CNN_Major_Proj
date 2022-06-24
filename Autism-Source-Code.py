#!/usr/bin/env python
# coding: utf-8

# # **Importing the Dataset**

# In[1]:


# pip install vpython


# In[2]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

# from IPython.print import print 


# print for notebooks , visualisation tool
# get_ipython().run_line_magic('matplotlib', 'inline')


data = pd.read_csv('C:/Users/User/ASD_Projs/ASD_3D_CNN_Major_Proj/ASD.csv')
(data.head(n=5))


# In[3]:






asd_data = pd.read_csv('C:/Users/User/ASD_Projs/autism_spectrum_detection-main-T9/ASD.csv', na_values=['?'])
asd_data.head(n=5)


# In[10]:


asd_data.describe()


# # **Cleaning the dataset**

# In[11]:


asd_data.loc[(asd_data['age'].isnull()) |(asd_data['gender'].isnull()) |(asd_data['ethnicity'].isnull()) 
|(asd_data['jundice'].isnull())|(asd_data['austim'].isnull()) |(asd_data['contry_of_res'].isnull())
            |(asd_data['used_app_before'].isnull())|(asd_data['result'].isnull())|(asd_data['age_desc'].isnull())
            |(asd_data['relation'].isnull())]


# In[12]:


asd_data.dropna(inplace=True)
asd_data.describe()


# Data types of all our features including the target feature. Moreover,  count the total number of instances and the target-class distribution.

# In[13]:


# Reminder of the features:
print(asd_data.dtypes)


# Total number of records in clean dataset
n_records = len(asd_data.index)

# TODO: Number of records where individual's with ASD in the clean dataset
n_asd_yes = len(asd_data[asd_data['Class/ASD'] == 'YES'])

# TODO: Number of records where individual's with no ASD in the clean dataset
n_asd_no = len(asd_data[asd_data['Class/ASD'] == 'NO'])

# Print the results
print ("Total number of records: {}".format(n_records))
print ("Individuals diagonised with ASD: {}".format(n_asd_yes))
print ("Individuals not diagonised with ASD: {}".format(n_asd_no))


# *Next*  we convert the Pandas dataframes into numpy arrays that can be used by scikit_learn. We create an array that extracts only the feature data we want to work with and another array that contains the classes (class/ASD).

# In[14]:


# Split the data into features and target label
asd_raw = asd_data['Class/ASD']
features_raw = asd_data[['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score',
                      'A9_Score','A10_Score']]


# Some of our models require the input data to be normalized, normalize the attribute data. Here, I use preprocessing.MinMaxScaler().

# In[15]:


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# numerical = ['age', 'result']
features_raw = pd.DataFrame(data = features_raw)
# features_minmax_transform = pd.DataFrame(data = features_raw)
# features_minmax_transform[numerical] = scaler.fit_transform(features_raw[numerical])
# features_minmax_transform
# # Show an example of a record with scaling applied

# (features_minmax_transform.head(n = 5))


# # **One-Hot-Coding**

# In[16]:


#One-hot encode the 'features_minmax_transform' data using pandas.get_dummies()
# features_final = pd.get_dummies(features_minmax_transform)
# # features_final = features_raw
# print(features_final.head(5))


# # Encode the 'all_classes_raw' data to numerical values
asd_classes = asd_raw.apply(lambda x: 1 if x == 'YES' else 0)



# # Print the number of features after one-hot encoding
# encoded = list(features_final.columns)
# print ("{} total features after one-hot encoding.".format(len(encoded)))

# # Uncomment the following line to see the encoded feature names
# print(encoded)


# In[17]:


# histogram of Class/ASD
from matplotlib import pyplot as plt
# 8 bins
# plt.hist(asd_classes, bins=10)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of Class/ASD')
plt.xlabel('Class/ASD from processed data')
plt.ylabel('Frequency')


# # **Shuffle and Split Data**

# In[18]:


from sklearn.model_selection import cross_val_score, train_test_split

np.random.seed(1234)

X_train, X_test, y_train, y_test = train_test_split(features_raw, asd_classes, train_size=0.80, random_state=1)


# Show the results of the split
print ("Training set has {} samples.".format(X_train.shape[0]))
print ("Testing set has {} samples.".format(X_test.shape[0]))
#asd_data


# # **Models**

# # ## **1.Decision tree**

# # In[19]:


# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier

# dectree = DecisionTreeClassifier(random_state=1)

# # Train the classifier on the training set
# dectree.fit(X_train, y_train)


# # ### Depiction of Decision Tree algorithm

# # In[20]:


# # import pydotplus 


# # dot_data = tree.export_graphviz(dectree,
# #                                 out_file=None,
# #                                 filled=True,
# #                                 rounded=True,
# #                                 special_characters=True)  
# # graph = pydotplus.graph_from_dot_data(dot_data)  

# # from IPython.print import Image 
# # Image(graph.create_png()) 


# # # **Evaluating Model Performance**

# # In[21]:


# # make class predictions for the testing set
# y_pred_class = dectree.predict(X_test)


# # In[22]:


# # print the first 25 true and predicted responses
# print('True:', y_test.values[0:25])
# print('False:', y_pred_class[0:25])


# # Confusion matrix

# # In[23]:


# from sklearn import metrics
# # this produces a 2x2 numpy array (matrix)
# #print(metrics.confusion_matrix(y_test, y_pred_class))

# # save confusion matrix and slice into four pieces
# confusion = metrics.confusion_matrix(y_test, y_pred_class)
# print(confusion)
# #[row, column]
# TP = confusion[1, 1]
# TN = confusion[0, 0]
# FP = confusion[0, 1]
# FN = confusion[1, 0]


# # Classification Accuracy

# # In[24]:


# # use float to perform true division, not integer division
# print((TP + TN) / float(TP + TN + FP + FN))


# # Classification Error:

# # In[25]:


# classification_error = (FP + FN) / float(TP + TN + FP + FN)

# print(classification_error)


# # Sensitivity:

# # In[26]:


# sensitivity = TP / float(FN + TP)

# print(sensitivity)
# print(metrics.recall_score(y_test, y_pred_class))


# # Specificity:

# # In[27]:


# specificity = TN / (TN + FP)

# print(specificity)


# # False Positive Rate:

# # In[28]:


# false_positive_rate = FP / float(TN + FP)

# print(false_positive_rate)
# #print(1 - specificity)


# # Precision:

# # In[29]:


# precision = TP / float(TP + FP)

# #print(precision)
# print(metrics.precision_score(y_test, y_pred_class))


# # ### Visualizing the classification prediction:

# # In[30]:


# # print the first 10 predicted responses
# # 1D array (vector) of binary values (0, 1)
# dectree.predict(X_test)[0:10]


# # In[31]:


# # print the first 10 predicted probabilities of class membership
# dectree.predict_proba(X_test)[0:10]


# # In[32]:


# # store the predicted probabilities for class 1
# y_pred_prob = dectree.predict_proba(X_test)[:, 1]
# print(y_pred_prob)


# # In[33]:


# # allow plots to appear in the notebook

# import matplotlib.pyplot as plt

# # adjust the font size 
# plt.rcParams['font.size'] = 12
# # histogram of predicted probabilities

# # 8 bins
# plt.hist(y_pred_prob, bins=10)

# # x-axis limit from 0 to 1
# plt.xlim(0,1)
# plt.title('Histogram of predicted probabilities')
# plt.xlabel('Predicted probability of ASD')
# plt.ylabel('Frequency')


# # \# **Receiver Operating Characteristic (ROC) Curves**

# # In[34]:



# # we pass y_test and y_pred_prob
# # we do not use y_pred_class, because it will give incorrect results without generating an error
# # roc_curve returns 3 objects fpr, tpr, thresholds
# # fpr: false positive rate
# # tpr: true positive rate
# y_t = np.array(y_test)
# print(y_t)
# print(y_pred_prob)


# fpr, tpr, thresholds = metrics.roc_curve(y_t, y_pred_prob)

# print(fpr)
# print(tpr)
# print(thresholds)

# plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.rcParams['font.size'] = 12
# plt.title('ROC curve for Autism classifier')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.grid(True)


# # Score metric for Model performance

# # In[35]:


# dectree.score(X_test, y_test)


# # Cross-validation:

# # In[36]:


# from sklearn.model_selection import cross_val_score

# dectree = DecisionTreeClassifier(random_state=1)

# cv_scores = cross_val_score(dectree, features_final, asd_classes, cv=10)

# cv_scores.mean()


# # 
# # AUC Score:

# # In[37]:


# # calculate cross-validated AUC
# from sklearn.model_selection import train_test_split
# cross_val_score(dectree, features_final, asd_classes, cv=10, scoring='roc_auc').mean()


# # F-beta Score:

# # In[38]:


# dectree.fit(X_train, y_train)
# from sklearn.metrics import fbeta_score
# predictions_test = dectree.predict(X_test)
# fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# ## **2.Random Forest**

# RandomForestClassifier instead to see whether it performs better.

# In[39]:


from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier(n_estimators=5, random_state=1)
cv_scores = cross_val_score(ranfor, features_raw, asd_classes, cv=10)
cv_scores.mean()


# AUC Score: 

# In[40]:


# calculate cross-validated AUC
from sklearn.model_selection import train_test_split
cross_val_score(ranfor, features_raw, asd_classes, cv=10, scoring='roc_auc').mean()


# F-beta Score:

# In[41]:


ranfor.fit(X_train, y_train)
from sklearn.metrics import fbeta_score
predictions_test = ranfor.predict(X_test)
fbeta_score(y_test, predictions_test, average='binary', beta=0.5)




# # ## **3.SVM**

# # In[42]:


# from sklearn import svm

# C = 1.0
# svc = svm.SVC(kernel='linear', C=C, gamma=2)


# # In[43]:


# cv_scores = cross_val_score(svc, features_final, asd_classes, cv=10)

# cv_scores.mean()


# # AUC Score:

# # In[44]:


# # calculate cross-validated AUC
# from sklearn.model_selection import train_test_split
# cross_val_score(svc, features_final, asd_classes, cv=10, scoring='roc_auc').mean()


# # F-beta Score:

# # In[45]:


# svc.fit(X_train, y_train)
# from sklearn.metrics import fbeta_score
# predictions_test = svc.predict(X_test)
# fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# # ## **4.K-Nearest-Neighbors (KNN)**

# # In[46]:


# from sklearn import neighbors

# knn = neighbors.KNeighborsClassifier(n_neighbors=10)
# cv_scores = cross_val_score(knn, features_final, asd_classes, cv=10)

# cv_scores.mean()


# # AUC Score:

# # In[47]:


# # calculate cross-validated AUC
# from sklearn.model_selection import train_test_split
# cross_val_score(knn, features_final, asd_classes, cv=10, scoring='roc_auc').mean()


# # F-beta Score:

# # In[48]:


# knn.fit(X_train, y_train)
# from sklearn.metrics import fbeta_score
# predictions_test = knn.predict(X_test)
# fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# # Choosing K is treachreous, so we iterate through different values of K. Hence we write a for loop to run KNN with K values ranging from 10 to 50 and see if K makes a substantial difference.
# # 
# # > Indented block
# # 
# # 

# # In[49]:


# for n in range(10, 50):
#     knn = neighbors.KNeighborsClassifier(n_neighbors=n)
#     cv_scores = cross_val_score(knn, features_final, asd_classes, cv=10)
#     print (n, cv_scores.mean())


# # ## **5.Naive Bayes**

# # In[50]:


# from sklearn.naive_bayes import MultinomialNB

# #scaler = preprocessing.MinMaxScaler()
# #all_features_minmax = scaler.fit_transform(all_features)

# nb = MultinomialNB()
# cv_scores = cross_val_score(nb, features_final, asd_classes, cv=10)

# cv_scores.mean()


# # AUC Score: 

# # In[51]:


# # calculate cross-validated AUC
# from sklearn.model_selection import train_test_split
# cross_val_score(nb, features_final, asd_classes, cv=10, scoring='roc_auc').mean()


# # F-beta Score:

# # In[52]:


# nb.fit(X_train, y_train)
# from sklearn.metrics import fbeta_score
# predictions_test = nb.predict(X_test)
# fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# # ## **6.Logistic Regression**

# # In[53]:


# from sklearn.linear_model import LogisticRegression

# logreg = LogisticRegression()
# cv_scores = cross_val_score(logreg, features_final, asd_classes, cv=10)
# cv_scores.mean()


# # AUC Score: 

# # In[54]:


# # calculate cross-validated AUC
# from sklearn.model_selection import train_test_split
# cv_scores_roc = cross_val_score(logreg, features_final, asd_classes, cv=10, scoring='roc_auc').mean()
# cv_scores_roc.mean()


# # F-beta Score:

# # In[55]:


# logreg.fit(X_train, y_train)
# from sklearn.metrics import fbeta_score
# predictions_test = logreg.predict(X_test)
# fbeta_score(y_test, predictions_test, average='binary', beta=0.5)


# # # **Model Tuning**

# # In[56]:


# from sklearn.metrics import fbeta_score
# from sklearn.metrics import accuracy_score

# from sklearn.metrics import make_scorer
# from sklearn.svm import SVC
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split


# def f_beta_score(y_true, y_predict):
#     return fbeta_score(y_true, y_predict, beta = 0.5)


# # TODO: Initialize the classifier
# clf = SVC(random_state = 1)

# # TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# # HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
# parameters = {'C':range(1,6),'kernel':['linear','poly','rbf','sigmoid'],'degree':range(1,6)}

# # TODO: Make an fbeta_score scoring object using make_scorer()
# scorer = make_scorer(f_beta_score)

# # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
# grid_obj = GridSearchCV(estimator = clf, param_grid = parameters, scoring = scorer)

# # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
# grid_fit = grid_obj.fit(X_train, y_train)

# # Get the estimator
# best_clf = grid_fit.best_estimator_

# # Make predictions using the unoptimized and model
# predictions = (clf.fit(X_train, y_train)).predict(X_test)
# best_predictions = best_clf.predict(X_test)

# # Report the before-and-afterscores
# print ("Unoptimized model\n------")
# print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
# print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
# print ("\nOptimized Model\n------")
# print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
# print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# # # Extracting Feature Importance

# # In[57]:


# # TODO: Import a supervised learning model that has 'feature_importances_'
# from sklearn.ensemble import GradientBoostingClassifier
# from matplotlib import pyplot

# from sklearn.datasets import make_classification

# # TODO: Train the supervised model on the training set using .fit(X_train, y_train)
# model = GradientBoostingClassifier(random_state=0)
# model.fit(X_train, y_train)

# # TODO: Extract the feature importances using .feature_importances_ 
# importances = model.feature_importances_

# for i,v in enumerate(importances):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importances))], importances)
# pyplot.show()
# # Plot


# # In[58]:


# # TODO: Import a supervised learning model that has 'feature_importances_'
# from sklearn.ensemble import AdaBoostClassifier


# # TODO: Train the supervised model on the training set using .fit(X_train, y_train)
# model = AdaBoostClassifier(random_state=0)
# model.fit(X_train, y_train)

# # TODO: Extract the feature importances using .feature_importances_ 
# importances = model.feature_importances_
# for i,v in enumerate(importances):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importances))], importances)
# pyplot.show()
# # Plot


# # \\# Building a MLP model architecture

# # In[59]:


# # Imports
# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation


# np.random.seed(42)


# # In[ ]:


# # Building the model architecture with one layer of length 4


# model = Sequential()
# model.add(Dense(8, activation='relu', input_dim= 94))
# model.add(Dropout(0.2))
# model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
   
    
# model.summary()


# # In[ ]:


# # Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])


# # In[ ]:


# # Running and evaluating the model
# hist = model.fit(X_train, y_train,
#           batch_size=16,
#           epochs=100,
#           validation_data=(X_test, y_test), 
#           verbose=2)


# # # Evaluating the model

# # In[ ]:


# # Evaluating the model on the training and testing set
# score = model.evaluate(X_train, y_train)
# print("\n Training Accuracy:", score[1])

# score = model.evaluate(X_test, y_test, verbose=0)
# print("\n Testing accuracy: ", score[1])


# # ## **Conclusion**

# # Rebuilding the model without the 'result' variable.

# # In[ ]:


# # Split the data into features and target label
# asd_raw = asd_data['Class/ASD']
# features_raw = asd_data[['age', 'gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 
#                       'relation','A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score',
#                       'A9_Score','A10_Score']]


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# numerical = ['age']

# features_minmax_transform = pd.DataFrame(data = features_raw)
# features_minmax_transform[numerical] = scaler.fit_transform(features_raw[numerical])
# features_minmax_transform
# # Show an example of a record with scaling applied
# print(features_minmax_transform.head(n = 5))


# # In[ ]:


# #One-hot encode the 'features_minmax_transform' data using pandas.get_dummies()
# features_final = pd.get_dummies(features_minmax_transform)
# print(features_final.head(5))


# # Encode the 'all_classes_raw' data to numerical values
# asd_classes = asd_raw.apply(lambda x: 1 if x == 'YES' else 0)



# # Print the number of features after one-hot encoding
# encoded = list(features_final.columns)
# print("{} total features after one-hot encoding.".format(len(encoded)))

# # Uncomment the following line to see the encoded feature names
# print(encoded)


# # In[ ]:


# from sklearn.model_selection import train_test_split

# np.random.seed(1234)

# X_train, X_test, y_train, y_test = train_test_split(features_final, asd_classes, train_size=0.80, random_state=1)


# # Show the results of the split
# print ("Training set has {} samples.".format(X_train.shape[0]))
# print ("Testing set has {} samples.".format(X_test.shape[0]))


# # In[ ]:


# ### (1) Decision Trees

# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier

# dectree = DecisionTreeClassifier(random_state=1)

# # Train the classifier on the training set
# dectree.fit(X_train, y_train)


# # In[ ]:


# import pydotplus 


# dot_data = tree.export_graphviz(dectree,
#                                 out_file=None,
#                                 filled=True,
#                                 rounded=True,
#                                 special_characters=True)  
# graph = pydotplus.graph_from_dot_data(dot_data)  

# from IPython.print import Image 
# Image(graph.create_png()) 


# # In[ ]:


# from sklearn.model_selection import cross_val_score

# dectree = DecisionTreeClassifier(random_state=1)

# cv_scores = cross_val_score(dectree, features_final, asd_classes, cv=10)

# cv_scores.mean()


# # In[ ]:


# from sklearn.ensemble import RandomForestClassifier

# ranfor = RandomForestClassifier(n_estimators=5, random_state=1)
# cv_scores = cross_val_score(ranfor, features_final, asd_classes, cv=10)

# cv_scores.mean()


# # In[ ]:


# from sklearn import neighbors

# knn = neighbors.KNeighborsClassifier(n_neighbors=10)
# cv_scores = cross_val_score(knn, features_final, asd_classes, cv=10)

# cv_scores.mean()


# # In[ ]:


# # TODO: Import a supervised learning model that has 'feature_importances_'
# from sklearn.ensemble import AdaBoostClassifier


# # TODO: Train the supervised model on the training set using .fit(X_train, y_train)
# model = AdaBoostClassifier(random_state=0)
# model.fit(X_train, y_train)

# # TODO: Extract the feature importances using .feature_importances_ 
# importances = model.feature_importances_


# for i,v in enumerate(importances):
# 	print('Feature: %0d, Score: %.5f' % (i,v))
# # plot feature importance
# pyplot.bar([x for x in range(len(importances))], importances)
# pyplot.show()

import pickle
pickle.dump(ranfor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))