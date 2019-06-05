#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:19:25 2019

@author: kamesh
"""

import pandas as pd 
from sklearn import preprocessing
import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

name_clmns = ['checking account','Duration', 'Credit history', 'Purpose',\
              'Credit amount', 'savings', 'Present employment', 'Installment rate',\
              'Personal status', 'debtors', 'residence since', 'Property',\
              'Age', 'plans', 'housing', 'existing credits', 'Job', 'liable',\
              'Telephone', 'foreign worker','class']
data = pd.read_csv('/home/kamesh/Downloads/german.data',sep=' ',names=name_clmns)

data.columns = name_clmns

#Encoding label to int
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


data = MultiColumnLabelEncoder(columns = ['checking account', 'Credit history', 'Purpose','savings', 'Present employment', 'Personal status', 'debtors', 'Property', 'plans', 'housing', 'Job', 'Telephone', 'foreign worker','class']).fit_transform(data)

#Exploring the data
#print(data.info(), data.nunique())

trace0 = len(data[data['class']==0])
trace1 = len(data[data['class']==1])

data_table = [trace0, trace1]
objects = ('Good','Bad')
y_pos = np.arange(len(objects))
plt.bar(y_pos, data_table, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Rankings')
plt.title('Credit Distribution')
#plt.show()
corr = data.corr()

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#Feature Selection usnig tree based feature selection

def feature_selection(data):
    X = data.iloc[:,0:len(data.columns)-1]
    Y = data.iloc[:,-1]
    clf = ensemble.ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, Y)
    clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    select = SelectFromModel.get_support(model)
    cols = [i for i, x in enumerate(select) if not x]
    data.drop(data.columns[cols],axis=1,inplace=True)
    return X, Y, data

X, y, data = feature_selection(data)
X = data.iloc[:,0:len(data.columns)-1]
Y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0,shuffle = True)

#ANN for predection

X = data.iloc[:,0:len(data.columns)-1]
y= data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
# =============================================================================
# import numpy as np
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential() # Initialising the ANN

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(data.columns)-1))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 1, epochs = 50)


y_pred = classifier.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy for ANN: "+ str(accuracy*100)+"%")

#SVM Classiffier
from sklearn import svm
from sklearn.metrics import confusion_matrix
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy for svm: "+ str(accuracy*100)+"%")

#Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

#import lightgbm and xgboost 
import lightgbm as lgb 
import xgboost as xgb 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7,shuffle = True)
dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test)
parameters={'max_depth':7, 'eta':.1, 'silent':1,'objective':'binary:logistic','eval_metric':'logloss','learning_rate':.01}
num_round=100
xg=xgb.train(parameters,dtrain,num_round)
ypred=xg.predict(dtest)
ypred_xg = [ 1 if y>=0.4 else 0 for y in ypred ]
from sklearn.metrics import accuracy_score 
cm = confusion_matrix(y_test, np.asarray(ypred_xg))
print(cm)
accuracy_xgb = accuracy_score(y_test,ypred_xg) 
print("XGBOOT:",accuracy_xgb)

#LGBM
train_data=lgb.Dataset(X_train,label=y_train)
param = {'num_leaves':150, 'objective':'binary','max_depth':7,'learning_rate':.01,'max_bin':200}
param['metric'] = ['binary_logloss']
param['boosting_type'] = 'gbdt'
num_round=50
lgbm=lgb.train(param,train_data,num_round)
ypred2=lgbm.predict(X_test)
ypred2_lg = [ 1 if y>=0.4 else 0 for y in ypred2 ]
cm = confusion_matrix(y_test, np.asarray(ypred2_lg))
print(cm)
accuracy_lgbm = accuracy_score(ypred2_lg,y_test)
print("LGBM",accuracy_lgbm)

from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=50,depth=7,learning_rate=0.05)
model.fit(X_train, y_train)
pred_cat = model.predict(X_test)
cm = confusion_matrix(y_test, np.asarray(pred_cat))
print(cm)
accuracy_cat = accuracy_score(pred_cat,y_test)
print("Catboost",accuracy_cat)

stack = [1 if pred_cat[i] + ypred2_lg[i] + pred_cat[i] >=2 else 0 for i in range(len(ypred_xg))]
cm = confusion_matrix(y_test, np.asarray(stack))
print(cm)
accuracy_stack = accuracy_score(y_test,np.asarray(stack))
print("Stacking:",accuracy_stack)

stack_lgxg = []
for i in range(len(ypred2_lg)):
    if ypred_xg[i] != ypred2_lg[i]:
        diff1 = ypred[i] - 0.5
        diff2 = ypred2[i] - 0.5
        if abs(diff1) < abs(diff2):
            if diff2 < 0:
                stack_lgxg.append(0)
            else:
                stack_lgxg.append(1)
        elif diff1 == diff2:
            stack_lgxg.append(ypred2_lg[i])
        else:
            if diff1 < 0:
                stack_lgxg.append(0)
            else:
                stack_lgxg.append(1) 
    else:
        stack_lgxg.append(ypred_xg[i])
cm = confusion_matrix(y_test, stack_lgxg)
print(cm)
accuracy_stacklgxg = accuracy_score(y_test,np.asarray(stack_lgxg)) 
print("Stackingxglg:",accuracy_stacklgxg)
    