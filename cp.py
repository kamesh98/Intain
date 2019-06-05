#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:23:10 2019

@author: kamesh
"""
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
import gc
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import ensemble 
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

data = pd.read_csv('/home/kamesh/Downloads/Dataset-2.csv',sep=',')
data_null = data.isna().sum()
data = data[data['Pan Number-Form 60/61'] != 'AXYPB7726K']
check = round(data.shape[0]/10)
delete = []
for key, values in data_null.items():
    if values > check:
        delete.append(key)
data.drop(delete, axis=1,inplace=True)
data = data.dropna(how='any',axis=0)
data = data.reset_index(drop=True)
data['N_DPD_BUCKET'] = np.nan
def dpdchanger(raw):
    if raw == 'Current':
        return 0
    else:
        return 1
data['N_DPD_BUCKET'] = data.apply(lambda x: dpdchanger(x['DPD Bucket']), axis=1)
data = data.drop(columns=['DPD Bucket', 'DPD'])
cols = data.columns
date_cols = []
name_cols = [] 
id1 = []
address = []
phone = []
for i in cols:
    if re.search('date', i, re.IGNORECASE): 
        date_cols.append(i)
    elif re.search('name', i, re.IGNORECASE): 
        name_cols.append(i)
    elif re.search('state', i, re.IGNORECASE): 
        address.append(i)
    if re.search('city', i, re.IGNORECASE): 
        address.append(i)
    if re.search('phone', i, re.IGNORECASE): 
        phone.append(i)
    if re.search('address', i, re.IGNORECASE): 
        address.append(i)
    if re.search('id', i, re.IGNORECASE): 
        id1.append(i)    
remove =[]
remove = ['SanctionReferenceNo', 'PinCode2', 'Pan Number-Form 60/61', 'PinCode1', 'Document Code 2 (Addr)']
remove += name_cols +id1 + address + phone
num_cols = data._get_numeric_data().columns
cat_data = list(set(cols) - set(num_cols) - set(date_cols) - set(remove))

data = data.drop(columns=remove)
def lookup(s):
    dates = {date:pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)

for i in date_cols:
    data[i] = lookup(data[i])


label = []
hot = []
remove1 = []
for i in cat_data:
    cnt = data[i].nunique()
    if cnt == 2 or cnt > 30:
        label.append(i)
    elif cnt ==1:
        remove1.append(i)
    else:
        hot.append(i)
data = data.drop(columns=remove1)
print(label,hot)

data = pd.get_dummies(data, columns=hot, drop_first=True)
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        '''

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

data = MultiColumnLabelEncoder(columns = label).fit_transform(data)

data['age'] = np.nan
data['accounttoloan'] = np.nan
data['procssingtime'] = np.nan
data['loanemi'] = np.nan

def diffrence_days(start,end):
    return (start-end).days
def diffrence_years(today,born):
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
from datetime import date
today = date.today()
data['age'] = data.apply(lambda x: diffrence_years(today, x['DateOfBirth']), axis=1)
data['accounttoloan'] = data.apply(lambda x: diffrence_days(x['LoanSanctionDate'], x['Account Opening Date']), axis=1)
data['procssingtime'] = data.apply(lambda x: diffrence_days(x['Process Date'], x['DocumentDate']), axis=1)
data['loanemi'] = data.apply(lambda x: diffrence_days(x['LoanSanctionDate'], x['EMI Start Date']), axis=1)
data = data.drop(columns=date_cols)

data = data.reindex(list([a for a in data.columns if a != 'N_DPD_BUCKET'] + ['N_DPD_BUCKET']), axis=1)
Y = data.iloc[:,-1]
X = data.iloc[:,0:len(data.columns)-1]

clf = ensemble.ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, Y)
clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
select = SelectFromModel.get_support(model)
cols = [i for i, x in enumerate(select) if not x]
data.drop(data.columns[cols],axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.model_selection import StratifiedShuffleSplit

stratified = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

for train_set, test_set in stratified.split(X, Y):
    stratified_train = data.loc[train_set]
    stratified_test = data.loc[test_set]
    print(train_set, test_set)
train_df = stratified_train
test_df = stratified_test

train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

x_train = train_df.drop('N_DPD_BUCKET', axis=1)
y_train = train_df['N_DPD_BUCKET']


x_test = test_df.drop('N_DPD_BUCKET', axis=1)
y_test = test_df['N_DPD_BUCKET']

del train_df
del test_df
del stratified_train
del stratified_test
gc.collect()
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
sm = SMOTE(random_state=12, ratio = 1.0)
x_train, y_train = sm.fit_sample(x_train, y_train)
x_test, y_test = sm.fit_sample(x_test, y_test)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
param = {'num_leaves':70, 'objective':'binary','max_depth':15,'learning_rate':.01,'max_bin':200}
param['metric'] = 'binary_logloss'
param['boosting_type'] = 'gbdt'
num_round=75
gbm = lgb.train(param,
                lgb_train,
                num_boost_round=num_round,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
ypred2_lg = [ 1 if y>=0.4855 else 0 for y in y_pred ]
cm = confusion_matrix(y_test, ypred2_lg)
print(cm)
accuracy_lgbm = accuracy_score(ypred2_lg,y_test)
print("LGBM",accuracy_lgbm)

import xgboost as xgb 
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)
parameters={'max_depth':15, 'eta':.1, 'silent':1,'objective':'binary:logistic','eval_metric':'logloss','learning_rate':.01}
num_round=50
xg=xgb.train(parameters,dtrain,num_round)
ypred=xg.predict(dtest)
ypred_xg = [ 1 if y>=0.4855 else 0 for y in ypred ]
from sklearn.metrics import accuracy_score 
cm = confusion_matrix(y_test, np.asarray(ypred_xg))
print(cm)
accuracy_xgb = accuracy_score(y_test,ypred_xg) 
print("XGBOOT:",accuracy_xgb)

from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=50,depth=7,learning_rate=0.05)
model.fit(x_train, y_train)
pred_cat = model.predict(x_test)
cm = confusion_matrix(y_test, np.asarray(pred_cat))
print(cm)
accuracy_cat = accuracy_score(pred_cat,y_test)
print("Catboost",accuracy_cat)

stack = [1 if pred_cat[i] + ypred2_lg[i] + ypred_xg[i] >=2 else 0 for i in range(len(ypred_xg))]
cm = confusion_matrix(y_test, np.asarray(stack))
print(cm)
accuracy_stack = accuracy_score(y_test,np.asarray(stack))
print("Stacking:",accuracy_stack)

stack_lgxg = []
for i in range(len(ypred2_lg)):
    if ypred_xg[i] != ypred2_lg[i]:
        diff1 = y_pred[i] - 0.485
        diff2 = ypred[i] - 0.485
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

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred_log = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred_log)
print(confusion)