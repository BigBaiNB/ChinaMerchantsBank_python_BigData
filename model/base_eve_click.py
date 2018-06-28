from xgboost import plot_importance
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
import sys
import scipy as sp
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
OFF_LINE = False
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from data_path import *

def LogisticRegression_train(X,Y):
    # logisticRegression_wight = LogisticRegression(C=0.4,solver='liblinear').fit(X,Y)
    clf = XGBClassifier(n_estimators=30,max_depth=5).fit(X,Y)
    logisticRegression_model = SelectFromModel(clf,prefit=True,threshold_=1e-5)
    sh = logisticRegression_model.transform(X)
    return sh

def split_the_evt_lbl(string_evt):
    """
    拆分每个evt_lbl
    """
    x = 0
    list_string = string_evt.split('-')
    A = list_string[0]

    return A

tf_orignal = pd.read_csv(pre_temp)
train_x_except_tf = pd.read_csv(pre_temp_train_x).drop(tf_orignal.drop('USRID',axis=1).columns,axis=1)
train_x_except_tf['USRID'] = tf_orignal['USRID']

test_x_except_tf = pd.read_csv(pre_test_x).drop(tf_orignal.drop('USRID',axis=1).columns,axis=1)
test_x_except_tf['USRID'] = tf_orignal['USRID']

train_flg_data = pd.read_csv(train_flg)

all_tf_data = pd.merge(tf_orignal,train_flg_data,on='USRID',how='left')

tf_X = tf_orignal.drop('USRID')
tf_Y = all_tf_data['FLAG']

useful_tf = LogisticRegression_train(tf_X,tf_Y)

