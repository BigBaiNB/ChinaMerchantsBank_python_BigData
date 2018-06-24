# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 18:12:05 2018

@author: puxuefei
"""

# -*- coding: utf-8 -*-
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
def xgb_model(train_set_x,train_set_y,test_set_x):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eta': 0.02,
              'max_depth': 4,  # 4 3 (3-10)
              'colsample_bytree': 0.8,#0.8 (0.1]
              'subsample': 0.7,
              'min_child_weight': 9,  # 2 3
              'silent':1
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=900)
    predict = model.predict(dvali)
    return predict

def count_none0(s):
    count_none0 = 0
    for i in s:
        if i != 0:
            count_none0 += 1
    return count_none0

def count_0(s):
    count_0 = 0
    for i in s:
        if i == 0:
            count_0 += 1
    return count_0

    
    
def log_EVT_LBL(data):
    EVT_LBL_len = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_len':len})
    EVT_LBL_set_len = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_set_len':lambda x:len(set(x))})
    
    data['hour'] = data.OCC_TIM.map(lambda x:x.hour)
    data['day'] = data.OCC_TIM.map(lambda x:x.day)
    
    return EVT_LBL_len,EVT_LBL_set_len

def log_OCC_TIM(data,recenttime):
    recent_time = recenttime
    time = data.groupby(['USRID'],as_index=False)['OCC_TIM'].agg({'recenttime':max})
    time['time_gap'] = (recent_time-time['recenttime']).dt.total_seconds()
    
    df_log = train_log.sort_values(['USRID','OCC_TIM'])
    df_log['next_time'] = data.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)
    df_log['next_time'] = df_log.next_time.dt.total_seconds()
    log = df_log.groupby(['USRID'],as_index=False)['next_time'].agg({
        'next_time_mean':np.mean,
        'next_time_std':np.std,
        'next_time_min':np.min,
        'next_time_max':np.max
    })
    log_temp = log.drop(['USRID'],axis=1)
    log['next_time_cuont0'] = log_temp.apply(count_0,axis=1)
    log['next_time_cuontnone0'] = log_temp.apply(count_none0,axis=1)
    time = pd.merge(time,log,on='USRID',how='left')
    data['dayofweek'] = data.OCC_TIM.dt.dayofweek
    df_dw = data.groupby(['USRID','dayofweek'])['USRID'].count().unstack()
    df_dw['dw_count0'] = df_dw.apply(count_0,axis=1)
    df_dw['dw_countnone0'] = df_dw.apply(count_none0,axis=1)
    df_dw.reset_index(inplace=True)
    time = pd.merge(time,df_dw,on='USRID',how='left')
    df_day = data.groupby(['USRID','day'])['USRID'].count().unstack()
    df_day['day_count0'] = df_day.apply(count_0,axis=1)
    df_day['day_countnone0'] = df_day.apply(count_none0,axis=1)
    df_day.reset_index(inplace=True)
    time = pd.merge(time,df_day,on='USRID',how='left')
    df_hour = data.groupby(['USRID','hour'])['USRID'].count().unstack()
    df_hour['hour_count0'] = df_hour.apply(count_0,axis=1)
    df_hour['hour_countnone0'] = df_hour.apply(count_none0,axis=1)
    df_hour.reset_index(inplace=True)
    time = pd.merge(time,df_hour,on='USRID',how='left')
    
    return time

def log_TYPE(data):
    df_log_type = data.groupby(['USRID','TCH_TYP'])['USRID'].count().unstack()
    df_log_type['df_logtype_count0'] = df_log_type.apply(count_0,axis=1)
    df_log_type['df_logtype_countnone0'] = df_log_type.apply(count_none0,axis=1)
    df_log_type.reset_index(inplace=True) 
    df_log_type.fillna(0,inplace=True)
    return df_log_type

def feture_imp(data_log,data_flag,n):
    df = data_log.groupby(['USRID','EVT_LBL'])['USRID'].count().unstack()
    df_c = pd.merge(df,data_flag,left_index=True,right_on='USRID',how='right')
    df_c.fillna(0,inplace=True)
    x = df_c.drop(['USRID','FLAG'],axis=1)
    y = df_c['FLAG']
    clf = XGBClassifier(n_estimators=30,max_depth=5)
    clf.fit(x,y)
    imp = clf.feature_importances_
    names = x.columns
    d={}
    for i in range(len(names)):
        d[names[i]] = imp[i]
    d = sorted(d.items(),key=lambda x:x[1],reverse=True)
    d = d[0:n]
    feture_list=[j[0] for j in d]
    return feture_list
    

def log_EVTLBL_STA(data,feature_list):
    df = data.groupby(['USRID','EVT_LBL'])['USRID'].count().unstack()
    df_new = pd.DataFrame()
    for i in feature_list:
        try:
             df_new[i] = df[i]
        except:
            df_new[i] = 0
    df_new.index = df.index
    df_new['df_new_count0'] = df_new.apply(count_0,axis=1)
    df_new['df_new_countnone0'] = df_new.apply(count_none0,axis=1)       
    return df_new


            
   

train_agg = pd.read_csv('E:/工商银行比比赛/train_agg.csv',sep='\t',engine='python')
train_flg = pd.read_csv('E:/工商银行比比赛/train_flg.csv',sep='\t',engine='python')
train_log = pd.read_csv('E:/工商银行比比赛/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'],engine='python')
    
recenttime =  max(train_log.OCC_TIM)   
        
all_train = pd.merge(train_flg,train_agg,on=['USRID'],how='left')
EVT_LBL_len,EVT_LBL_set_len = log_EVT_LBL(train_log)
time = log_OCC_TIM(train_log,recenttime)
log_type = log_TYPE(train_log)
feature_list = feture_imp(train_log,train_flg,25)
df_evtblb_sta = log_EVTLBL_STA(train_log,feature_list)

all_train = pd.merge(all_train,EVT_LBL_len,on=['USRID'],how='left')
all_train = pd.merge(all_train,EVT_LBL_set_len,on=['USRID'],how='left')
all_train = pd.merge(all_train,time,on=['USRID'],how='left')
all_train = pd.merge(all_train,log_type,on='USRID',how='left')
all_train = pd.merge(all_train,df_evtblb_sta,left_on='USRID',right_index=True,how='left')
all_train.time_gap.fillna(max(all_train.time_gap)+1,inplace=True)
all_train.fillna(0,inplace=True)
 
train_x = all_train.drop(['USRID', 'FLAG','recenttime'], axis=1).values
train_y = all_train['FLAG'].values

auc_list = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
for train_index, test_index in skf.split(train_x, train_y):
    print('Train: %s | test: %s' % (train_index, test_index))
    X_train, X_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]

    pred_value = xgb_model(X_train, y_train, X_test)
    print(pred_value)
    print(y_test)

    pred_value = np.array(pred_value)
    pred_value = [ele + 1 for ele in pred_value]

    y_test = np.array(y_test)
    y_test = [ele + 1 for ele in y_test]

    fpr, tpr, thresholds = roc_curve(y_test, pred_value, pos_label=2)
    
    auc = metrics.auc(fpr, tpr)
    print('auc value:',auc)
    auc_list.append(auc)

print('validate result:',np.mean(auc_list))



test_agg = pd.read_csv('E:/工商银行比比赛/test_agg.csv',sep='\t',engine='python')
test_log = pd.read_csv('E:/工商银行比比赛/test_log.csv',sep='\t',parse_dates = ['OCC_TIM'],engine='python')

EVT_LBL_len,EVT_LBL_set_len = log_EVT_LBL(test_log)
time = log_OCC_TIM(test_log,recenttime)
log_type = log_TYPE(test_log)
df_evtblb_sta = log_EVTLBL_STA(test_log,feature_list)

test_set = pd.merge(test_agg,EVT_LBL_len,on=['USRID'],how='left')
test_set = pd.merge(test_set,EVT_LBL_set_len,on=['USRID'],how='left')
test_set = pd.merge(test_set,time,on=['USRID'],how='left')
test_set = pd.merge(test_set,log_type,on='USRID',how='left')
test_set = pd.merge(test_set,df_evtblb_sta,left_on='USRID',right_index=True,how='left')
test_set.time_gap.fillna(max(test_set.time_gap)+1,inplace=True)
test_set.fillna(0,inplace=True)
  
    
    
 ###########################
result_name = test_set[['USRID']]
train_x = all_train.drop(['USRID', 'FLAG','recenttime'], axis=1).values
train_y = all_train['FLAG'].values

test_x = test_set.drop(['USRID','recenttime'], axis=1).values
test_x = test_x

pred_result = xgb_model(train_x,train_y,test_x)
result_name['RST'] = pred_result
result_name.to_csv('d:/test_result.csv',index=None,sep='\t')