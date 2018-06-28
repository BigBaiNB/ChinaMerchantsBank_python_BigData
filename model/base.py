# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 18:12:05 2018

@author: puxuefei
"""

# -*- coding: utf-8 -*-
from data_path import result_path,original_test_agg,original_test_log,original_train_agg,original_train_flg,original_train_log,pre_test_not_in_data,pre_test_both_have_data,pre_train_both_have_data,pre_train_not_in_data,pre_temp,pre_temp_train_x,pre_test_x
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


def xgb_model_not(train_set_x,train_set_y,test_set_x):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eta': 0.02,
              'max_depth': 4,  # 4 3 (3-10)
              'colsample_bytree': 0.8,#0.8 (0.1]
              'subsample': 0.7,
              'min_child_weight': 12,  # 2 3
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
    EVT_LBL_len = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_len':len})#点击次数
    EVT_LBL_set_len = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_set_len':lambda x:len(set(x))})#模块种类
    
    data['hour'] = data.OCC_TIM.map(lambda x:x.hour)
    data['day'] = data.OCC_TIM.map(lambda x:x.day)
    
    return EVT_LBL_len,EVT_LBL_set_len

def log_OCC_TIM(data,recenttime):
    
    tem = pd.DataFrame()

    recent_time = recenttime
    time = data.groupby(['USRID'],as_index=False)['OCC_TIM'].agg({'recenttime':max})
    """
    如果为了方便索引：将USRID作为索引，那就as_index=True，只需  df.loc['bk1']就可查到   as_index=False那时你将不得不像这样得到它： df.loc[df.books=='bk1']
    如果为了整合数据，还是为False
    """
    time['time_gap'] = (recent_time-time['recenttime']).dt.total_seconds()#与最大的时间做差值
    
    df_log = train_log.sort_values(['USRID','OCC_TIM'])
    df_log['next_time'] = data.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)#前后两次时间的差值，取绝对值
    df_log['next_time'] = df_log.next_time.dt.total_seconds()#将差值转化为秒
    log = df_log.groupby(['USRID'],as_index=False)['next_time'].agg({
        'next_time_mean':np.mean,#求均值
        'next_time_std':np.std,#计算标准差，标准差能反映一个数据集的离散程度
        'next_time_min':np.min,
        'next_time_max':np.max
    })
    log_temp = log.drop(['USRID'],axis=1)
    # log['next_time_cuont0'] = log_temp.apply(count_0,axis=1)#统计一行中每列为0的个数
    # log['next_time_cuontnone0'] = log_temp.apply(count_none0,axis=1)#统计一行中每列非0的个数
    tem['next_time_cuont0'] = log_temp.apply(count_0,axis=1)
    tem['next_time_cuontnone0'] = log_temp.apply(count_none0,axis=1)
    log = log.join(tem)
    tem = pd.DataFrame()#清空

    time = pd.merge(time,log,on='USRID',how='left')


    data['dayofweek'] = data.OCC_TIM.dt.dayofweek#日期转化为星期,并增加一列
    df_dw = data.groupby(['USRID','dayofweek'])['USRID'].count().unstack().fillna(0)#count按名字做serial，unstack变成列
    # print('groupby(['USRID','dayofweek'])['USRID'].count().unstack()',df_dw,sep'\n')
    # df_dw['dw_count0'] = df_dw.apply(count_0,axis=1)
    # df_dw['dw_countnone0'] = df_dw.apply(count_none0,axis=1)
    tem['dw_count0'] = df_dw.apply(count_0,axis=1)
    tem['dw_countnone0'] = df_dw.apply(count_none0,axis=1)
    df_dw = df_dw.join(tem)
    tem = pd.DataFrame()#清空

    df_dw.reset_index(inplace=True)#增加索引

    time = pd.merge(time,df_dw,on='USRID',how='left')

    df_day = data.groupby(['USRID','day'])['USRID'].count().unstack().fillna(0)
    # df_day['day_count0'] = df_day.apply(count_0,axis=1)
    # df_day['day_countnone0'] = df_day.apply(count_none0,axis=1)
    tem['day_count0'] = df_day.apply(count_0,axis=1)
    tem['day_countnone0'] = df_day.apply(count_none0,axis=1)
    df_day = df_day.join(tem)
    tem = pd.DataFrame()#清空

    df_day.reset_index(inplace=True)

    time = pd.merge(time,df_day,on='USRID',how='left')

    df_hour = data.groupby(['USRID','hour'])['USRID'].count().unstack().fillna(0)
    # df_hour['hour_count0'] = df_hour.apply(count_0,axis=1)
    # df_hour['hour_countnone0'] = df_hour.apply(count_none0,axis=1)
    tem['hour_count0'] = df_hour.apply(count_0,axis=1)
    tem['hour_countnone0'] = df_hour.apply(count_none0,axis=1)
    df_hour = df_hour.join(tem)
    tem = pd.DataFrame()#清空

    df_hour.reset_index(inplace=True)
    time = pd.merge(time,df_hour,on='USRID',how='left')
    
    return time
"""
存在问题：
第一：每个apply
"""
def log_TYPE(data):
    tem =pd.DataFrame()
    df_log_type = data.groupby(['USRID','TCH_TYP'])['USRID'].count().unstack().fillna(0)
    # df_log_type['df_logtype_count0'] = df_log_type.apply(count_0,axis=1)
    # df_log_type['df_logtype_countnone0'] = df_log_type.apply(count_none0,axis=1)
    tem['df_logtype_count0'] = df_log_type.apply(count_0,axis=1)
    tem['df_logtype_countnone0'] = df_log_type.apply(count_none0,axis=1)
    df_log_type = df_log_type.join(tem)
    tem = pd.DataFrame()#清空

    df_log_type.reset_index(inplace=True) 
    df_log_type.fillna(0,inplace=True)
    return df_log_type

def feture_imp(data_log,data_flag,n):
    df = data_log.groupby(['USRID','EVT_LBL'])['USRID'].count().unstack()
    # df_c = pd.merge(df,data_flag,left_index=True,right_on='USRID',how='right')
    df_c = pd.merge(df,data_flag,left_index=True,right_on='USRID',how='left')#只取有evt的
    df_c.fillna(0,inplace=True)
    x = df_c.drop(['USRID','FLAG'],axis=1)
    y = df_c['FLAG']
    clf = XGBClassifier(n_estimators=30,max_depth=5)
    clf.fit(x,y)
    imp = clf.feature_importances_
    names = x.columns#取得列名
    d={}
    for i in range(len(names)):#获得每列在模型中的权重
        d[names[i]] = imp[i]
    """
    Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。
    http://www.runoob.com/python/att-dictionary-items.html

    sort 与 sorted 区别：

    sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。

    list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的
    是一个新的 list，而不是在原来的基础上进行的操作。

    sorted(iterable[, cmp[, key[, reverse]]])
    参数说明：

    iterable -- 可迭代对象。
    cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
    key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。

    """
    d = sorted(d.items(),key=lambda x:x[1],reverse=True)
    d = d[0:n]
    feture_list=[j[0] for j in d]#取出第一个
    return feture_list
    

def log_EVTLBL_STA(data,feature_list):
    df = data.groupby(['USRID','EVT_LBL'])['USRID'].count().unstack().fillna(0)
    df_new = pd.DataFrame()
    for i in feature_list:
        try:
             df_new[i] = df[i]
        except:
            df_new[i] = 0
    df_new.index = df.index
    tem = pd.DataFrame()
    # df_new['df_new_count0'] = df_new.apply(count_0,axis=1)
    # df_new['df_new_countnone0'] = df_new.apply(count_none0,axis=1)   
    tem['df_new_count0'] = df_new.apply(count_0,axis=1)
    tem['df_new_countnone0'] = df_new.apply(count_none0,axis=1)
    df_new.join(tem)  
    return df_new


def K_straf(train_x,train_y,model):# k折交叉切分,训练模型
    
    auc_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)# k折交叉切分,n_splits表示结果有几块
    for train_index, test_index in skf.split(train_x, train_y):#注意返回的是拆分的行号所组成的列表
        print('Train: %s | test: %s' % (train_index, test_index))
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]

        pred_value = model(X_train, y_train, X_test)
        print(pred_value)
        print(y_test)

        pred_value = np.array(pred_value)
        pred_value = [ele + 1 for ele in pred_value]

        y_test = np.array(y_test)
        y_test = [ele + 1 for ele in y_test]

        fpr, tpr, thresholds = roc_curve(y_test, pred_value, pos_label=2)#pos_label：输出积极标签值
        """
        ROC曲线指受试者工作特征曲线/接收器操作特性(receiver operating characteristic，ROC)曲线,
        是反映灵敏性和特效性连续变量的综合指标,是用构图法揭示敏感性和特异性的相互关系，它通过将连续变量设定
        出多个不同的临界值，从而计算出一系列敏感性和特异性。ROC曲线是根据一系列不同的二分类方式（分界值或决定阈），
        以真正例率（也就是灵敏度）（True Positive Rate,TPR）为纵坐标，假正例率（1-特效性）（False Positive Rate,FPR）
        为横坐标绘制的曲线。

        ROC观察模型正确地识别正例的比例与模型错误地把负例数据识别成正例的比例之间的权衡。TPR的增加以FPR的增加为代价。
        ROC曲线下的面积是模型准确率的度量，AUC（Area under roccurve）。

        纵坐标：真正率（True Positive Rate , TPR）或灵敏度（sensitivity）

        TPR = TP /（TP + FN）  （正样本预测结果数 / 正样本实际数）

        横坐标：假正率（False Positive Rate , FPR）

        FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）
        """
        
        auc = metrics.auc(fpr, tpr)
        print('auc value:',auc)
        auc_list.append(auc)

    print('validate result:',np.mean(auc_list))    


####################  
def tf_idf_rvt(data_tfv):
    """
    通过tf_idf重新建立模型
    返回df
    """

    """
    使用tf-idf提取文档特征，也就是表示文档，基于侧袋模型 
    1、演示tf-idf的用法，关键词提取 
    参数信息：
    输入：list of str   str是用空格分割的好的单词 输入数据格式为： list of list[word] 或者 list of string[必须满足是空格切分的单词]
    1.1、smooth_idf: idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
    1.2、max_features：默认为None，可设为int，对所有关键词的ifidf进行降序排序，只取前max_features个作为关键词集
    1.3、min_df：包含词语的文档最小个数，如果某个词的document frequence小于min_df，则这个词不会被当作关键词，如果是float类型[0,1],则表示文档频率
    1.4、max_df：同上
    1.5、itidf用于表示文本 也就是content
    1.6、ngram_range：（1，3）表示3-1个特征进行组合，生成新的特征 
    """
    result_df = data_tfv[['USRID']]
    tfv = TfidfVectorizer(min_df=1, max_df=0.95, sublinear_tf=True, stop_words='english')
    # 此时获得的特征过于稀疏，维度太高，不能直接使用
    # 训练tfidf模型,统计了每个单词的在每篇文章中的tfidf
    document = data_tfv['EVE_List']
    tfv.fit(document)
    # 返回的result是一个matrix，行是文本的个数，列是词语的个数
    result = tfv.transform(document)#注意返回的是一个矩阵
    # size = result.shape[1]

    tf_idf_featuer = pd.DataFrame(data=result.todense(),columns=tfv.get_feature_names())
    result_df = result_df.join(tf_idf_featuer)
    # print(result.shape)
    return result_df


"""
整个EVE做tf，idf
"""

def split_tf_EVE_all_data(data_orginal_tfv):
    result_data = data_orginal_tfv.groupby(['USRID'],as_index=False)['EVT_LBL'].agg({'EVE_List':lambda x:' '.join(x)})
    result = tf_idf_rvt(result_data)
    return result

####################



# train_agg = pd.read_csv('E:/工商银行比比赛/train_agg.csv',sep='\t',engine='python')
# train_flg = pd.read_csv('E:/工商银行比比赛/train_flg.csv',sep='\t',engine='python')
# train_log = pd.read_csv('E:/工商银行比比赛/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'],engine='python')
train_agg = pd.read_csv(pre_train_both_have_data,sep=',',engine='python')#均含有的
train_flg = pd.read_csv(original_train_flg,sep='\t',engine='python')
train_log = pd.read_csv(original_train_log,sep='\t',parse_dates = ['OCC_TIM'],engine='python')
test_log = pd.read_csv(original_test_log,sep='\t',parse_dates = ['OCC_TIM'],engine='python')

recenttime =  max(train_log.OCC_TIM)   
        
all_train = pd.merge(train_flg,train_agg,on=['USRID'],how='right')#all_train包含agg与flg

EVT_LBL_len,EVT_LBL_set_len = log_EVT_LBL(train_log)#获得点击次数，种类的个数，增加hour，day

time = log_OCC_TIM(train_log,recenttime)#获得时间特征
log_type = log_TYPE(train_log)#获得访问方式
feature_list = feture_imp(train_log,train_flg,25)#获得EVE中前25重要的特征
df_evtblb_sta = log_EVTLBL_STA(train_log,feature_list)#重新建立特征，只取得EVE中前25


tf_idf_orignal_data = train_log.append(test_log)
tf_idf_result = split_tf_EVE_all_data(tf_idf_orignal_data)#tf_idft特征
# print('tf_idf_shape:',tf_idf_result.shape)
# tf_idf_result.to_csv(pre_temp,index=False)

all_train = pd.merge(all_train,tf_idf_result,on=['USRID'],how='left')
all_train = pd.merge(all_train,EVT_LBL_len,on=['USRID'],how='left')#增加点击次数
all_train = pd.merge(all_train,EVT_LBL_set_len,on=['USRID'],how='left')#增加点击种类
all_train = pd.merge(all_train,time,on=['USRID'],how='left')#增加各种时间特征
all_train = pd.merge(all_train,log_type,on='USRID',how='left')#增加访问种类
all_train = pd.merge(all_train,df_evtblb_sta,left_on='USRID',right_index=True,how='left')#获得重要的25维EVE
all_train.time_gap.fillna(max(all_train.time_gap)+1,inplace=True)#对于没有时间特征的填充最大时间间隔+1
all_train.fillna(0,inplace=True)
 
train_x = all_train.drop(['USRID', 'FLAG','recenttime'], axis=1).values#取数据变为array
train_y = all_train['FLAG'].values
# K_straf(train_x,train_y,xgb_model)
#￥
# print('all_train',all_train.shape)



test_agg = pd.read_csv(pre_test_both_have_data,sep=',',engine='python')


EVT_LBL_len,EVT_LBL_set_len = log_EVT_LBL(test_log)
time = log_OCC_TIM(test_log,recenttime)
log_type = log_TYPE(test_log)
df_evtblb_sta = log_EVTLBL_STA(test_log,feature_list)


test_set = pd.merge(test_agg,EVT_LBL_len,on=['USRID'],how='left')
test_set = pd.merge(test_set,tf_idf_result,on=['USRID'],how='left')
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

# print('train_x_shape:',train_x.shape)
# print('test_x_shape:',test_x.shape)
# all_train.drop(['USRID', 'FLAG','recenttime'], axis=1).to_csv(pre_temp_train_x,index=False)
# test_set.drop(['USRID','recenttime'], axis=1).to_csv(pre_test_x,index=False)

pred_result = xgb_model(train_x,train_y,test_x)
result_name['RST'] = pred_result


########################################
train_without_in_x = pd.read_csv(pre_train_not_in_data,sep=',',engine='python')
all_data = pd.merge(train_without_in_x,train_flg,on='USRID',how='left')
train_x_data = all_data.drop(['USRID','FLAG'],axis=1).values
train_y_data = all_data['FLAG'].values

# K_straf(train_x_data,train_y_data,xgb_model_not)
#$

test_without_in_x = pd.read_csv(pre_test_not_in_data,sep=',',engine='python')
result_without_name = test_without_in_x[['USRID']]

all_data_test = test_without_in_x.drop(['USRID'],axis=1).values
pre_without = xgb_model_not(train_x_data,train_y_data,all_data_test)

result_without_name['RST'] = pre_without
result_name = result_name.append(result_without_name)

result_name.to_csv(result_path,index=None,sep='\t')
print(".....finsh")

