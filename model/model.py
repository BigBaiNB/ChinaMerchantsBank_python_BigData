import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV  

from data_path import *

def micri_avg_f1(y_true,y_pred):
    """
    评分函数
    """
    return roc_auc_score(y_true,y_pred)

def get_X_Y():
    """
    通过切片的方式获得X，Y
    """
    data_train = pd.read_csv(train_agg,sep='\t')
    data_result = pd.read_csv(train_flg,sep='\t')
    all_data = pd.merge(data_train,data_result,on='USRID')
    X = all_data.iloc[:,:-2]#切片
    Y = all_data.iloc[:,-1]#切片

"""
X_train,X_test, y_train, y_test =cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)

cross_validatio为交叉验证
参数解释：
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
"""

def Cross_validation_data(X,Y,train_way):
    """
    交叉验证
    """
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.6,random_state=0 )#交叉验证

    # train_way = LogisticRegression(C=0.5,solver='liblinear')
    train_way.fit(x_train,y_train)

    y_pred = train_way.predict(x_test)#按概率输出

    print(micri_avg_f1(y_test,train_way.predict(x_test)))
    print(y_pred)


# print(Y[456])
# # print(all_data)
# print(X.shape,Y.shape,sep='|')
# print(all_data.shape)

# train_matrix = np.delete(data_train,0,axis=0)#axis=0代表删除行，=1代表删除列

# intSet = train_matrix[:,30].argsort(0)
# print(intSet)

# train_matrix = train_matrix[]
# print(train_matrix)

def creat_V_name():
    name_list = []
    for i in range(1,30):
        name_list.append('V'+str(i))
    return name_list

def silpt_the_agg_and_log_data(input_agg_file_name,input_time_file_name,out_both_have_file,out_not_in_file):
    """
    将agg中有log的和没有log的部分 分别分割出来
    """
    agg_data = pd.read_csv(input_agg_file_name)
    log_data = pd.read_csv(input_time_file_name)

    both_have_data = pd.merge(agg_data,log_data,how='inner',on='USRID')
    both_have_data.pop('Time')
    #注意：not_in_agg中列名是乱序了,需要按列名重新排序
    not_in_agg = agg_data.append(log_data)
    not_in_agg.drop_duplicates(subset=['USRID'],keep=False,inplace=True)
    not_in_agg.pop('Time')
    not_in_agg.rank(axis=1)#列名排序,按的字符串，所以排序后还是乱的


    """
    drop_duplicates
    代码中subset对应的值是列名，表示只考虑这两列，将这两列对应值相同的行进行去重。默认值为subset=None表示考虑所有列。

    keep='first'表示保留第一次出现的重复行，是默认值。keep另外两个取值为"last"和False，分别表示保留最后一次出现的重复行和去除所有重复行。

    inplace=True表示直接在原来的DataFrame上删除重复项，而默认值False表示生成一个副本。
    """

    
    not_in_agg.to_csv(out_not_in_file,index=False,columns=creat_V_name().extend(['USRID']))
    both_have_data.to_csv(out_both_have_file,index=False)
    # return both_have_data,not

# silpt_the_agg_and_log_data(test_agg,pre_test_Time,pre_test_both_have_data,pre_test_not_in_data)
# silpt_the_agg_and_log_data(train_agg,pre_train_Time,pre_train_both_have_data,pre_train_not_in_data)

def pretict_withot_feature(both_have_file,not_in_agg_file,log_file):
    """
    预测缺失的参数
    并将参数填满后反回
    """
    both_have_data = pd.read_csv(both_have_file)
    not_in_data = pd.read_csv(not_in_agg_file)
    log_data = pd.read_csv(log_file)

    new_data = pd.merge(both_have_data,log_data,how='left',on='USRID')


    # print(new_data)
    column_size = new_data.columns.size#获取列数

    #需要调整的参数
    param_test_1 = {  
        'n_estimators': range(10, 100, 2)
    }

    param_test_2 = {
        'min_samples_leaf':range(1, 30, 1),
        'min_samples_split':range(20,201,10)
    }

    param_test_3 = {
        'max_features': range(1,25,1)
    }

    parameter = {
        'max_features': 1,
        'min_samples_leaf':1,
        'min_samples_split':2,
        'n_estimators': 10
    }

    for i in range(column_size-31):
        X = both_have_data.iloc[:,:30]
        Y = new_data.iloc[:,i+31]

        """
        gsearch1.grid_scores_,gsearch1.best_params_, gsearch1.best_score_  
        #输出结果如下：  
        ([mean:0.80681, std: 0.02236, params: {'n_estimators': 10},  
        mean: 0.81600, std: 0.03275, params:{'n_estimators': 20},  
        mean: 0.81818, std: 0.03136, params:{'n_estimators': 30},  
        mean: 0.81838, std: 0.03118, params:{'n_estimators': 40},  
        mean: 0.82034, std: 0.03001, params:{'n_estimators': 50},  
        mean: 0.82113, std: 0.02966, params:{'n_estimators': 60},  
        mean: 0.81992, std: 0.02836, params:{'n_estimators': 70}],  
        {'n_estimators':60},  
        0.8211334476626017)  
        """
        train_way = RandomForestRegressor(n_estimators=parameter['n_estimators'],min_samples_leaf=parameter['min_samples_leaf'],max_features=parameter['max_features'],min_samples_split=parameter['min_samples_split'],random_state=50,oob_score=True)
        for i in range(1,3):
            param_test = param_test_1
            if i==2:
                param_test = param_test_2
            elif i==3:
                param_test = param_test_3

            gsearch = GridSearchCV(train_way, param_grid=param_test,cv=5 ) 
            gsearch.fit(X,y=Y)
            parameter.update(gsearch.best_params_)
            train_way = RandomForestRegressor(n_estimators=parameter['n_estimators'],min_samples_leaf=parameter['min_samples_leaf'],max_features=parameter['max_features'],min_samples_split=parameter['min_samples_split'],random_state=50,oob_score=True)
            print(i)
        # train_way = RandomForestRegressor(n_estimators=parameter['n_estimators'],min_samples_leaf=parameter['min_samples_leaf'],max_features=parameter['max_features'],min_samples_split=parameter['min_samples_split'],random_state=50,oob_score=True)
        predict_y = train_way.predict(not_in_data)
        # print(type(predict_y)) 返回是一个array
        # Cross_validation_data(X,Y,train_way)
        print(predict_y)
        print(parameter)
        pd.DataFrame(predict_y.tolist()).to_csv({'Time':pre_temp},index=False)
        break


pretict_withot_feature(pre_test_both_have_data,pre_test_not_in_data,pre_test_Time)



