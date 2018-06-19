import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

from data_path import *

def micri_avg_f1(y_true,y_pred):
    """
    评分函数
    """
    return f1_score(y_true,y_pred,average='micro')

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
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=.6,random_state=0 )

train_way = LogisticRegression(C=0.5,solver='liblinear')
train_way.fit(x_train,y_train)

y_pred = train_way.predict_proba(x_test)#按概率输出

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