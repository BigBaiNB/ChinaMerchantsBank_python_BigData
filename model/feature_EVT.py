import numpy as np
import pandas as pd
from data_path import *
#导入lasso的相关包
from sklearn.linear_model import LogisticRegression

def reply_numb(all_data):
    #判断是否能将EVT中低3个数作为特征
    #返回最后一类的个数
    data_dic = dict()
    check_dic = dict()
    num = 0
    for string_data in all_data:
        list_data = string_data.split('-')
        last_string = str(list_data.pop())
        check_string = str(list_data[0])+str(list_data[1])

        # if(data_dic.has_key(last_string)):#判断字典中是否含有索引
        if(last_string in data_dic.keys()):
            if(data_dic[last_string] != check_string):
                check_dic[last_string] = 'FALSE'
                print(last_string)
        
        else:
            data_dic[last_string] = check_string
            check_dic[last_string] = 'TRUE'
            num += 1
    print(num)
    # print(check_dic)

def get_score():
    """
    获得每个板块的成交的可能性，作为评分标准
    """
    all_evt = pd.read_csv(pre_train_log_EVT)
    all_flg = pd.read_csv(train_flg)
    all_information = pd.merge(all_evt,all_flg,how='left',on='USRID')
    # all_information.to_csv(pre_train_log_EVT,index=False)

    # print(all_information[:10])
    # print(all_evt.columns.size)
    # name = []

    Click_number = all_information['Click_Num']
    Click_sum = 0#点击总数
    for i in Click_number:
        Click_sum += i

    all_flag_1_sum = 0#总共flage为1的总数
    # for i in all_information['FLAG']:
    #     all_flag_1_sum += i

    evaluate_table = []#评分表

    for i in range(all_evt.columns.size-2):
        # evaluate = dict()#评分
        find_name = 'EVT'+str(i)#检索列名
        EVT = all_information[find_name]
        FLAG = all_information['FLAG']
        evt_click_sum = 0#当前类型evt被点击的总次数
        flag_1_sum = 0#当前类型evt所产生的购买次数的总数

        P = 0.0

        for evt_click,flag_result in zip(EVT,FLAG):
            if(evt_click != 0):
                evt_click_sum += evt_click
                # print(ev)
                if(flag_result == 1):
                    flag_1_sum += evt_click
                    
        P = flag_1_sum / evt_click_sum
        # evaluate[find_name] = P
        evaluate_table.append(P)
        # print(evt_click_sum)
        # print(flag_1_sum)
        # print(type(P))
        # break

            
        
        # print(Y)
        # model_logic = LogisticRegression(C=0.5,solver='liblinear')
        # model_logic.fit(X,Y)

        # result_1 = model_logic.predict(1)
        # evaluate[find_name] = model_logic.predict(1)
        # print(result_1)
        # break
    return evaluate_table

def change_to_evaluate():
    """
    降维：将EVT变为一个分值
    """
    EVT_table = pd.read_csv(pre_EVT_table)
    EVT_user = pd.read_csv(pre_train_log_EVT)
    new_EVT_score = []
    # print(len(EVT_user))
    for i in range(len(EVT_user)):#h行，
        Score = 0
        for a in range(len(EVT_table)):#列
            EVT_user.iloc[i,a+2]*
            
        

        
# data = pd.read_csv(train_log)['EVT_LBL']
# reply_numb(data)
pd.DataFrame(get_score()).to_csv(pre_EVT_table,index=False,header=False)
# print(pd.read_csv(pre_EVT_table).as_matrix)

# all_evt = pd.read_csv(pre_train_log_EVT)
# all_flg = pd.read_csv(train_flg)
# all_information = pd.merge(all_evt,all_flg,how='left',on='USRID')
# print(all_information['FLAG'])
change_to_evaluate()