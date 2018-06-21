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

def change_to_evaluate(Evt_table_file,pre_log_Evt_file,result_file):
    """
    降维：将EVT变为一个分值
    得出用户的evt得分
    """
    # EVT_table = pd.read_csv(pre_train_EVT_table)
    # EVT_user = pd.read_csv(pre_train_log_EVT)

    EVT_table = pd.read_csv(Evt_table_file)
    EVT_user = pd.read_csv(pre_log_Evt_file)
    new_EVT_score = []
    # print(len(EVT_user))
    i = 0#记录行号
    for userid in EVT_user['USRID']:#h行，
        tem = dict()
        Score = 0
        for a in range(EVT_table.iloc[:,0].size):#列
            Score += EVT_user.iloc[i,a+2]*EVT_table.iloc[a,1]
        tem['USRID']=userid
        tem['Evalute']=Score
        new_EVT_score.append(tem)
        i += 1
    pd.DataFrame(new_EVT_score).to_csv(result_file,index=False)


def split_the_evt_lbl(string_evt):
    """
    拆分每个evt_lbl
    """
    x = 0
    list_string = string_evt.split('-')
    A = list_string[0]
    C = list_string[2]

    return A,C

def get_new_evt_evaluate(new_evt,orig_evt):
    A,C = split_the_evt_lbl(new_evt)
    
    evalute_up = 0.0
    evalute_down = 0.0
    evalute = 0.0
    have_up = False
    have_down = False
    i = 0 #记录行号
    for evt in orig_evt['Orignal_EVT']:
        OA,OC = split_the_evt_lbl(evt)
        if OA == A:
            iC = int(C)
            iOC = int(OC)
            if iOC>iC:
                have_up = True
                evalute_up = orig_evt.at[i,'Evaluate']
                break
            if iOC<iC:
                have_down = True
                evalute_down = orig_evt.at[i,'Evaluate']
                

    if have_down and have_up:
        evalute = (evalute_down + evalute_up)/2
    elif have_up:
        evalute = evalute_up
    elif have_down:
        evalute = evalute_down
    
    return evalute,have_up or have_down
        


def deal_test_evt():
    """
    处理test中的evt数据
    得出所有模块的得分
    """
    test_data_all = pd.read_csv(test_log)
    evt_evalute_table = pd.read_csv(pre_train_EVT_table)

    # evt_table = np.array(evt_evalute_table['EVT_LBL']).tolist()

    evt_test_data = set(test_data_all['EVT_LBL'])
    evt_train_data = set(evt_evalute_table['Orignal_EVT'])

    without_evt = evt_test_data - evt_train_data #在test中，但是，没有在train中的evt

    # print([x for x in without_evt])
    with_out_information = []
    for evt in without_evt:
        tem = dict()
        evalute,check = get_new_evt_evaluate(evt,evt_evalute_table)
        if check :
            tem['Evaluate'] = evalute
        else:
            print(evt)
            tem['Evaluate'] = 0.0
        tem['Orignal_EVT'] = evt
        tem['EVT_LBL'] = evt.split('-')[-1]
        with_out_information.append(tem)
    pd.DataFrame(with_out_information).append(evt_evalute_table).sort_values(by=['Orignal_EVT']).to_csv(pre_all_evt_evalute,index=False)

        

        
# data = pd.read_csv(train_log)['EVT_LBL']
# reply_numb(data)
# 
# print(pd.read_csv(pre_EVT_table).as_matrix)

# all_evt = pd.read_csv(pre_train_log_EVT)
# all_flg = pd.read_csv(train_flg)
# all_information = pd.merge(all_evt,all_flg,how='left',on='USRID')
# print(all_information['FLAG'])

# change_to_evaluate()

# table = pd.read_csv(pre_train_EVT_Name_table)
# pd.DataFrame({'Evaluate':get_score()}).join(table).to_csv(pre_train_EVT_table,index=False)
# print(EVT_Orignal = pd.read_csv(train_log,sep=',',index_col=2))

# s = 'a,q,w'
# print(type(s.split(',')))
# print( 1>3 or 1<3)

# deal_test_evt()

change_to_evaluate(pre_all_evt_evalute,pre_test_log_EVT,pre_test_UserID_EVT_Evalute)
print('-----finh')

# A,B=get_new_evt_evaluate('520-1836-3683',pd.read_csv(pre_train_EVT_table))
# print(A)
# print(B)