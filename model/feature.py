import numpy as np
import pandas as pd
from data_path import *
import codecs

EVT_SIZE = 595#默认的EVT表识别的个数，在groupby中会重新计算并更新
TCH_SIZE = 3
# def manage_the_data(original_data):
#     #处理原始数据，分割为id click_number(点击次数) EVT_LBL(模块名称) Most_TCH_TYP(最多的点击方式)
#     the_information = []
#     new_data = original_data.apply()

#     return the_information
    
def change_the_EVT_LBL(string_data):
    #去除EVT_LBL中的‘-’
    # new_RVT_LBL = string_data.replace('-','')
    #只保留最后一个数据作为特征值
    new_RVT_LBL = int(string_data.split('-')[-1])
    return new_RVT_LBL

# def creat_EVT_vector(data):
#     #创建EVT_VERTOR向量
#     all_the_information = []

#     # data = pd.read_csv(train_log)
#     # data = pd.read_csv(file_name)
#     evt_data_deal = set(data['EVT_LBL'])#set是一个集合，利用集合的特点，不能有重复而达到快速去除重复
#     for evt in evt_data_deal:
#         tem = dict()
#         # tem['USRID']=usrid
#         tem['EVT_LBL'] = int(evt.split('-')[-1])
#         tem['Orignal_EVT'] = evt
#         # print(type(int(evt.split('-').pop())))
#         # break
#         all_the_information.append(tem)

#     all_data = pd.DataFrame(all_the_information).sort_values(by=['Orignal_EVT'])
#     # all_data.to_csv(pre_train_EVT_Name_table,index=False,columns=['Orignal_EVT','EVT_LBL'])
#     # print(all_data.iloc[:,0].size)
#     # orginal_evt = set(all_data['Orignal_EVT'])
#     # # print(len(EVT_vetor))
#     # # print(EVT_vetor)
#     return all_data

def EVT_Name(size):
    name = []
    for i in range(size):
        name.append('EVT'+str(i))
    return name

    
def data_groupby(data_file_name):
    """
    作用：
    将groupby的数据进行处理，和拆分

    拆分方式：usrid - EVT(采用) - Click_Num - TCH_TYP(形成一个x-y-z的一个点表示app-web-h5的各个的点击次数)
    """
    data = pd.read_csv(data_file_name)
    evalute_table = pd.read_csv(pre_all_evt_evalute)#注意，第一遍用train中的table取处理，得出概率后在用总的处理test
    # list_userID = []
    all_the_information = []
    EVT_Vector_list = np.array(evalute_table['EVT_LBL']).tolist()
    global EVT_SIZE
    EVT_SIZE = len(EVT_Vector_list)#总共有多少的EVT标识,修改全局变量
    # print(EVT_SIZE)

    groups = data.groupby('USRID')['EVT_LBL','OCC_TIM','TCH_TYP']
    EVT_Zero_Vector = np.zeros((len(groups),EVT_SIZE),dtype=np.int)#构建0矩阵用于数据填充
    TCH_TYP_Zero_Vector = np.zeros((len(groups),TCH_SIZE),dtype=np.int)#共计3列，每列为APP，web，h5
    
    # print(EVT_Zero_Vector.shape)
    i = 0#记录当前行 
    #排序特征
    for uesid,group in groups:
        
        # list_TYP = []
        app_x = 0
        web_y = 0
        h5_z = 0

        tem = dict()
        num = 0

        for evt,tch in zip(group['EVT_LBL'],group['TCH_TYP']):
            # print(evt)
            # print(tch)
            evt_number = change_the_EVT_LBL(evt)
            print(evt_number)
            index_evt = EVT_Vector_list.index(evt_number)
            EVT_Zero_Vector[i,index_evt] += 1
            # print(index_evt,evt_number,sep='|')
            num += 1
            if tch==0:
                TCH_TYP_Zero_Vector[i,0] += 1#app
            elif tch==1:
                TCH_TYP_Zero_Vector[i,1] += 1#web
            elif tch==2:
                TCH_TYP_Zero_Vector[i,2] += 1#h5
            

        tem['USRID'] = uesid
        tem['Click_Num'] = num
        # print(num)
        # if(num == 9):
        #     print(list_TYP)
        #     break
        all_the_information.append(tem)
        i += 1
    first_df = pd.DataFrame(all_the_information,columns=['USRID','Click_Num'])
    EVT_df = pd.DataFrame(EVT_Zero_Vector,columns=EVT_Name(EVT_SIZE))
    TCH_df = pd.DataFrame(TCH_TYP_Zero_Vector,columns=['APP','WEB','H5'])

    new_df = first_df.join(EVT_df).join(TCH_df)
    # print(first_df)
    # new_df = pd.merge(first_df,EVT_df)
    # new_df = pd.merge(new_df,TCH_df)
    # print(type(new_df))
    # print(new_df[0,0])
    print('..............finish')
    return new_df

# def 

def split_to_csv(data,all_log_vertor,EVT_file_name,TCH_file_name):
    """
    保存特征到文件
    """
    # data.to_csv(pre_train_log,index=False,columns=['USRID','EVT_LBL','Click_Num','TCH_TYP'])
    list_name = ['USRID','Click_Num']
    list_name.extend(EVT_Name(EVT_SIZE))
    # print(len(list_name))
    # data.to_csv(pre_train_log_EVT,index=False,columns=list_name)
    # data.to_csv(pre_train_log_TCH,index=False,columns=['USRID','Click_Num','APP','WEB','H5'])
    data.to_csv(all_test_log_vertor,index=False)
    data.to_csv(EVT_file_name,index=False,columns=list_name)
    data.to_csv(TCH_file_name,index=False,columns=['USRID','Click_Num','APP','WEB','H5'])
    print('-----finish')


if __name__ == '__main__':
    # print(change_the_EVT_LBL('163-577-913'))
    # print(type(change_the_EVT_LBL('163-577-913')))
    # data = pd.read_csv(train_log)
    # creat_EVT_vector(data)

    # print(type(data))
    # a = data_groupby(data)
    # print(type(a))
    
    # print(data)
    # new_group_data = data.groupby('USRID')['EVT_LBL','OCC_TIM','TCH_TYP'].apply(data_groupby)
    # new_group_data = data_groupby(data.groupby('USRID')['EVT_LBL','OCC_TIM','TCH_TYP'])
    # data_groupby(data.groupby('USRID')['EVT_LBL','OCC_TIM','TCH_TYP'])
    # print(type(data.groupby('USRID')['EVT_LBL','OCC_TIM','TCH_TYP']))
    # print(new_group_data[['EVT_LBL','Click_Num']][:1])
    # split_to_csv(new_group_data)
    # print(dict(list(new_group_data)))
    
    # print(pd.DataFrame(new_group_data,columns=['USRID','EVT_LBL','OCC_TIM','TCH_TYP']))
    # print(creat_EVT_vector(pd.read_csv(pre_train_log_EVT)))
    # creat_EVT_vector()



    # data = pd.read_csv(all_train_log_vertor)
    # split_to_csv(data)
    # list = ['USRID','Click_Num']
    # list.extend(EVT_Name(EVT_SIZE))
    # list = EVT_Name(EVT_SIZE)
    # print(list)

    # 
    data = data_groupby(test_log)
    split_to_csv(data,all_test_log_vertor,pre_test_log_EVT,pre_test_log_TCH)

    # data = data_groupby(train_log)
    # split_to_csv(data,all_train_log_vertor,pre_train_log_EVT,pre_train_log_TCH)

    # print(EVT_SIZE)
    # data = pd.read_csv(test_log)
    # creat_EVT_vector(data)

