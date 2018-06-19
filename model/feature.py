import numpy as np
import pandas as pd
from data_path import *
import codecs

def manage_the_data(original_data):
    #处理原始数据，分割为id click_number(点击次数) EVT_LBL(模块名称) Most_TCH_TYP(最多的点击方式)
    the_information = []
    new_data = original_data.apply()

    return the_information
    
def change_the_EVT_LBL(string_data):
    #去除EVT_LBL中的‘-’
    # new_RVT_LBL = string_data.replace('-','')
    new_RVT_LBL = string_data.split('-')
    return new_RVT_LBL



def data_groupby(groups):
    """
    作用：
    将groupby的数据进行处理，和拆分

    拆分方式：usrid - EVT(采用) - Click_Num - TCH_TYP(形成一个x-y-z的一个点表示app-web-h5的各个的点击次数)
    """
    # list_userID = []
    all_the_information = []
    #排序特征
    for uesid,group in groups:
        list_EVT = []
        # list_TYP = []
        app_x = 0
        web_y = 0
        h5_z = 0

        tem = dict()
        num = 0
        tem['USRID'] = uesid

        for evt,tch in zip(group['EVT_LBL'],group['TCH_TYP']):
            # print(evt)
            # print(tch)
            list_EVT += change_the_EVT_LBL(evt)
            num += 1
            if evt==0:
                app_x += 1
            elif evt==1:
                web_y += 1
            elif evt==2:
                h5_z += 1
       
        tem['EVT_LBL'] = list_EVT
        tem['Click_Num'] = num
        # print(num)
        # if(num == 9):
        #     print(list_TYP)
        #     break
        
        tem['TCH_TYP'] = [app_x,web_y,h5_z]
        all_the_information.append(tem)
    return pd.DataFrame(all_the_information,columns=['USRID','EVT_LBL','Click_Num','TCH_TYP'])

def split_to_csv(data):
    """
    保存特征到文件
    """
    data.to_csv(pre_train_log,index=False,columns=['USRID','EVT_LBL','Click_Num','TCH_TYP'])
    data[['USRID','EVT_LBL','Click_Num']].to_csv(pre_train_log_EVT,index=False,columns=['USRID','EVT_LBL','Click_Num'])
    data[['USRID','TCH_TYP','Click_Num']].to_csv(pre_train_log_TCH,index=False,columns=['USRID','TCH_TYP','Click_Num'])
    print('-----finish')


if __name__ == '__main__':
    print(change_the_EVT_LBL('163-577-913'))
    print(type(change_the_EVT_LBL('163-577-913')))
    # data = pd.read_csv(train_log)
    # print(type(data))
    # a = data_groupby(data)
    # print(type(a))
    
    # print(data)
    # new_group_data = data.groupby('USRID')['EVT_LBL','OCC_TIM','TCH_TYP'].apply(data_groupby)
    # new_group_data = data_groupby(data.groupby('USRID')['EVT_LBL','OCC_TIM','TCH_TYP'])
    # print(new_group_data['TCH_TYP'][:1])
    # split_to_csv(new_group_data)
    # print(dict(list(new_group_data)))
    
    # print(pd.DataFrame(new_group_data,columns=['USRID','EVT_LBL','OCC_TIM','TCH_TYP']))

    


