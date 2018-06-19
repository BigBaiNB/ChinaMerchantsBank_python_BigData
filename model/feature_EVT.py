import numpy as np
import pandas as pd
from data_path import *

def reply_numb(all_data):
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
data = pd.read_csv(train_log)['EVT_LBL']
reply_numb(data)
