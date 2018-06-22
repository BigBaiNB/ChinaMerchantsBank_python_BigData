import numpy as np
import pandas as pd
from data_path import *

from datetime import datetime,timedelta
import time


# print((end_date - start_date).total_seconds())

def get_seconds(first_time,second_time):
    #获取时间中，两个时间的差距的秒数
    start_date = datetime.strptime(first_time, "%Y-%m-%d %H:%S:%M")
    end_date = datetime.strptime(second_time, "%Y-%m-%d %H:%S:%M")
    return (end_date - start_date).total_seconds()

def get_all_user_time_to_seconds(file_name,out_file_name):
    """
    将用户的时间分割成，每
    """
    data_all = pd.read_csv(file_name)
    date_time_all = data_all['OCC_TIM']

    groups = data_all.groupby('USRID')['OCC_TIM']

    all_information = []

    for usrid,time_group in groups:
        tem = dict()
        tem['USRID'] = usrid
        click_number = []#记录一个用户，一小时点击频率的所有数
        click = 1 #一小时点击次数
        first_in = True #是否是第一个时间进入
        first_time = ''#第一个时间


        for time in time_group:
            
            if first_in :
                first_time = time
                first_in = False
                click = 1 #一小时点击次数
                click_number.append(click)
            else:
                click_check = int(get_seconds(first_time,time)/3600)#判断是否是属于一个时间段
                if click_check > 0:
                    # first_in = True
                    click_number[-1] = click
                    first_time = time
                    click = 1 #一小时点击次数
                    click_number.append(click)
                else:
                    click += 1
        f = len(click_number)#总频数
        click_number.sort()

        new_click_number = set(click_number)#用于遍历类别

        weight = 0
        for i in new_click_number:
            weight += click_number.count(i)*i
        
        tem['Time'] = weight / f
        all_information.append(tem)
    pd.DataFrame(all_information).to_csv(out_file_name,index=False)

get_all_user_time_to_seconds(test_log,pre_test_Time)
