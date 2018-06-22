import numpy as np
import pandas as pd
from data_path import *

'''
sort_value()参数

by : str or list of str
Name or list of names to sort by.
if axis is 0 or ‘index’ then by may contain index levels and/or column labels
if axis is 1 or ‘columns’ then by may contain column levels and/or index labels
Changed in version 0.23.0: Allow specifying index or column level names.
axis : {0 or ‘index’, 1 or ‘columns’}, default 0
Axis to be sorted
ascending : bool or list of bool, default True
Sort ascending vs. descending. Specify list for multiple sort orders. If this is a list of bools, must match the length of the by.
inplace : bool, default False
if True, perform operation in-place
kind : {‘quicksort’, ‘mergesort’, ‘heapsort’}, default ‘quicksort’
Choice of sorting algorithm. See also ndarray.np.sort for more information. mergesort is the only stable algorithm. For DataFrames, this option is only applied when sorting on a single column or label.
na_position : {‘first’, ‘last’}, default ‘last’
first puts NaNs at the beginning, last puts NaNs at the end
Returns:	
sorted_obj : DataFrame
'''
# pd.read_csv(original_test_agg,sep='\t').sort_values(by=['USRID']).to_csv(test_agg,index=False)
# pd.read_csv(original_test_log,sep='\t').sort_values(by=['USRID','OCC_TIM']).to_csv(test_log,index=False)
# pd.read_csv(original_train_agg,sep='\t').sort_values(by=['USRID']).to_csv(train_agg,index=False)
# pd.read_csv(original_train_flg,sep='\t').sort_values(by=['USRID']).to_csv(train_flg,index=False)
pd.read_csv(original_train_log,sep='\t').sort_values(by=['USRID','OCC_TIM']).to_csv(train_log,index=False)
