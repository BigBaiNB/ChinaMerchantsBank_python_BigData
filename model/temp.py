import numpy as np
import pandas as pd
from pandas import DataFrame
from data_path import *

train_x = pd.read_csv(pre_temp_train_x,nrows=1).columns
test_x = pd.read_csv(pre_test_x,nrows=1).columns
print(list(set(train_x) ^ set(test_x)))
