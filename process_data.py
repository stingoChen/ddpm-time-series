import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *
from sklearn import preprocessing

path = "./electricity1.csv"
my_datafiles = pd.read_csv(path)
my_datafiles = my_datafiles[:26208]

my_datas = my_datafiles["OT"].values

#
# my_data_split = int(len(my_datas) / step)

my_data_list = []
st = 0
while st <= len(my_datas)-step*2:
    my_data_list.append(my_datas[st: st + step * 2])
    st += 24

# # my_data_list = my_data_list[:-1]
# print(my_data_list)

Train_dataset = np.array(my_data_list)
scaler = preprocessing.MinMaxScaler()
_Train_dataset = scaler.fit_transform(Train_dataset)

Train_dataset = _Train_dataset[:-1]
print("Number of dataset is : ", len(Train_dataset))
# x1x2x2x3         -> xn  [168, 168]
Test_dataset = np.array(_Train_dataset[-1, :step]).reshape(1, -1)
print(len(Test_dataset[0]))
# Test_dataset_1 = np.array(Train_dataset[-1, 144:]).reshape(1, -1)
# print(Test_dataset)
