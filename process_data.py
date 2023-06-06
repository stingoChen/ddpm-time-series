import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *
from sklearn import preprocessing


my_datafiles = pd.read_csv(path)
my_datas = my_datafiles["GHI"].values
my_data_split = int(len(my_datas) / step)

my_data_list = []
for data_split in range(my_data_split - 1):
    my_data_ = my_datas[data_split * step: (data_split + 2) * step]
    my_data_list.append(my_data_)

Train_dataset = np.array(my_data_list)
scaler = preprocessing.MinMaxScaler()
Train_dataset = scaler.fit_transform(Train_dataset)
if len(Train_dataset) % 2 == 1:
    Train_dataset = Train_dataset[:-1]
print("Number of dataset is : ", len(Train_dataset))

Test_dataset = np.array(Train_dataset[-1, 144:]).reshape(1, -1)
# Test_dataset_1 = np.array(Train_dataset[-1, 144:]).reshape(1, -1)
