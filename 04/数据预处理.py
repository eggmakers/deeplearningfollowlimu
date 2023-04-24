#创建一个人工数据集，存在csv文件中
import os 
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n')
    f.write('NA, Pave, 127500\n')
    f.write('2, NA, 106000\n')
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000\n')


#提取原始数据集
import pandas as pd

data = pd.read_csv(data_file)
print(data)


#为了处理缺失的数据，进行插值和删除
inputs, outputs = data.iloc[:, 0 : 2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())#填充缺失的值
print(inputs)
print('-------------------------------------')
print(outputs)
#string的处理
inputs = pd.get_dummies(inputs)
print(inputs)


#转换为张量格式
import torch
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)


#Q & A
#reshape和arange的区别
a = torch.arange(12)
b = a.reshape((3, 4))
b[:] = 2
print(a)


