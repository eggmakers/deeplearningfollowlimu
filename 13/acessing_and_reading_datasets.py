import warnings
import numpy as np
import pandas as pd
from download_data import *
import torch

warnings.filterwarnings(action='ignore')
import paddle
from paddle import nn

warnings.filterwarnings("ignore", category=DeprecationWarning)
from d2l import paddle as d2l


#使用pandas读入并处理数据
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(train_data)
print(test_data)


#前四个和最后两个特征，以及相应标签
print(train_data.iloc[0 : 4, [0, 1, 2, 3, -3, -2, -1]])


#在每个样本中，第一个特征是ID，要删除
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


#替换缺失的值为相应特征的平均值。通过将特征重新缩放到零均值和单位方差来标准化数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)


#处理离散值，用一次独热编码替换他们
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)


#从pandas格式中提取Numpy格式，并将其转换为张量表示
n_train = train_data.shape[0]
train_features = paddle.to_tensor(all_features[:n_train].values, dtype=paddle.float32)
test_features = paddle.to_tensor(all_features[n_train:].values, dtype=paddle.float32)
train_labels = paddle.to_tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=paddle.float32)


#训练
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net


#用价格预测的对数来衡量差异
def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


#借助Adam优化器训练函数
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate*1.0,
                                      parameters=net.parameters(),
                                      weight_decay=weight_decay*1.0)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            optimizer.clear_grad()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

#K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = paddle.concat([X_train, X_part], 0)
            y_train = paddle.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


#返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


#模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
d2l.plt.show()