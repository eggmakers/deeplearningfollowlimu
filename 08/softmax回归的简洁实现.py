import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


#输出层是一个全连接层
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)


#在损失函数中传递未归一化的预测，同时计算softmax及其对数
loss = nn.CrossEntropyLoss()
#使用学习率为0.1的小批量随机梯度作为优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


#训练模型
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()