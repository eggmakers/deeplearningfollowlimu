import torch
from torch import nn
from d2l import torch as d2l


#生成以前一样的一些数据
n_train, n_test, num_inputs, batch_size = 50, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=True)


#初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


#定义范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

#l1范数惩罚
def l1_penalty(w):
    return torch.sum(torch.abs(w))


#训练函数
def train(lambd, penalty=2):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            if penalty == 1:
                l = loss(net(X), y) + lambd * l1_penalty(w)
            if penalty == 2:
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


#忽略正则化直接训练
train(lambd=0)
d2l.plt.show()


#使用权重衰减
train(lambd=10, penalty=1)
d2l.plt.show()