import torch

x = torch.arange(4.0)
print(x)


#存放梯度
x.requires_grad_(True)
print(x.grad)
y = 2 * torch.dot(x, x)
print(y)


#反向传播
y.backward()
print(y.grad)
print(x.grad == 4 * x)


#清除累积的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)


#求批量中每个样本单独计算的偏导数和
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)


#将某些计算移动到记录的计算图之外
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)


#通过python控制流，我们仍然可以计算得到的变量的梯度
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)
