import torch
import os

x = torch.arange(12)

print(x)


#访问张量的形状
print(x.shape)


#访问总数
print(x.numel())


#改变形状但不改变元素数量和数值
X = x.reshape(3, 4)
print(X)


#创建全0或全1
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))


#包含特定的值
x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(x)
print(x.shape)


#算术运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)


#连接张量
x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((x, y), dim=0))
print(torch.cat((x, y), dim=1))


#逻辑运算
print(x == y)


#求和
print(x.sum())


#广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)


#访问元素
print(x[-1])
print(x[1 : 3])


#写入元素
x[1, 2] = 9
print(x)


#多个元素赋值
x[0 : 2, :] = 12
print(x)


#分配内存
before = id(y)
y = y + x
print(id(y) == before)


#原地操作
z = torch.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))


#如果后续计算没有重复使用x，可以使用x[:] = x + y或x += y来减少内存的开销
before = id(x)
x += y
print(id(x) == before)


#转换为numpy张量
A = x.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))


#numpy
a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))

#practice
#1.运行本节中的代码。将本节中的条件语句`X == Y`更改为`X < Y`或`X > Y`，然后看看你可以得到什么样的张量。
print(x < y, '\n',  x > y)


#2. 用其他形状（例如三维张量）替换广播机制中按元素操作的两个张量。结果是否与预期相同？
a = torch.arange(3).reshape((1, 3, 1))
b = torch.arange(2).reshape((1, 1, 2))
print(a)
print(b)
print(a + b)